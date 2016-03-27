#include "ITMVisualisationEngine.h"
#include "ITMPixelUtils.h"
#include "ITMCUDAUtils.h"

#include "ITMRepresentationAccess.h"
#include "ITMLibDefines.h"
#include "ITMSceneReconstructionEngine.h"

using namespace ITMLib::Engine;



#ifndef FAR_AWAY
#define FAR_AWAY 999999.9f
#endif

#ifndef VERY_CLOSE
#define VERY_CLOSE 0.05f
#endif

static const CONSTPTR(int) renderingBlockSizeX = 16;
static const CONSTPTR(int) renderingBlockSizeY = 16;

static const CONSTPTR(int) MAX_RENDERING_BLOCKS = 65536 * 4;
//static const int MAX_RENDERING_BLOCKS = 16384;
static const CONSTPTR(int) minmaximg_subsample = 8;

/**
Project visible blocks into the desired image.

Compute the bounding box (upperLeft, lowerRight, zRange) of the
projection of all eight corners in image space and store the minimum
and maximum Z coordinates of the block in the camera coordinate
system
*/
CPU_AND_GPU inline bool ProjectSingleBlock(
    const THREADPTR(Vector3s) & blockPos,
    const THREADPTR(Matrix4f) & pose,
    const THREADPTR(Vector4f) & intrinsics,
    const THREADPTR(Vector2i) & imgSize,
    float voxelSize,
    THREADPTR(Vector2i) & upperLeft, //!< [out]
    THREADPTR(Vector2i) & lowerRight,  //!< [out]
    THREADPTR(Vector2f) & zRange //!< [out]
    )
{
    const Vector2i minmaxImgSize = imgSize / minmaximg_subsample;
    upperLeft = minmaxImgSize;
    lowerRight = Vector2i(-1, -1);
    // zMin, zmax
    zRange = Vector2f(FAR_AWAY, VERY_CLOSE);

    // project all 8 corners down to 2D image
    for (int corner = 0; corner < 8; ++corner)
    {
        Vector3s tmp = blockPos;
        tmp.x += (corner & 1) ? 1 : 0;
        tmp.y += (corner & 2) ? 1 : 0;
        tmp.z += (corner & 4) ? 1 : 0;
        Vector4f pt3d(TO_FLOAT3(tmp) * (float)SDF_BLOCK_SIZE * voxelSize, 1.0f);
        pt3d = pose * pt3d;

        Vector2f pt2d;
        if (!projectNoBounds(intrinsics, pt3d, pt2d)) continue;
        pt2d /= minmaximg_subsample;

        // remember bounding box, zmin and zmax
        if (upperLeft.x > floor(pt2d.x)) upperLeft.x = (int)floor(pt2d.x);
        if (lowerRight.x < ceil(pt2d.x)) lowerRight.x = (int)ceil(pt2d.x);
        if (upperLeft.y > floor(pt2d.y)) upperLeft.y = (int)floor(pt2d.y);
        if (lowerRight.y < ceil(pt2d.y)) lowerRight.y = (int)ceil(pt2d.y);
        if (zRange.x > pt3d.z) zRange.x = pt3d.z;
        if (zRange.y < pt3d.z) zRange.y = pt3d.z;
    }

    // do some sanity checks and respect image bounds
    if (upperLeft.x < 0) upperLeft.x = 0;
    if (upperLeft.y < 0) upperLeft.y = 0;
    if (lowerRight.x >= minmaxImgSize.x) lowerRight.x = minmaxImgSize.x - 1;
    if (lowerRight.y >= minmaxImgSize.y) lowerRight.y = minmaxImgSize.y - 1;
    if (upperLeft.x > lowerRight.x) return false;
    if (upperLeft.y > lowerRight.y) return false;
    //if (zRange.y <= VERY_CLOSE) return false; never seems to happen
    if (zRange.x < VERY_CLOSE) zRange.x = VERY_CLOSE;
    if (zRange.y < VERY_CLOSE) return false;

    return true;
}

/**
Split image-depth space bounding box described by (upperLeft, lowerRight, zRange)
into (renderingBlockSizeX by renderingBlockSizeY) pixel (or less) RenderingBlocks of same zRange.

Store the resulting blocks into renderingBlockList,
incrementing the current position 'offset' in this list.
*/
CPU_AND_GPU inline void CreateRenderingBlocks(
    DEVICEPTR(RenderingBlock) *renderingBlockList, //!< [out]
    int offset, //!< [out]

    const THREADPTR(Vector2i) & upperLeft,
    const THREADPTR(Vector2i) & lowerRight,
    const THREADPTR(Vector2f) & zRange)
{
    for (int by = 0; by < ceil((float)(1 + lowerRight.y - upperLeft.y) / renderingBlockSizeY); ++by) {
        for (int bx = 0; bx < ceil((float)(1 + lowerRight.x - upperLeft.x) / renderingBlockSizeX); ++bx) {
            // End if list is full.
            if (offset >= MAX_RENDERING_BLOCKS) return;
            //for each rendering block: add it to the list
            DEVICEPTR(RenderingBlock) & b(renderingBlockList[offset++]);

            b.upperLeft.x = upperLeft.x + bx*renderingBlockSizeX;
            b.upperLeft.y = upperLeft.y + by*renderingBlockSizeY;

            // lowerRight corner
            b.lowerRight.x = upperLeft.x + (bx + 1)*renderingBlockSizeX - 1;
            b.lowerRight.y = upperLeft.y + (by + 1)*renderingBlockSizeY - 1;

            // Stay within image bounds (renderingBlockSizeX, renderingBlockSizeY) might not fit
            if (b.lowerRight.x>lowerRight.x) b.lowerRight.x = lowerRight.x;
            if (b.lowerRight.y>lowerRight.y) b.lowerRight.y = lowerRight.y;

            b.zRange = zRange;
        }
    }
}

/// \param x,y [in] camera space pixel determining ray direction
/// \returns whether any intersection was found
CPU_AND_GPU inline bool castRay(
    DEVICEPTR(Vector4f) &pt_out, //!< [out] the intersection point. w is 1 for a valid point, 0 for no intersection; in voxel-fractional-world-coordinates

    const int x, const int y,
    const CONSTPTR(ITMVoxelBlock) *voxelData,
    const CONSTPTR(typename ITMVoxelBlockHash::IndexData) *voxelIndex,
    const Matrix4f invM, //!< camera-to-world transform
    const Vector4f invProjParams, //!< camera-to-world transform
    const float oneOverVoxelSize,
    const float mu,
    const CONSTPTR(Vector2f) & viewFrustum_minmax)
{
    Vector4f pt_camera_f;
    Vector3f pt_block_s, pt_block_e;

    float totalLength;


    // Starting point
    pt_camera_f = depthTo3DInvProjParams(invProjParams, x, y, viewFrustum_minmax.x);
    // Lengths given in voxel-fractional-coordinates (such that one voxel has size 1)
    totalLength = length(TO_VECTOR3(pt_camera_f)) * oneOverVoxelSize;
    // in voxel-fractional-world-coordinates (such that one voxel has size 1)
    pt_block_s = TO_VECTOR3(invM * pt_camera_f) * oneOverVoxelSize;

    // End point
    pt_camera_f = depthTo3DInvProjParams(invProjParams, x, y, viewFrustum_minmax.y);
    const float totalLengthMax = length(TO_VECTOR3(pt_camera_f)) * oneOverVoxelSize;
    pt_block_e = TO_VECTOR3(invM * pt_camera_f) * oneOverVoxelSize;


    // Raymarching
    const Vector3f rayDirection = normalize(pt_block_e - pt_block_s);
    Vector3f pt_result = pt_block_s; // Current position in voxel-fractional-world-coordinates
    const float stepScale = mu * oneOverVoxelSize;
    typename ITMVoxelBlockHash::IndexCache cache;
    float sdfValue = 1.0f;
    bool hash_found;
    float stepLength;
    while (totalLength < totalLengthMax) {
        // D(X)
        sdfValue = readFromSDF_float_uninterpolated(voxelData, voxelIndex, pt_result, hash_found, cache);

        if (!hash_found) {
            //  First we try to find an allocated voxel block, and the length of the steps we take is determined by the block size
            stepLength = SDF_BLOCK_SIZE;
        }
        else {
            // If we found an allocated block, 
            // [Once we are inside the truncation band], the values from the SDF give us conservative step lengths.

            // using trilinear interpolation only if we have read values in the range −0.5 ≤ D(X) ≤ 0.1
            if ((sdfValue <= 0.1f) && (sdfValue >= -0.5f)) {
                sdfValue = readFromSDF_float_interpolated(voxelData, voxelIndex, pt_result, hash_found, cache);
            }
            // once we read a negative value from the SDF, we found the intersection with the surface.
            if (sdfValue <= 0.0f) break;

            stepLength = MAX(
                sdfValue * stepScale,
                1.0f // if we are outside the truncation band µ, our step size is determined by the truncation band 
                // (note that the distance is normalized to lie in [-1,1] within the truncation band)
                );
        }

        pt_result += stepLength * rayDirection;
        totalLength += stepLength;
    }

    bool pt_found;
    //  If the T - SDF value is negative after such a trilinear interpolation, the surface
    //  has indeed been found and we terminate the ray, performing one last
    //  trilinear interpolation step for a smoother appearance.
    if (sdfValue <= 0.0f)
    {
        // Refine position
        stepLength = sdfValue * stepScale;
        pt_result += stepLength * rayDirection;

        // Read again
        sdfValue = readFromSDF_float_interpolated(voxelData, voxelIndex, pt_result, hash_found, cache);
        // Refine position
        stepLength = sdfValue * stepScale;
        pt_result += stepLength * rayDirection;

        pt_found = true;
    }
    else pt_found = false;

    pt_out = Vector4f(pt_result, (pt_found) ? 1.0f : 0.0f);

    return pt_found;
}

/// Compute normal in the distance field via the gradient.
/// c.f. computeSingleNormalFromSDF
CPU_AND_GPU inline void computeNormalAndAngle(
    THREADPTR(bool) & foundPoint, //!< in,out
    const THREADPTR(Vector3f) & point,
    const CONSTPTR(ITMVoxelBlock) *voxelBlockData,
    const CONSTPTR(typename ITMVoxelBlockHash::IndexData) *indexData,
    const THREADPTR(Vector3f) & lightSource,
    THREADPTR(Vector3f) & outNormal,
    THREADPTR(float) & angle //!< outNormal . lightSource
    )
{
    if (!foundPoint) return;

    outNormal = normalize(computeSingleNormalFromSDF(voxelBlockData, indexData, point));

    angle = dot(outNormal, lightSource);
    if (!(angle > 0.0)) foundPoint = false;
}

/**
Computing the surface normal in image space given raycasted image.

In image space, since the normals are computed on a regular grid,
there are only 4 uninterpolated read operations followed by a cross-product.

\returns normal_out[idx].w = sigmaZ_out[idx] = -1 on error where idx = x + y * imgDims.x
*/
template <bool useSmoothing>
CPU_AND_GPU inline void computeNormalAndAngle(
    THREADPTR(bool) & foundPoint, //!< in,out. Set to false when the normal cannot be computed
    const THREADPTR(int) &x, const THREADPTR(int) &y,
    const CONSTPTR(Vector4f) *pointsRay,
    const THREADPTR(Vector3f) & lightSource,
    const THREADPTR(float) &voxelSize,
    const THREADPTR(Vector2i) &imgSize,
    THREADPTR(Vector3f) & outNormal,
    THREADPTR(float) & angle //!< outNormal . lightSource
    )
{
    if (!foundPoint) return;

    // Lookup world coordinates of points surrounding (x,y)
    // and compute forward difference vectors
    Vector4f xp1_y, xm1_y, x_yp1, x_ym1;
    Vector4f diff_x(0.0f, 0.0f, 0.0f, 0.0f), diff_y(0.0f, 0.0f, 0.0f, 0.0f);

    // If useSmoothing, use positions 2 away
    int extraDelta = useSmoothing ? 1 : 0;

#define d(x) (x + extraDelta)

    if (y <= d(1) || y >= imgSize.y - d(2) || x <= d(1) || x >= imgSize.x - d(2)) { foundPoint = false; return; }

#define lookupNeighbors() \
    xp1_y = sampleNearest(pointsRay, x + d(1), y, imgSize);\
    x_yp1 = sampleNearest(pointsRay, x, y + d(1), imgSize);\
    xm1_y = sampleNearest(pointsRay, x - d(1), y, imgSize);\
    x_ym1 = sampleNearest(pointsRay, x, y - d(1), imgSize);\
    diff_x = xp1_y - xm1_y;\
    diff_y = x_yp1 - x_ym1;

    lookupNeighbors();

#define isAnyPointIllegal() (xp1_y.w <= 0 || x_yp1.w <= 0 || xm1_y.w <= 0 || x_ym1.w <= 0)

    float length_diff = MAX(length2(diff_x.toVector3()), length2(diff_y.toVector3()));
    bool lengthDiffTooLarge = (length_diff * voxelSize * voxelSize > (0.15f * 0.15f));

    if (isAnyPointIllegal() || (lengthDiffTooLarge && useSmoothing)) {
        if (!useSmoothing) { foundPoint = false; return; }

        // In case we used smoothing, try again without extra delta 
        extraDelta = 0;
        lookupNeighbors();

        if (isAnyPointIllegal()){ foundPoint = false; return; }
    }

#undef d
#undef isAnyPointIllegal
#undef lookupNeighbors

    // TODO why the extra minus?
    outNormal = normalize(-cross(diff_x.toVector3(), diff_y.toVector3()));

    angle = dot(outNormal, lightSource);
    if (!(angle > 0.0)) foundPoint = false;
}




#define DRAWFUNCTIONPARAMS \
DEVICEPTR(Vector4u) & dest,\
const CONSTPTR(Vector3f) & point, /* in voxel-fractional world coordinates, comes from raycastResult*/\
const CONSTPTR(ITMVoxelBlock) *voxelBlockData, \
const CONSTPTR(typename ITMVoxelBlockHash::IndexData) *indexData,\
const THREADPTR(Vector3f) & normal_obj,\
const THREADPTR(float) & angle

// PIXEL SHADERS
// " Finally a coloured or shaded rendering of the surface is trivially computed, as desired for the visualisation."
CPU_AND_GPU inline void drawPixelGrey(DRAWFUNCTIONPARAMS)
{
    float outRes = (0.8f * angle + 0.2f) * 255.0f;
    dest = Vector4u((uchar)outRes);
}

CPU_AND_GPU inline void drawPixelNormal(DRAWFUNCTIONPARAMS)
{
    dest.r = (uchar)((0.3f + (-normal_obj.r + 1.0f)*0.35f)*255.0f);
    dest.g = (uchar)((0.3f + (-normal_obj.g + 1.0f)*0.35f)*255.0f);
    dest.b = (uchar)((0.3f + (-normal_obj.b + 1.0f)*0.35f)*255.0f);
}

CPU_AND_GPU inline void drawPixelColour(DRAWFUNCTIONPARAMS)
{
    Vector3f clr = readFromSDF_color4u_interpolated(voxelBlockData, indexData, point);
    dest = Vector4u(TO_UCHAR3(clr), 255); 
}

#define PROCESS_AND_DRAW_PIXEL(PROCESSFUNCTION, DRAWFUNCTION) \
CPU_AND_GPU inline void PROCESSFUNCTION(DEVICEPTR(Vector4u) &outRendering, const CONSTPTR(Vector3f) & point,\
    bool foundPoint, const CONSTPTR(ITMVoxelBlock) *voxelData, const CONSTPTR(typename ITMVoxelBlockHash::IndexData) *voxelIndex,\
	Vector3f lightSource) {\
	Vector3f outNormal;\
	float angle;\
    computeNormalAndAngle(foundPoint, point, voxelData, voxelIndex, lightSource, outNormal, angle);\
    if (foundPoint) DRAWFUNCTION(outRendering, point, voxelData, voxelIndex, outNormal, angle);\
    	else outRendering = Vector4u((uchar)0);\
}

PROCESS_AND_DRAW_PIXEL(processPixelColour, drawPixelColour)
PROCESS_AND_DRAW_PIXEL(processPixelGrey, drawPixelGrey)
PROCESS_AND_DRAW_PIXEL(processPixelNormal, drawPixelNormal)


CPU_AND_GPU inline void processPixelICPPost(
const float angle,
const Vector3f outNormal,
DEVICEPTR(Vector4f) &pointsMap, //<! [out] trackingState->pointCloud->locations (world space conversion of point)
DEVICEPTR(Vector4f) &normalsMap,
const THREADPTR(Vector3f) & point, //<! [in] renderState->raycastResult, in voxel-fractional-world-coordinates!
const bool foundPoint,
const float voxelSize)
{

    if (!foundPoint)
    {
        pointsMap = normalsMap = IllegalColor<Vector4f>::make();
        return;
    }

    pointsMap = Vector4f(point * voxelSize, 1);
    normalsMap = Vector4f(outNormal, 0);
}

/**
Produces a shaded image (outRendering) and a point cloud for e.g. tracking.
Uses image space normals.
*/
/// \param useSmoothing whether to compute normals by forward differences two pixels away (true) or just one pixel away (false)
template<bool useSmoothing>
CPU_AND_GPU inline void processPixelICP(
    DEVICEPTR(Vector4f) *const pointsMap, //!< [out] receives output points in world coordinates
    DEVICEPTR(Vector4f) *const normalsMap,

    const CONSTPTR(Vector4f) *pointsRay, //!< [in] points in voxel-fractional-world-coordinates (renderState->raycastResult)
    const THREADPTR(Vector2i) &imgSize,
    const THREADPTR(int) &x,
    const THREADPTR(int) &y,
    const float voxelSize,
    const THREADPTR(Vector3f) &lightSource)
{
    Vector3f outNormal;
    float angle;

    int locId = pixelLocId(x, y, imgSize);
    Vector4f point = pointsRay[locId];

    bool foundPoint = point.w > 0.0f;

    computeNormalAndAngle<useSmoothing>(foundPoint, x, y, pointsRay, lightSource, voxelSize, imgSize, outNormal, angle);

    processPixelICPPost(
        angle, outNormal,
        pointsMap[locId],
        normalsMap[locId],
        point.toVector3(),
        foundPoint,
        voxelSize);
}

/// as val goes from x0 to x1, output goes from y0 to y1 linearly
inline float interpolate(float val, float y0, float x0, float y1, float x1) {
	return (val - x0)*(y1 - y0) / (x1 - x0) + y0;
}

/**
1   ---
0__/   \___
where the angles are at
-.75, -.25, .25, .75
*/
inline float base(float val) {
	if (val <= -0.75f) return 0.0f;
	else if (val <= -0.25f) return interpolate(val, 0.0f, -0.75f, 1.0f, -0.25f);
	else if (val <= 0.25f) return 1.0f;
	else if (val <= 0.75f) return interpolate(val, 1.0f, 0.25f, 0.0f, 0.75f);
	else return 0.0;
}

void ITMVisualisationEngine::DepthToUchar4(ITMUChar4Image *dst, const ITMFloatImage *src)
{
    dst->Clear();
    Vector4u * const dest = dst->GetData(MEMORYDEVICE_CPU);
	float const * const source = src->GetData(MEMORYDEVICE_CPU);
	const int dataSize = static_cast<int>(dst->dataSize);

    // lims =  #@source & /@ {Min, Max}
	float lims[2];
	lims[0] = 100000.0f; lims[1] = -100000.0f;

	for (int idx = 0; idx < dataSize; idx++)
	{
		float sourceVal = source[idx];
		if (sourceVal > 0.0f) { lims[0] = MIN(lims[0], sourceVal); lims[1] = MAX(lims[1], sourceVal); }
	}
	if (lims[0] == lims[1]) return;

    // Rescaled rgb-converted depth
    const float scale = 1.0f / (lims[1] - lims[0]);
	for (int idx = 0; idx < dataSize; idx++)
	{
		float sourceVal = source[idx];

        if (sourceVal <= 0.0f) continue;
		sourceVal = (sourceVal - lims[0]) * scale;

        dest[idx].r = (uchar)(base(sourceVal - 0.5f) * 255.0f); // shows the range 0 to 1.25
		dest[idx].g = (uchar)(base(sourceVal) * 255.0f); // shows the range 0 to .75
		dest[idx].b = (uchar)(base(sourceVal + 0.5f) * 255.0f); // shows the range 
		dest[idx].a = 255;
	}
}


inline dim3 getGridSize(dim3 taskSize, dim3 blockSize)
{
    return dim3((taskSize.x + blockSize.x - 1) / blockSize.x, (taskSize.y + blockSize.y - 1) / blockSize.y, (taskSize.z + blockSize.z - 1) / blockSize.z);
}

inline dim3 getGridSize(Vector2i taskSize, dim3 blockSize) { return getGridSize(dim3(taskSize.x, taskSize.y), blockSize); }

//device implementations

KERNEL projectAndSplitBlocks_device(
    const ITMHashEntry * const hashEntries,
    const ITMVoxelBlock * const localVBA,
    const Matrix4f pose_M,
    const Vector4f intrinsics,
    const Vector2i imgSize,
    const float voxelSize,
    RenderingBlock *renderingBlocks, //!< [out]
    uint *noTotalBlocks //!< [out]
    )
{
    Vector2i upperLeft, lowerRight;
    Vector2f zRange;
    bool validProjection = false;

    const int in_offset = threadIdx.x + blockDim.x * blockIdx.x;

    // ignoring visible list:
    VoxelBlockPos pos = localVBA[in_offset].pos;
    if (pos != INVALID_VOXEL_BLOCK_POS)
        // Shared:
        validProjection = ProjectSingleBlock(
            pos,//blockData.pos, 
            pose_M, intrinsics, imgSize, voxelSize, upperLeft, lowerRight, zRange);

    Vector2i requiredRenderingBlocks(ceilf((float)(lowerRight.x - upperLeft.x + 1) / renderingBlockSizeX),
        ceilf((float)(lowerRight.y - upperLeft.y + 1) / renderingBlockSizeY));

    size_t requiredNumBlocks = requiredRenderingBlocks.x * requiredRenderingBlocks.y;
    if (!validProjection) requiredNumBlocks = 0;

    int out_offset = computePrefixSum_device<uint>(requiredNumBlocks, noTotalBlocks, blockDim.x, threadIdx.x);
    if (!validProjection) return;
    if ((out_offset == -1) || (out_offset + requiredNumBlocks > MAX_RENDERING_BLOCKS)) return;

    CreateRenderingBlocks(renderingBlocks, out_offset, upperLeft, lowerRight, zRange);
}

KERNEL fillBlocks_device(const uint *noTotalBlocks, const RenderingBlock *renderingBlocks,
    Vector2i imgSize,
    Vector2f *minmaxData //!< [out]
    )
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    int block = blockIdx.x * 4 + blockIdx.y;
    if (block >= *noTotalBlocks) return;

    const RenderingBlock & b(renderingBlocks[block]);
    int xpos = b.upperLeft.x + x;
    if (xpos > b.lowerRight.x) return;
    int ypos = b.upperLeft.y + y;
    if (ypos > b.lowerRight.y) return;

    Vector2f & pixel(minmaxData[xpos + ypos*imgSize.x]);
    atomicMin(&pixel.x, b.zRange.x); atomicMax(&pixel.y, b.zRange.y);
}

KERNEL genericRaycast_device(Vector4f *out_ptsRay, const ITMVoxelBlock *voxelData, const typename ITMVoxelBlockHash::IndexData *voxelIndex,
    Vector2i imgSize, Matrix4f invM, Vector4f invProjParams, float oneOverVoxelSize, const Vector2f *minmaximg, float mu)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

    if (x >= imgSize.x || y >= imgSize.y) return;

    int locId = x + y * imgSize.x;
    int locId2 = (int)floor((float)x / minmaximg_subsample) + (int)floor((float)y / minmaximg_subsample) * imgSize.x;

    castRay(out_ptsRay[locId], x, y, voxelData, voxelIndex, invM, invProjParams, oneOverVoxelSize, mu, minmaximg[locId2]);
}

KERNEL renderICP_device(Vector4f *pointsMap, Vector4f *normalsMap, const Vector4f *pointsRay,
    float voxelSize, Vector2i imgSize, Vector3f lightSource)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

    if (x >= imgSize.x || y >= imgSize.y) return;

    processPixelICP<true>(pointsMap, normalsMap, pointsRay, imgSize, x, y, voxelSize, lightSource);
}

/*
renderGrey_device, processPixelGrey
renderColourFromNormal_device, processPixelNormal
renderColour_device, processPixelColour
*/
#define RENDER_PROCESS_PIXEL(RENDERFUN, PROCESSPIXELFUN) \
KERNEL RENDERFUN ## _device(Vector4u *outRendering, const Vector4f *ptsRay, const ITMVoxelBlock *voxelData,\
    const typename ITMVoxelBlockHash::IndexData *voxelIndex, Vector2i imgSize, Vector3f lightSource) { \
    int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);\
    if (x >= imgSize.x || y >= imgSize.y) return;\
    int locId = pixelLocId(x, y, imgSize);\
    Vector4f ptRay = ptsRay[locId];\
    PROCESSPIXELFUN(outRendering[locId], ptRay.toVector3(), ptRay.w > 0, voxelData, voxelIndex, lightSource);\
}

RENDER_PROCESS_PIXEL(renderGrey, processPixelGrey)
RENDER_PROCESS_PIXEL(renderColourFromNormal, processPixelNormal)
RENDER_PROCESS_PIXEL(renderColour, processPixelColour)

// class implementation
ITMVisualisationEngine::ITMVisualisationEngine(ITMScene *scene) : scene(scene)
{
    cudaSafeCall(cudaMalloc((void**)&renderingBlockList_device, sizeof(RenderingBlock) * MAX_RENDERING_BLOCKS));
    cudaSafeCall(cudaMalloc((void**)&noTotalBlocks_device, sizeof(uint)));
}

ITMVisualisationEngine::~ITMVisualisationEngine(void)
{
    cudaSafeCall(cudaFree(noTotalBlocks_device));
    cudaSafeCall(cudaFree(renderingBlockList_device));
}

ITMRenderState* ITMVisualisationEngine::CreateRenderState(const Vector2i & imgSize) const
{
    return new ITMRenderState(
        imgSize
        );
}

void ITMVisualisationEngine::CreateExpectedDepths(
    const ITMPose *pose, const ITMIntrinsics *intrinsics,
    ITMRenderState *renderState) const
{
    const float voxelSize = this->scene->sceneParams->voxelSize;

    Vector2i imgSize = renderState->renderingRangeImage->noDims;

    //go through list of voxel blocks, create rendering blocks storing min and max depth in that range
    const ITMHashEntry *hash_entries = this->scene->index.GetEntries();
    {
        dim3 blockSize(256);
        dim3 gridSize((int)ceil((float)SDF_LOCAL_BLOCK_NUM / (float)blockSize.x));

        cudaSafeCall(cudaMemset(noTotalBlocks_device, 0, sizeof(uint)));

        projectAndSplitBlocks_device << <gridSize, blockSize >> >(
            hash_entries,
            scene->localVBA.GetVoxelBlocks(),
            pose->GetM(),
            intrinsics->projectionParamsSimple.all, imgSize, voxelSize,

            renderingBlockList_device, noTotalBlocks_device);
    }
    uint noTotalBlocks;
    cudaSafeCall(cudaMemcpy(&noTotalBlocks, noTotalBlocks_device, sizeof(uint), cudaMemcpyDeviceToHost));
    if (noTotalBlocks > (unsigned)MAX_RENDERING_BLOCKS) noTotalBlocks = MAX_RENDERING_BLOCKS;

    // go through rendering blocks and fill minmaxData
    Vector2f * const minmaxData = renderState->renderingRangeImage->GetData(MEMORYDEVICE_CUDA);
    // 1. reset
    memsetKernel<Vector2f>(minmaxData, Vector2f(FAR_AWAY, VERY_CLOSE), renderState->renderingRangeImage->dataSize);

    // 2. copy from rendering blocks
    dim3 blockSize(16, 16);
    dim3 gridSize((unsigned int)ceil((float)noTotalBlocks / 4.0f), 4);
    fillBlocks_device << <gridSize, blockSize >> >(noTotalBlocks_device, renderingBlockList_device, imgSize, minmaxData);
}

/// uses renderingRangeImage, creates raycastResult
static void GenericRaycast(
    const ITMScene *const scene,
    const Vector2i& imgSize,
    const Matrix4f& invM,
    const Vector4f projParams, 
    ITMRenderState *const renderState //!< [in, out] uses renderingRangeImage, creates raycastResult
    )
{
    const float voxelSize = scene->sceneParams->voxelSize;
    const float oneOverVoxelSize = 1.0f / voxelSize;

    // for speedup (?)
    Vector4f invProjParams(1.0f / projParams.x, 1.0f / projParams.y, projParams.z, projParams.w);

    dim3 cudaBlockSize(16, 12);
    dim3 gridSize((int)ceil((float)imgSize.x / (float)cudaBlockSize.x), (int)ceil((float)imgSize.y / (float)cudaBlockSize.y));
    genericRaycast_device << <gridSize, cudaBlockSize >> >(
        renderState->raycastResult->GetData(MEMORYDEVICE_CUDA),
        scene->localVBA.GetVoxelBlocks(),
        scene->index.GetEntries(),
        imgSize,
        invM,
        invProjParams,
        oneOverVoxelSize,
        renderState->renderingRangeImage->GetData(MEMORYDEVICE_CUDA),
        scene->sceneParams->mu
        );
}

static void RenderImage_common(
    const ITMScene *const scene,
    const ITMPose *const pose,
    const ITMIntrinsics *const intrinsics,
    ITMRenderState *const renderState,
    ITMUChar4Image *const outputImage,
    const ITMVisualisationEngine::RenderImageType type)
{
    Vector2i imgSize = outputImage->noDims;
    Matrix4f invM = pose->GetInvM();

    GenericRaycast(scene, imgSize, invM, intrinsics->projectionParamsSimple.all, renderState);

    Vector3f lightSource = -Vector3f(invM.getColumn(2));
    Vector4u *outRendering = outputImage->GetData(MEMORYDEVICE_CUDA);
    Vector4f *pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CUDA);

    dim3 cudaBlockSize(8, 8);
    dim3 gridSize((int)ceil((float)imgSize.x / (float)cudaBlockSize.x), (int)ceil((float)imgSize.y / (float)cudaBlockSize.y));

    switch (type) {
    case ITMVisualisationEngine::RENDER_COLOUR_FROM_VOLUME:
        renderColour_device << <gridSize, cudaBlockSize >> >(outRendering, pointsRay, scene->localVBA.GetVoxelBlocks(),
            scene->index.GetEntries(), imgSize, lightSource);
        break;
    case ITMVisualisationEngine::RENDER_COLOUR_FROM_NORMAL:
        renderColourFromNormal_device << <gridSize, cudaBlockSize >> >(outRendering, pointsRay, scene->localVBA.GetVoxelBlocks(),
            scene->index.GetEntries(), imgSize, lightSource);
        break;
    case ITMVisualisationEngine::RENDER_SHADED_GREYSCALE:
    default:
        renderGrey_device << <gridSize, cudaBlockSize >> >(outRendering, pointsRay, scene->localVBA.GetVoxelBlocks(),
            scene->index.GetEntries(), imgSize, lightSource);
        break;
    }
}

void CreateICPMaps_common(const ITMScene *scene, Vector4f intrinsics_d, ITMTrackingState *trackingState, ITMRenderState *renderState)
{
    Vector2i imgSize = renderState->raycastResult->noDims;
    Matrix4f invM = trackingState->pose_d->GetInvM();

    GenericRaycast(scene, imgSize, invM, intrinsics_d, renderState);

    // Remember the pose from which this point cloud was rendered
    trackingState->pointCloud->pose_pointCloud->SetFrom(trackingState->pose_d);

    Vector4f *pointsMap = trackingState->pointCloud->locations->GetData(MEMORYDEVICE_CUDA);
    Vector4f *normalsMap = trackingState->pointCloud->normals->GetData(MEMORYDEVICE_CUDA);
    Vector4f *pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CUDA);

    Vector3f lightSource = -Vector3f(invM.getColumn(2));

    dim3 cudaBlockSize(16, 12);
    dim3 gridSize((int)ceil((float)imgSize.x / (float)cudaBlockSize.x), (int)ceil((float)imgSize.y / (float)cudaBlockSize.y));
    renderICP_device << <gridSize, cudaBlockSize >> >(pointsMap, normalsMap, pointsRay,
        scene->sceneParams->voxelSize, imgSize, lightSource);
}

void ITMVisualisationEngine::RenderImage(const ITMPose *pose, const ITMIntrinsics *intrinsics,
    ITMRenderState *renderState, ITMUChar4Image *outputImage, ITMVisualisationEngine::RenderImageType type) const
{
    CreateExpectedDepths(pose, intrinsics, renderState);
    RenderImage_common(this->scene, pose, intrinsics, renderState, outputImage, type);
}

void ITMVisualisationEngine::CreateICPMaps(
    const ITMIntrinsics * const intrinsics_d,
    ITMTrackingState *const trackingState,
    ITMRenderState *const renderStateTemp) const
{
    CreateExpectedDepths(trackingState->pose_d, intrinsics_d, renderStateTemp);
    CreateICPMaps_common(this->scene, intrinsics_d->projectionParamsSimple.all, trackingState, renderStateTemp);
}

