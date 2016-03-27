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

// reduce passing and renaming of recurring variables using globals
static __managed__ Matrix4f pose_M; //!< world-to-camera transform
static __managed__ Matrix4f invPose_M; //!< camera-to-world transform
static __managed__ Vector4f projParams;
static __managed__ Vector4f invProjParams;

static __managed__ Vector2i imgSize;

static __managed__ DEVICEPTR(RenderingBlock*) renderingBlocks = 0; // valid from 0 to noTotalBlocks-1
static __managed__ uint noTotalBlocks = 0;
static __managed__ DEVICEPTR(Vector2f *) renderingRangeImage = 0; //  = renderState->renderingRangeImage, minmaxData
static __managed__ DEVICEPTR(Vector4f *) raycastResult = 0; //  = renderState->raycastResult

// for ICP
static __managed__ DEVICEPTR(Vector4f) * pointsMap = 0; //!< [out] receives output points in world coordinates
static __managed__ DEVICEPTR(Vector4f) * normalsMap = 0;

// for RenderImage
static __managed__ DEVICEPTR(Vector4u) * outRendering = 0; //;outputImage->GetData(MEMORYDEVICE_CUDA);
static __managed__ Vector3f towardsCamera;

// === create expected depths/rendering range image ===

/**
Project visible blocks into the desired image.

Compute the bounding box (upperLeft, lowerRight, zRange) of the
projection of all eight corners in image space and store the minimum
and maximum Z coordinates of the block in the camera coordinate
system
*/
GPU_ONLY inline bool ProjectSingleBlock(
    const THREADPTR(Vector3s) & blockPos,
    THREADPTR(Vector2i) & upperLeft, //!< [out]
    THREADPTR(Vector2i) & lowerRight,  //!< [out]
    THREADPTR(Vector2f) & zRange //!< [out]
    )
{
    upperLeft = imgSize;
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
        pt3d = pose_M * pt3d;

        Vector2f pt2d;
        if (!projectNoBounds(projParams, pt3d, pt2d)) continue;

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
    if (lowerRight.x >= imgSize.x) lowerRight.x = imgSize.x - 1;
    if (lowerRight.y >= imgSize.y) lowerRight.y = imgSize.y - 1;
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
GPU_ONLY inline void CreateRenderingBlocks(
    int offset, //!< writes to renderingBlocks starting at offset

    const THREADPTR(Vector2i) & upperLeft,
    const THREADPTR(Vector2i) & lowerRight,
    const THREADPTR(Vector2f) & zRange)
{
    for (int by = 0; by < ceil((float)(1 + lowerRight.y - upperLeft.y) / renderingBlockSizeY); ++by) {
        for (int bx = 0; bx < ceil((float)(1 + lowerRight.x - upperLeft.x) / renderingBlockSizeX); ++bx) {
            // End if list is full.
            if (offset >= MAX_RENDERING_BLOCKS) return;
            //for each rendering block: add it to the list
            DEVICEPTR(RenderingBlock) & b(renderingBlocks[offset++]);

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


struct ProjectAndSplitBlock {
    static GPU_ONLY void process(ITMVoxelBlock* vb) {
        Vector2i upperLeft, lowerRight;
        Vector2f zRange;
        bool validProjection = false;
        VoxelBlockPos pos = vb->pos;

        // Find projection rectangle
        validProjection = ProjectSingleBlock(pos,
            upperLeft, lowerRight, zRange);

        // Split into blocks
        Vector2i requiredRenderingBlocks(
            ceilf((float)(lowerRight.x - upperLeft.x + 1) / renderingBlockSizeX),
            ceilf((float)(lowerRight.y - upperLeft.y + 1) / renderingBlockSizeY)
            );

        size_t requiredNumBlocks = requiredRenderingBlocks.x * requiredRenderingBlocks.y;

        if (!validProjection) requiredNumBlocks = 0;

        int out_offset = computePrefixSum_device<uint>(requiredNumBlocks, &noTotalBlocks, blockDim.x, threadIdx.x);
        if (!validProjection) return;
        if ((out_offset == -1) || (out_offset + requiredNumBlocks > MAX_RENDERING_BLOCKS)) return;

        CreateRenderingBlocks(out_offset, upperLeft, lowerRight, zRange);
    }
};


KERNEL fillBlocks_device()
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    int block = blockIdx.x;
    if (block >= noTotalBlocks) return;

    const RenderingBlock & b(renderingBlocks[block]);
    int xpos = b.upperLeft.x + x;
    if (xpos > b.lowerRight.x) return;
    int ypos = b.upperLeft.y + y;
    if (ypos > b.lowerRight.y) return;

    Vector2f & pixel(renderingRangeImage[xpos + ypos*imgSize.x]);
    atomicMin(&pixel.x, b.zRange.x);
    atomicMax(&pixel.y, b.zRange.y);
}


void ITMVisualisationEngine::CreateExpectedDepths() {
    // Reset rendering block list
    cudaDeviceSynchronize(); // want to use current scene etc.
    ::renderingBlocks = renderingBlockList_device;
    noTotalBlocks = 0;

    // go through list of voxel blocks, create rendering blocks storing min and max depth in that range
    Scene::getCurrentScene()->doForEachAllocatedVoxelBlock<ProjectAndSplitBlock>();
    cudaDeviceSynchronize(); // want to read imgSize

    // go through rendering blocks and fill minmaxData
    // 1. reset
    memsetKernel<Vector2f>(renderingRangeImage, Vector2f(FAR_AWAY, VERY_CLOSE), imgSize.x * imgSize.y);

    cudaDeviceSynchronize(); // want to read noTotalBlocks
    // 2. copy from rendering blocks
    dim3 blockSize(renderingBlockSizeX, renderingBlockSizeY);
    LAUNCH_KERNEL(fillBlocks_device, noTotalBlocks, blockSize);
}
// === raycasting, rendering ===
/// \param x,y [in] camera space pixel determining ray direction
/// \returns whether any intersection was found
GPU_ONLY inline bool castRay(
    DEVICEPTR(Vector4f) &pt_out, //!< [out] the intersection point. w is 1 for a valid point, 0 for no intersection; in voxel-fractional-world-coordinates

    const int x, const int y,
    const CONSTPTR(Vector2f) & viewFrustum_minmax //!< determines line segment (together with ray direction) on which the first intersection is searched
    )
{
    Vector4f pt_camera_f;
    Vector3f pt_block_s, pt_block_e;

    float totalLength;


    // Starting point
    pt_camera_f = depthTo3DInvProjParams(invProjParams, x, y, viewFrustum_minmax.x);
    // Lengths given in voxel-fractional-coordinates (such that one voxel has size 1)
    totalLength = length(TO_VECTOR3(pt_camera_f)) * oneOverVoxelSize;
    // in voxel-fractional-world-coordinates (such that one voxel has size 1)
    pt_block_s = TO_VECTOR3(invPose_M * pt_camera_f) * oneOverVoxelSize;

    // End point
    pt_camera_f = depthTo3DInvProjParams(invProjParams, x, y, viewFrustum_minmax.y);
    const float totalLengthMax = length(TO_VECTOR3(pt_camera_f)) * oneOverVoxelSize;
    pt_block_e = TO_VECTOR3(invPose_M * pt_camera_f) * oneOverVoxelSize;


    // Raymarching
    const Vector3f rayDirection = normalize(pt_block_e - pt_block_s);
    Vector3f pt_result = pt_block_s; // Current position in voxel-fractional-world-coordinates
    const float stepScale = mu * oneOverVoxelSize;
    // TODO use caching, we will access the same voxel block multiple times
    float sdfValue = 1.0f;
    bool hash_found;

    // in voxel-fractional-world-coordinates (1.0f means step one voxel)
    float stepLength;

    while (totalLength < totalLengthMax) {
        // D(X)
        sdfValue = readFromSDF_float_uninterpolated(pt_result, hash_found);

        if (!hash_found) {
            //  First we try to find an allocated voxel block, and the length of the steps we take is determined by the block size
            stepLength = SDF_BLOCK_SIZE;
        }
        else {
            // If we found an allocated block, 
            // [Once we are inside the truncation band], the values from the SDF give us conservative step lengths.

            // using trilinear interpolation only if we have read values in the range −0.5 ≤ D(X) ≤ 0.1
            if ((sdfValue <= 0.1f) && (sdfValue >= -0.5f)) {
                sdfValue = readFromSDF_float_interpolated(pt_result, hash_found);
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
        sdfValue = readFromSDF_float_interpolated(pt_result, hash_found);
        // Refine position
        stepLength = sdfValue * stepScale;
        pt_result += stepLength * rayDirection;

        pt_found = true;
    }
    else pt_found = false;

    pt_out = Vector4f(pt_result, (pt_found) ? 1.0f : 0.0f);

    return pt_found;
}

// Loop over pixels
KERNEL genericRaycast_device()
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

    if (x >= imgSize.x || y >= imgSize.y) return;

    int locId = x + y * imgSize.x;
    castRay(raycastResult[locId], x, y, renderingRangeImage[locId]);
}


/// Compute normal in the distance field via the gradient.
/// c.f. computeSingleNormalFromSDF
GPU_ONLY inline void computeNormalAndAngle(
    THREADPTR(bool) & foundPoint, //!< in,out
    const THREADPTR(Vector3f) & point, //!< [in]
    const THREADPTR(Vector3f) & towardsCamera,
    THREADPTR(Vector3f) & outNormal,
    THREADPTR(float) & angle //!< outNormal . towardsCamera
    )
{
    if (!foundPoint) return;

    outNormal = normalize(computeSingleNormalFromSDF(point));

    angle = dot(outNormal, towardsCamera);
    // dont consider points not facing the camera (raycast will hit these, do backface culling now)
    if (!(angle > 0.0)) foundPoint = false;
}

/**
Computing the surface normal in image space given raycasted image.

In image space, since the normals are computed on a regular grid,
there are only 4 uninterpolated read operations followed by a cross-product.

\returns normal_out[idx].w = sigmaZ_out[idx] = -1 on error where idx = x + y * imgDims.x
*/
template <bool useSmoothing>
GPU_ONLY inline void computeNormalImageSpace(
    THREADPTR(bool) & foundPoint, //!< [in,out] Set to false when the normal cannot be computed
    const THREADPTR(int) &x, const THREADPTR(int) &y,
    THREADPTR(Vector3f) & outNormal
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
    xp1_y = sampleNearest(raycastResult, x + d(1), y, imgSize);\
    x_yp1 = sampleNearest(raycastResult, x, y + d(1), imgSize);\
    xm1_y = sampleNearest(raycastResult, x - d(1), y, imgSize);\
    x_ym1 = sampleNearest(raycastResult, x, y - d(1), imgSize);\
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

    float angle = dot(outNormal, towardsCamera);
    // dont consider points not facing the camera (raycast will hit these, do backface culling now)
    if (!(angle > 0.0)) foundPoint = false;
}


GPU_ONLY inline void processPixelICPPost(
    const Vector3f outNormal,//<! [in] 
    DEVICEPTR(Vector4f) &pointsMap, //<! [out] trackingState->pointCloud->locations (world space conversion of point)
    DEVICEPTR(Vector4f) &normalsMap,
    const THREADPTR(Vector3f) & point, //<! [in] renderState->raycastResult, in voxel-fractional-world-coordinates!
    const bool foundPoint)
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
GPU_ONLY inline void processPixelICP(
    const THREADPTR(int) &x,
    const THREADPTR(int) &y)
{
    int locId = pixelLocId(x, y, imgSize);
    Vector4f point = raycastResult[locId];

    bool foundPoint = point.w > 0.0f;

    Vector3f outNormal;
    // TODO could we use the world space normals here? not without change
    computeNormalImageSpace<useSmoothing>(
        foundPoint, x, y, outNormal);

    processPixelICPPost(
        outNormal,
        pointsMap[locId],
        normalsMap[locId],
        point.toVector3(),
        foundPoint);
}

KERNEL renderICP_device()
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);
    if (x >= imgSize.x || y >= imgSize.y) return;
    processPixelICP<true>(x, y);
}

// PIXEL SHADERS
// " Finally a coloured or shaded rendering of the surface is trivially computed, as desired for the visualisation."

#define DRAWFUNCTIONPARAMS \
DEVICEPTR(Vector4u) & dest,/* in voxel-fractional world coordinates, comes from raycastResult*/\
const CONSTPTR(Vector3f) & point, /* in voxel-fractional world coordinates, comes from raycastResult*/\
const THREADPTR(Vector3f) & normal_obj,\
const THREADPTR(float) & angle

GPU_ONLY inline void drawPixelGrey(DRAWFUNCTIONPARAMS)
{
    float outRes = (0.8f * angle + 0.2f) * 255.0f;
    dest = Vector4u((uchar)outRes);
}

GPU_ONLY inline void drawPixelNormal(DRAWFUNCTIONPARAMS) {
    dest.r = (uchar)((0.3f + (-normal_obj.r + 1.0f)*0.35f)*255.0f);
    dest.g = (uchar)((0.3f + (-normal_obj.g + 1.0f)*0.35f)*255.0f);
    dest.b = (uchar)((0.3f + (-normal_obj.b + 1.0f)*0.35f)*255.0f);
}

GPU_ONLY inline void drawPixelColour(DRAWFUNCTIONPARAMS) {
    Vector3f clr = readFromSDF_color4u_interpolated(point);
    dest = Vector4u(TO_UCHAR3(clr), 255); 
}

#define PROCESS_AND_DRAW_PIXEL(PROCESSFUNCTION, DRAWFUNCTION) \
GPU_ONLY inline void PROCESSFUNCTION(DEVICEPTR(Vector4u) &outRendering, const CONSTPTR(Vector3f) & point,\
    bool foundPoint) {\
	Vector3f outNormal;\
	float angle;\
    computeNormalAndAngle(foundPoint, point, towardsCamera, outNormal, angle);\
    if (foundPoint) DRAWFUNCTION(outRendering, point, outNormal, angle);\
    else outRendering = Vector4u((uchar)0);\
}

PROCESS_AND_DRAW_PIXEL(processPixelColour, drawPixelColour)
PROCESS_AND_DRAW_PIXEL(processPixelGrey, drawPixelGrey)
PROCESS_AND_DRAW_PIXEL(processPixelNormal, drawPixelNormal)

/*
renderGrey_device, processPixelGrey
renderColourFromNormal_device, processPixelNormal
renderColour_device, processPixelColour

Loop over pixels
*/
#define RENDER_PROCESS_PIXEL(RENDERFUN, PROCESSPIXELFUN) \
KERNEL RENDERFUN ## _device() { \
    int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);\
    if (x >= imgSize.x || y >= imgSize.y) return;\
    int locId = pixelLocId(x, y, imgSize);\
    PROCESSPIXELFUN(outRendering[locId], raycastResult[locId].toVector3(), raycastResult[locId].w > 0);\
}

RENDER_PROCESS_PIXEL(renderGrey, processPixelGrey)
RENDER_PROCESS_PIXEL(renderColourFromNormal, processPixelNormal)
RENDER_PROCESS_PIXEL(renderColour, processPixelColour)

// class implementation
ITMVisualisationEngine::ITMVisualisationEngine() {
    cudaSafeCall(cudaMalloc(&renderingBlockList_device, sizeof(RenderingBlock) * MAX_RENDERING_BLOCKS));
}

ITMVisualisationEngine::~ITMVisualisationEngine(void) {
    cudaSafeCall(cudaFree(renderingBlockList_device));
}

ITMRenderState* ITMVisualisationEngine::CreateRenderState(const Vector2i & imgSize) const {
    return new ITMRenderState(imgSize);
}

inline dim3 getGridSize(dim3 taskSize, dim3 blockSize)
{
    return dim3((taskSize.x + blockSize.x - 1) / blockSize.x, (taskSize.y + blockSize.y - 1) / blockSize.y, (taskSize.z + blockSize.z - 1) / blockSize.z);
}
/// uses renderingRangeImage, creates raycastResult
static void GenericRaycast()
{
    dim3 blockSize(16, 16);
    genericRaycast_device << <getGridSize(dim3(imgSize.x, imgSize.y), blockSize), blockSize >> >();
}

static void RenderImage_common(
    const ITMVisualisationEngine::RenderImageType type)
{
    cudaDeviceSynchronize(); // want to read imgSize -- todo why is this needed?
    dim3 blockSize(8, 8);
    dim3 gridSize = getGridSize(dim3(imgSize.x, imgSize.y), blockSize);

    switch (type) {
    case ITMVisualisationEngine::RENDER_COLOUR_FROM_VOLUME:
        renderColour_device << <gridSize, blockSize >> >();
        break;
    case ITMVisualisationEngine::RENDER_COLOUR_FROM_NORMAL:
        renderColourFromNormal_device << <gridSize, blockSize >> >();
        break;
    case ITMVisualisationEngine::RENDER_SHADED_GREYSCALE:
    default:
        renderGrey_device << <gridSize, blockSize >> >();
        break;
    }
}

void CreateICPMaps_common() {
    cudaDeviceSynchronize(); // want to read imgSize -- todo why is this needed?
    dim3 blockSize(16, 12); // TODO why 12
    renderICP_device << <getGridSize(dim3(imgSize.x, imgSize.y), blockSize), blockSize >> >();
}

/// Initializes raycastResult
void ITMVisualisationEngine::Common(
    const ITMPose *pose,
    const ITMIntrinsics *intrinsics,
    ITMRenderState *renderState) {
    // Set up globals
    renderingRangeImage = renderState->renderingRangeImage->GetData(MEMORYDEVICE_CUDA);
    raycastResult = renderState->raycastResult->GetData(MEMORYDEVICE_CUDA);

    pose_M = pose->GetM();
    invPose_M = pose->GetInvM();

    // (negative camera z axis)
    towardsCamera = -Vector3f(invPose_M.getColumn(2));

    projParams = intrinsics->projectionParamsSimple.all;
    invProjParams = intrinsics->getInverseProjParams();

    imgSize = renderState->renderingRangeImage->noDims;

    CreateExpectedDepths();
    GenericRaycast();
}

void ITMVisualisationEngine::RenderImage(
    const ITMPose *pose,
    const ITMIntrinsics *intrinsics,
    ITMRenderState *renderState,

    ITMUChar4Image *outputImage,
    ITMVisualisationEngine::RenderImageType type)
{
    assert(Scene::getCurrentScene());
    assert(outputImage->noDims == renderState->renderingRangeImage->noDims);
    assert(outputImage->noDims == renderState->raycastResult->noDims); 
    outRendering = outputImage->GetData(MEMORYDEVICE_CUDA);

    Common(pose, intrinsics, renderState);

    RenderImage_common(type);
}

void ITMVisualisationEngine::CreateICPMaps(
    ITMTrackingState * const trackingState, // [in, out] builds trackingState->pointCloud, renders from trackingState->pose_d 
    const ITMIntrinsics * const intrinsics_d,
    ITMRenderState *const renderState //!< [in, out] builds renderingRangeImage for one-time use
    )
{
    assert(Scene::getCurrentScene());
    cudaDeviceSynchronize(); // fix sporadic access violation with ==?

    assert(trackingState->pointCloud->locations->noDims == renderState->renderingRangeImage->noDims);
    assert(trackingState->pointCloud->normals->noDims == renderState->renderingRangeImage->noDims);
    pointsMap = trackingState->pointCloud->locations->GetData(MEMORYDEVICE_CUDA);
    normalsMap = trackingState->pointCloud->normals->GetData(MEMORYDEVICE_CUDA);

    // Remember the pose from which this point cloud was rendered
    trackingState->pointCloud->pose_pointCloud->SetFrom(trackingState->pose_d);

    Common(trackingState->pose_d, intrinsics_d, renderState); 

    CreateICPMaps_common();
}

