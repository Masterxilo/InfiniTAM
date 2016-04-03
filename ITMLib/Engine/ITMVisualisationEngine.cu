#include "ITMVisualisationEngine.h"
#include "ITMPixelUtils.h"
#include "ITMCUDAUtils.h"
#include "ITMRepresentationAccess.h"
#include "ITMLibDefines.h"
#include "ITMSceneReconstructionEngine.h"


// reduce passing and renaming of recurring variables using globals
static __managed__ Matrix4f invPose_M; //!< camera-to-world transform
static __managed__ Vector4f projParams;

static __managed__ Vector2i imgSize;

static __managed__ DEVICEPTR(Vector4f *) raycastResult = 0; //  = renderState->raycastResult

// for ICP
static __managed__ DEVICEPTR(Vector4f) * pointsMap = 0; //!< [out] receives output points in world coordinates
static __managed__ DEVICEPTR(Vector4f) * normalsMap = 0;//!< [out] receives world space normals computed from pointsMap

// for RenderImage
static __managed__ DEVICEPTR(Vector4u) * outRendering = 0; //;outputImage->GetData(MEMORYDEVICE_CUDA);
static __managed__ Vector3f towardsCamera;

// === raycasting, rendering ===
/// \param x,y [in] camera space pixel determining ray direction
//!< [out] raycastResult[locId]: the intersection point. 
// w is 1 for a valid point, 0 for no intersection; in voxel-fractional-world-coordinates
struct castRay {
    forEachPixelNoImage_process()
    {
        // Starting point
        Vector4f pt_camera_f = depthTo3D(projParams, x, y, viewFrustum_min);
        // Lengths given in voxel-fractional-coordinates (such that one voxel has size 1)
        float totalLength = length(TO_VECTOR3(pt_camera_f)) * oneOverVoxelSize;
        // in voxel-fractional-world-coordinates (such that one voxel has size 1)
        const Vector3f pt_block_s = TO_VECTOR3(invPose_M * pt_camera_f) * oneOverVoxelSize;

        // End point
        pt_camera_f = depthTo3D(projParams, x, y, viewFrustum_max);
        const float totalLengthMax = length(TO_VECTOR3(pt_camera_f)) * oneOverVoxelSize;
        const Vector3f pt_block_e = TO_VECTOR3(invPose_M * pt_camera_f) * oneOverVoxelSize;


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

        raycastResult[locId] = Vector4f(pt_result, (pt_found) ? 1.0f : 0.0f);
    }
};

/// Compute normal in the distance field via the gradient.
/// c.f. computeSingleNormalFromSDF
GPU_ONLY inline void computeNormalAndAngle(
    THREADPTR(bool) & foundPoint, //!< [in,out]
    const THREADPTR(Vector3f) & point, //!< [in]
    THREADPTR(Vector3f) & outNormal,//!< [out] 
    THREADPTR(float) & angle //!< [out] outNormal . towardsCamera
    )
{
    if (!foundPoint) return;

    outNormal = normalize(computeSingleNormalFromSDF(point));

    angle = dot(outNormal, towardsCamera);
    // dont consider points not facing the camera (raycast will hit these, do backface culling now)
    if (!(angle > 0.0)) foundPoint = false;
}

/**
Computing the surface normal in image space given raycasted image (raycastResult).

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

#define useSmoothing true

/**
Produces a shaded image (outRendering) and a point cloud for e.g. tracking.
Uses image space normals.
*/
/// \param useSmoothing whether to compute normals by forward differences two pixels away (true) or just one pixel away (false)
struct processPixelICP {
    forEachPixelNoImage_process() {
        const Vector4f point = raycastResult[locId];

        bool foundPoint = point.w > 0.0f;

        Vector3f outNormal;
        // TODO could we use the world space normals here? not without change
        computeNormalImageSpace<useSmoothing>(
            foundPoint, x, y, outNormal);


        if (!foundPoint)
        {
            pointsMap[locId] = normalsMap[locId] = IllegalColor<Vector4f>::make();
            return;
        }

        pointsMap[locId] = Vector4f(point.toVector3() * voxelSize, 1);
        normalsMap[locId] = Vector4f(outNormal, 0);
    }
};

// PIXEL SHADERS
// " Finally a coloured or shaded rendering of the surface is trivially computed, as desired for the visualisation."

#define DRAWFUNCTIONPARAMS \
DEVICEPTR(Vector4u) & dest,/* in voxel-fractional world coordinates, comes from raycastResult*/\
const CONSTPTR(Vector3f) & point, /* in voxel-fractional world coordinates, comes from raycastResult*/\
const THREADPTR(Vector3f) & normal_obj,\
const THREADPTR(float) & angle

GPU_ONLY inline void drawPixelGrey(DRAWFUNCTIONPARAMS)
{
    const float outRes = (0.8f * angle + 0.2f) * 255.0f;
    dest = Vector4u((uchar)outRes);
}

GPU_ONLY inline void drawPixelNormal(DRAWFUNCTIONPARAMS) {
    dest.r = (uchar)((0.3f + (normal_obj.r + 1.0f)*0.35f)*255.0f);
    dest.g = (uchar)((0.3f + (normal_obj.g + 1.0f)*0.35f)*255.0f);
    dest.b = (uchar)((0.3f + (normal_obj.b + 1.0f)*0.35f)*255.0f);
}

GPU_ONLY inline void drawPixelColour(DRAWFUNCTIONPARAMS) {
    const Vector3f clr = readFromSDF_color4u_interpolated(point);
    dest = Vector4u(TO_UCHAR3(clr), 255); 
}

#define PROCESS_AND_DRAW_PIXEL(PROCESSFUNCTION, DRAWFUNCTION) \
struct PROCESSFUNCTION { \
    forEachPixelNoImage_process() {\
        DEVICEPTR(Vector4u) &outRender = outRendering[locId]; \
        const CONSTPTR(Vector3f) point = raycastResult[locId].toVector3(); \
        bool foundPoint = raycastResult[locId].w > 0; \
        \
        Vector3f outNormal; \
        float angle; \
        computeNormalAndAngle(foundPoint, point, outNormal, angle); \
        if (foundPoint) DRAWFUNCTION(outRender, point, outNormal, angle); \
        else outRender = Vector4u((uchar)0); \
    }\
};

PROCESS_AND_DRAW_PIXEL(renderColour, drawPixelColour)
PROCESS_AND_DRAW_PIXEL(renderGrey, drawPixelGrey)
PROCESS_AND_DRAW_PIXEL(renderColourFromNormal, drawPixelNormal)


/// Initializes raycastResult
static void Common(
    const ITMPose *pose,
    const ITMIntrinsics *intrinsics,
    ITMRenderState *renderState) {
    // Set up globals
    raycastResult = renderState->raycastResult->GetData(MEMORYDEVICE_CUDA);

    invPose_M = pose->GetInvM();
    projParams = intrinsics->projectionParamsSimple.all;
    imgSize = renderState->raycastResult->noDims;
    assert(imgSize.area() > 1);

    // (negative camera z axis)
    towardsCamera = -Vector3f(invPose_M.getColumn(2));

    forEachPixelNoImage<castRay>(imgSize);
}


void RenderImage(
    const ITMPose *pose,
    const ITMIntrinsics *intrinsics,
    ITMRenderState *renderState,
    ITMUChar4Image *outputImage,
    std::string shader)
{
    // Check dimensions and set output
    assert(Scene::getCurrentScene());
    assert(outputImage->noDims == renderState->raycastResult->noDims);
    outRendering = outputImage->GetData(MEMORYDEVICE_CUDA);

    Common(pose, intrinsics, renderState);
    cudaDeviceSynchronize(); // want to read imgSize -- todo why is this needed?
#define isShader(s) if (shader == #s) {forEachPixelNoImage<s>(imgSize); return;}
    isShader(renderColour);
    isShader(renderColourFromNormal);
    isShader(renderGrey);
    assert(false); // unkown shader
}

void CreateICPMaps(
    ITMTrackingState * const trackingState, // [in, out] builds trackingState->pointCloud, renders from trackingState->pose_d 
    const ITMIntrinsics * const intrinsics_d,
    ITMRenderState *const renderState //!< [in, out] builds renderingRangeImage for one-time use
    )
{
    assert(Scene::getCurrentScene());
    // Check dimensions and set output
    assert(trackingState->pointCloud->locations->noDims == trackingState->pointCloud->normals->noDims);
    assert(trackingState->pointCloud->normals->noDims == renderState->raycastResult->noDims);
    pointsMap = trackingState->pointCloud->locations->GetData(MEMORYDEVICE_CUDA);
    normalsMap = trackingState->pointCloud->normals->GetData(MEMORYDEVICE_CUDA);

    // Remember the pose from which this point cloud was rendered
    trackingState->pointCloud->pose_pointCloud->SetFrom(trackingState->pose_d);

    Common(trackingState->pose_d, intrinsics_d, renderState);

    // Create ICP maps
    cudaDeviceSynchronize(); // want to read imgSize -- todo why is this needed?
    forEachPixelNoImage<processPixelICP>(imgSize);
}