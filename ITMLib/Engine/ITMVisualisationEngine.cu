﻿#include "ITMVisualisationEngine.h"
#include "ITMPixelUtils.h"
#include "ITMCUDAUtils.h"
#include "ITMLibDefines.h"
#include "ITMSceneReconstructionEngine.h"
#include "ITMRepresentationAccess.h"
/**
the 3D intersection locations generated by the latest raycast
in voxelCoordinates
*/
static __managed__ PointImage* raycastResult;

// for ICP
//!< [out] receives output points in world coordinates
//!< [out] receives world space normals computed from points (image space)
static __managed__ DEVICEPTR(RayImage) * lastFrameICPMap = 0;

// for RenderImage
static __managed__ CameraImage<Vector4u>* outRendering = 0;
static __managed__ Vector3f towardsCamera;

// written by rendering
static __managed__ ITMFloatImage* outDepth;

// === raycasting, rendering ===
/// \param x,y [in] camera space pixel determining ray direction
//!< [out] raycastResult[locId]: the intersection point. 
// w is 1 for a valid point, 0 for no intersection; in voxel-fractional-world-coordinates
struct castRay {
    forEachPixelNoImage_process()
    {
        // Find 3d position of depth pixel xy, in eye coordinates
        auto pt_camera_f = raycastResult->getRayThroughPixel(Vector2i(x, y), viewFrustum_min);
        assert(pt_camera_f.origin.coordinateSystem == raycastResult->eyeCoordinates);
        auto l = pt_camera_f.endpoint().location;
        assert(l.z == viewFrustum_min);

        // Length given in voxel-fractional-coordinates (such that one voxel has size 1)
        auto pt_camera_f_vc = voxelCoordinates->convert(pt_camera_f);
        float totalLength = length(pt_camera_f_vc.direction.direction);
        assert(voxelSize < 1);
        assert(totalLength > length(pt_camera_f.direction.direction));
        assert(abs(
            totalLength - length(pt_camera_f.direction.direction) / voxelSize) < 0.0001f);

        // in voxel-fractional-world-coordinates (such that one voxel has size 1)
        assert(pt_camera_f.endpoint().coordinateSystem == raycastResult->eyeCoordinates);
        assert(!(pt_camera_f_vc.endpoint().coordinateSystem == raycastResult->eyeCoordinates));
        const auto pt_block_s = pt_camera_f_vc.endpoint();

        // End point
        auto pt_camera_e = raycastResult->getRayThroughPixel(Vector2i(x, y), viewFrustum_max);
        auto pt_camera_e_vc = voxelCoordinates->convert(pt_camera_e);
        const float totalLengthMax = length(pt_camera_e_vc.direction.direction);
        const auto pt_block_e = pt_camera_e_vc.endpoint();

        assert(totalLength < totalLengthMax);
        assert(pt_block_s.coordinateSystem == voxelCoordinates);
        assert(pt_block_e.coordinateSystem == voxelCoordinates);

        // Raymarching
        const auto rayDirection = Vector(voxelCoordinates, normalize(pt_block_e.location - pt_block_s.location));
        auto pt_result = pt_block_s; // Current position in voxel-fractional-world-coordinates
        const float stepScale = mu * oneOverVoxelSize; // sdf values are distances in world-coordinates, normalized by division through mu. This is the factor to convert to voxelCoordinates.

        // TODO use caching, we will access the same voxel block multiple times
        float sdfValue = 1.0f;
        bool hash_found;

        // in voxel-fractional-world-coordinates (1.0f means step one voxel)
        float stepLength;

        while (totalLength < totalLengthMax) {
            // D(X)
            sdfValue = readFromSDF_float_uninterpolated(pt_result.location, hash_found);

            if (!hash_found) {
                //  First we try to find an allocated voxel block, and the length of the steps we take is determined by the block size
                stepLength = SDF_BLOCK_SIZE;
            }
            else {
                // If we found an allocated block, 
                // [Once we are inside the truncation band], the values from the SDF give us conservative step lengths.

                // using trilinear interpolation only if we have read values in the range −0.5 ≤ D(X) ≤ 0.1
                if ((sdfValue <= 0.1f) && (sdfValue >= -0.5f)) {
                    sdfValue = readFromSDF_float_interpolated(pt_result.location, hash_found);
                }
                // once we read a negative value from the SDF, we found the intersection with the surface.
                if (sdfValue <= 0.0f) break;

                stepLength = MAX(
                    sdfValue * stepScale,
                    1.0f // if we are outside the truncation band µ, our step size is determined by the truncation band 
                    // (note that the distance is normalized to lie in [-1,1] within the truncation band)
                    );
            }

            pt_result = pt_result + rayDirection * stepLength;
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
            pt_result = pt_result + rayDirection * stepLength;

            // Read again
            sdfValue = readFromSDF_float_interpolated(pt_result.location, hash_found);
            // Refine position
            stepLength = sdfValue * stepScale;
            pt_result = pt_result + rayDirection * stepLength;

            pt_found = true;
        }
        else pt_found = false;

        raycastResult->image->GetData()[locId] = Vector4f(pt_result.location, (pt_found) ? 1.0f : 0.0f);
        assert(raycastResult->pointCoordinates == voxelCoordinates);
        assert(pt_result.coordinateSystem == voxelCoordinates);
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
        DEVICEPTR(Vector4u) &outRender = outRendering->image->GetData()[locId]; \
Point voxelCoordinatePoint = raycastResult->getPointForPixel(Vector2i(x,y));\
assert(voxelCoordinatePoint.coordinateSystem == voxelCoordinates); \
        const CONSTPTR(Vector3f) point = voxelCoordinatePoint.location; \
float& outZ = ::outDepth->GetData()[locId];\
    auto a = outRendering->eyeCoordinates->convert(voxelCoordinatePoint);\
outZ = a.location.z; /* in world / eye coordinates (distance) */ \
        bool foundPoint = raycastResult->image->GetData()[locId].w > 0; \
        \
        Vector3f outNormal; \
        float angle; \
computeNormalAndAngle(foundPoint, point, outNormal, angle); \
        if (foundPoint) {/*assert(outZ >= viewFrustum_min && outZ <= viewFrustum_max); -- approx*/DRAWFUNCTION(outRender, point, outNormal, angle);} \
        else {\
            outRender = Vector4u((uchar)0); outZ = 0;\
        } \
    }\
};

PROCESS_AND_DRAW_PIXEL(renderColour, drawPixelColour)
PROCESS_AND_DRAW_PIXEL(renderGrey, drawPixelGrey)
PROCESS_AND_DRAW_PIXEL(renderColourFromNormal, drawPixelNormal)


/// Initializes raycastResult
static void Common(
const ITMPose *pose,
const ITMIntrinsics *intrinsics,
const Vector2i imgSize
) {
    assert(imgSize.area() > 1);
    auto raycastImage = new ITMFloat4Image(imgSize);
    auto invPose_M = pose->GetInvM();
    auto cameraCs = new CoordinateSystem(invPose_M);
    raycastResult = new PointImage(
        raycastImage,
        voxelCoordinates,

        cameraCs,
        intrinsics->projectionParamsSimple.all
        );

    // (negative camera z axis)
    towardsCamera = -Vector3f(invPose_M.getColumn(2));

    forEachPixelNoImage<castRay>(imgSize);
}

CameraImage<Vector4u>* RenderImage(
    const ITMPose *pose,
    const ITMIntrinsics *intrinsics,
    const Vector2i imgSize,
    ITMFloatImage* const outDepth,
    std::string shader)
{
    assert(imgSize.area() > 1);
    assert(outDepth);
    assert(outDepth->noDims == imgSize);
    ::outDepth = outDepth;

    auto outImage = new ITMUChar4Image(imgSize);
    auto outCs = new CoordinateSystem(pose->GetInvM());
    outRendering = new CameraImage<Vector4u>(
        outImage,
        outCs,
        intrinsics->projectionParamsSimple.all
        );

    Common(pose, intrinsics, outRendering->imgSize());
    cudaDeviceSynchronize(); // want to read imgSize


#define isShader(s) if (shader == #s) {forEachPixelNoImage<s>(outRendering->imgSize());cudaDeviceSynchronize(); return outRendering;}
    isShader(renderColour);
    isShader(renderColourFromNormal);
    isShader(renderGrey);
    assert(false); // unkown shader
    return nullptr;
}

/// Computing the surface normal in image space given raycasted image (raycastResult).
///
/// In image space, since the normals are computed on a regular grid,
/// there are only 4 uninterpolated read operations followed by a cross-product.
/// (here we might do more when useSmoothing is true, and we step 2 pixels wide to find // //further-away neighbors)
///
/// \returns normal_out[idx].w = sigmaZ_out[idx] = -1 on error where idx = x + y * imgDims.x
template <bool useSmoothing>
GPU_ONLY inline void computeNormalImageSpace(
    THREADPTR(bool) & foundPoint, //!< [in,out] Set to false when the normal cannot be computed
    const THREADPTR(int) &x, const THREADPTR(int) &y,
    THREADPTR(Vector3f) & outNormal
    )
{
    if (!foundPoint) return;
    const Vector2i imgSize = raycastResult->imgSize();

    // Lookup world coordinates of points surrounding (x,y)
    // and compute forward difference vectors
    Vector4f xp1_y, xm1_y, x_yp1, x_ym1;
    Vector4f diff_x(0.0f, 0.0f, 0.0f, 0.0f), diff_y(0.0f, 0.0f, 0.0f, 0.0f);

    // If useSmoothing, use positions 2 away
    int extraDelta = useSmoothing ? 1 : 0;

#define d(x) (x + extraDelta)

    if (y <= d(1) || y >= imgSize.y - d(2) || x <= d(1) || x >= imgSize.x - d(2)) { foundPoint = false; return; }

#define lookupNeighbors() \
    xp1_y = sampleNearest(raycastResult->image->GetData(), x + d(1), y, imgSize);\
    x_yp1 = sampleNearest(raycastResult->image->GetData(), x, y + d(1), imgSize);\
    xm1_y = sampleNearest(raycastResult->image->GetData(), x - d(1), y, imgSize);\
    x_ym1 = sampleNearest(raycastResult->image->GetData(), x, y - d(1), imgSize);\
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

    // TODO why the extra minus? -- it probably does not matter because we compute the distance to a plane which would be the same with the inverse normal
    outNormal = normalize(-cross(diff_x.toVector3(), diff_y.toVector3()));

    float angle = dot(outNormal, towardsCamera);
    // dont consider points not facing the camera (raycast will hit these, do backface culling now)
    if (!(angle > 0.0)) foundPoint = false;
}

#define useSmoothing true

static __managed__ RayImage* outIcpMap = 0;
/// Produces a shaded image (outRendering) and a point cloud for e.g. tracking.
/// Uses image space normals.
/// \param useSmoothing whether to compute normals by forward differences two pixels away (true) or just one pixel away (false)
struct processPixelICP {
    forEachPixelNoImage_process() {
        const Vector4f point = raycastResult->image->GetData()[locId];
        assert(raycastResult->pointCoordinates == voxelCoordinates);

        bool foundPoint = point.w > 0.0f;

        Vector3f outNormal;
        // TODO could we use the world space normals here? not without change
        computeNormalImageSpace<useSmoothing>(
            foundPoint, x, y, outNormal);

#define pointsMap outIcpMap->image->GetData()
#define normalsMap outIcpMap->normalImage->GetData()

        if (!foundPoint)
        {
            pointsMap[locId] = normalsMap[locId] = IllegalColor<Vector4f>::make();
            return;
        }

        // Convert point to world coordinates
        pointsMap[locId] = Vector4f(point.toVector3() * voxelSize, 1);
        // Normals are the same whether in world or voxel coordinates
        normalsMap[locId] = Vector4f(outNormal, 0);
#undef pointsMap
#undef normalsMap
    }
};
void approxEqual(float a, float b, const float eps = 0.00001) {
    assert(abs(a - b) < eps);
}


void approxEqual(Matrix4f a, Matrix4f b, const float eps = 0.00001) {
    for (int i = 0; i < 4 * 4; i++)
        approxEqual(a.m[i], b.m[i], eps);
}

void approxEqual(Matrix3f a, Matrix3f b, const float eps = 0.00001) {
    for (int i = 0; i < 3 * 3; i++)
        approxEqual(a.m[i], b.m[i], eps);
}

// 1. raycast scene from current viewpoint 
// to create point cloud for tracking
RayImage * CreateICPMapsForCurrentView() {
    assert(currentView);

    auto imgSize_d = currentView->depthImage->imgSize();
    assert(imgSize_d.area() > 1);
    auto pointsMap = new ITMFloat4Image(imgSize_d);
    auto normalsMap = new ITMFloat4Image(imgSize_d);

    assert(!outIcpMap);
    outIcpMap = new RayImage(
        pointsMap, 
        normalsMap,
        CoordinateSystem::global(),

        currentView->depthImage->eyeCoordinates,
        currentView->depthImage->cameraIntrinsics
        );

    assert(Scene::getCurrentScene());

    // TODO reduce conversion friction
    ITMPose pose; pose.SetM(currentView->depthImage->eyeCoordinates->fromGlobal);
    ITMIntrinsics intrin; 
    intrin.projectionParamsSimple.all = currentView->depthImage->cameraIntrinsics;
    Common(
        &pose, //trackingState->pose_d,
        &intrin,
        imgSize_d
        );
    cudaDeviceSynchronize(); 

    approxEqual(raycastResult->eyeCoordinates->fromGlobal, currentView->depthImage->eyeCoordinates->fromGlobal);
    assert(raycastResult->pointCoordinates == voxelCoordinates);

    // Create ICP maps
    forEachPixelNoImage<processPixelICP>(imgSize_d);
    cudaDeviceSynchronize();

    // defensive
    assert(outIcpMap->eyeCoordinates == currentView->depthImage->eyeCoordinates);
    assert(outIcpMap->pointCoordinates == CoordinateSystem::global());
    assert(outIcpMap->imgSize() == imgSize_d);
    assert(outIcpMap->normalImage->noDims == imgSize_d);
    auto icpMap = outIcpMap;
    outIcpMap = 0;
    return icpMap;
}
