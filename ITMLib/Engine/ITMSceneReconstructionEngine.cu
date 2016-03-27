// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMSceneReconstructionEngine.h"
#include "ITMCUDAUtils.h"
#include "ITMLibDefines.h"
#include "ITMPixelUtils.h"
#include "ITMRepresentationAccess.h"
#include "ITMLocalVBA.h"


// Reduce passing by using global variables.
// Note that this whole program should be run single-threaded on a
// single GPU and single GPU context, so this is no problem
// TODO use __const__ memory (TODO is passing parameters faster?)
static __managed__ /*const*/ Vector2i imgSize_d;
static __managed__ /*const*/ Vector2i imgSize_rgb;
static __managed__ /*const*/ Matrix4f M_d;
static __managed__ /*const*/ Matrix4f invM_d;//!< depth to world transformation
static __managed__ /*const*/ Matrix4f M_rgb;
static __managed__ /*const*/ Vector4f invProjParams_d;//!< Note: Inverse projection parameters to avoid division by fx, fy.
static __managed__ /*const*/ Vector4f projParams_d;
static __managed__ /*const*/ Vector4f projParams_rgb;
/// current depth image
static __managed__ /*const*/ float* depth;
/// current color image
static __managed__ /*const*/ Vector4u *rgb;

/// Fusion Stage - Camera Data Integration
/// \returns \f$\eta\f$, -1 on failure
// Note that the stored T-SDF values are normalized to lie
// in [-1,1] within the truncation band.
GPU_ONLY inline float computeUpdatedVoxelDepthInfo(
    DEVICEPTR(ITMVoxel) &voxel, //!< X
    const THREADPTR(Vector4f) & pt_model)
{

    float depth_measure, eta, oldF, newF;
    int oldW, newW;

    // project point into depth image
    /// X_d, depth camera coordinate system
    Vector4f pt_camera;
    /// \pi(K_dX_d), projection into the depth image
    Vector2f pt_image;
    if (!projectModel(projParams_d, M_d,
        imgSize_d, pt_model, pt_camera, pt_image)) return -1;

    // get measured depth from image, no interpolation
    /// I_d(\pi(K_dX_d))
    depth_measure = sampleNearest(depth, pt_image, imgSize_d);
    if (depth_measure <= 0.0) return -1;

    /// I_d(\pi(K_dX_d)) - X_d^(z)          (3)
    eta = depth_measure - pt_camera.z;
    // check whether voxel needs updating
    if (eta < -mu) return eta;

    // compute updated SDF value and reliability (number of observations)
    /// D(X), w(X)
    oldF = voxel.getSDF();
    oldW = voxel.w_depth;

    // newF, normalized for -1 to 1
    newF = MIN(1.0f, eta / mu);
    newW = 1;

    updateVoxelDepthInformation(
        voxel,
        oldF, oldW, newF, newW);

    return eta;
}

/// \returns early on failure
GPU_ONLY inline void computeUpdatedVoxelColorInfo(
    DEVICEPTR(ITMVoxel) &voxel,
    const THREADPTR(Vector4f) & pt_model)
{
    Vector4f pt_camera; Vector2f pt_image;
    Vector3f oldC, newC;
    int newW, oldW;

    if (!projectModel(projParams_rgb, M_rgb,
        imgSize_rgb, pt_model, pt_camera, pt_image)) return;

    oldW = (float)voxel.w_color;
    oldC = TO_FLOAT3(voxel.clr);

    /// Like formula (4) for depth
    newC = TO_VECTOR3(interpolateBilinear<Vector4f>(rgb, pt_image, imgSize_rgb));
    newW = 1;

    updateVoxelColorInformation(
        voxel,
        oldC, oldW, newC, newW);
}


GPU_ONLY static void computeUpdatedVoxelInfo(
    DEVICEPTR(ITMVoxel) & voxel, //!< [in, out] updated voxel
    const THREADPTR(Vector4f) & pt_model)
{
    float eta = computeUpdatedVoxelDepthInfo(voxel, pt_model);

    // Only the voxels within +- 25% mu of the surface get color
    if ((eta > mu) || (fabs(eta / mu) > 0.25f)) return;
    computeUpdatedVoxelColorInfo(voxel, pt_model);
}

/// Determine the blocks around a given depth sample that are currently visible
/// and need to be allocated.
/// Builds hashVisibility and entriesAllocType.
/// \param x,y [in] loop over depth image.
GPU_ONLY inline void buildHashAllocAndVisibleTypePP(const int x, const int y) {
    float depth_measure; unsigned int hashIdx; int noSteps;
    Vector4f pt_camera_f; Vector3f point_e, point, direction; 

    // Find 3d position of depth pixel xy, in world coordinates
    depth_measure = sampleNearest(depth,x, y, imgSize_d);
    if (depth_measure <= 0 || (depth_measure - mu) < 0 || (depth_measure - mu) < viewFrustum_min || (depth_measure + mu) > viewFrustum_max) return;

    pt_camera_f = depthTo3DInvProjParams(invProjParams_d, x, y, depth_measure);

    // distance from camera
    float norm = length(pt_camera_f.toVector3());

    // Transform into fractional block coordinates the found point +- mu
    // TODO why /norm? An adhoc fix to not allocate too much when far away and allocate more when nearby?
#define oneOverVoxelBlockWorldspaceSize (1.0f / (voxelSize * SDF_BLOCK_SIZE))
    point = TO_VECTOR3(invM_d * (pt_camera_f * (1.0f - mu / norm))) * oneOverVoxelBlockWorldspaceSize;
    point_e = TO_VECTOR3(invM_d * (pt_camera_f * (1.0f + mu / norm))) * oneOverVoxelBlockWorldspaceSize;

    // We will step along point -> point_e and add all voxel blocks we encounter to the visible list
    // "Create a segment on the line of sight in the range of the T-SDF truncation band"
    direction = point_e - point;
    norm = length(direction);
    noSteps = (int)ceil(2.0f*norm);

    direction /= (float)(noSteps - 1);

    //add neighbouring blocks
    for (int i = 0; i < noSteps; i++)
    {
        // "take the block coordinates of voxels on this line segment"
        VoxelBlockPos blockPos = TO_SHORT_FLOOR3(point);
        Scene::requestCurrentSceneVoxelBlockAllocation(blockPos);

        point += direction;
    }
}

#include <cuda_runtime.h>

struct IntegrateVoxel {
    static GPU_ONLY void process(ITMVoxelBlock* vb, ITMVoxel* v, Vector3i localPos) {
        Vector3i globalPos = vb->pos.toInt() * SDF_BLOCK_SIZE;
        const Vector4f pt_model = Vector4f(
            (globalPos.toFloat() + localPos.toFloat()) * voxelSize, 1.f);

        computeUpdatedVoxelInfo(*v, pt_model);
    }
};

/// Loop over depth image pixels
KERNEL buildHashAllocAndVisibleType_device() {
    const int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > imgSize_d.x - 1 || y > imgSize_d.y - 1) return;

    buildHashAllocAndVisibleTypePP(x, y);
}

// host methods

/// Fusion stage of the system
void ITMSceneReconstructionEngine_ProcessFrame(
    const ITMView * const view,
    const ITMTrackingState * const trackingState)
{
    assert(Scene::getCurrentScene());
    imgSize_d = view->depth->noDims;
    imgSize_rgb = view->rgb->noDims;
    M_d = trackingState->pose_d->GetM();
    invM_d = trackingState->pose_d->GetInvM();
    M_rgb = view->calib->trafo_rgb_to_depth.calib_inv * M_d;
    projParams_d = view->calib->intrinsics_d.projectionParamsSimple.all;
    projParams_rgb = view->calib->intrinsics_rgb.projectionParamsSimple.all;
    invProjParams_d = view->calib->intrinsics_d.getInverseProjParams();
    depth = view->depth->GetData(MEMORYDEVICE_CUDA);
    rgb = view->rgb->GetData(MEMORYDEVICE_CUDA);

    // allocation
    const dim3 cudaBlockSizeHV(16, 16);
    const dim3 gridSizeHV((int)ceil((float)imgSize_d.x / (float)cudaBlockSizeHV.x), (int)ceil((float)imgSize_d.y / (float)cudaBlockSizeHV.y));

    LAUNCH_KERNEL(buildHashAllocAndVisibleType_device, gridSizeHV, cudaBlockSizeHV);

    Scene::performCurrentSceneAllocations();

    // camera data integration
    Scene::getCurrentScene()->doForEachAllocatedVoxel<IntegrateVoxel>();
}

