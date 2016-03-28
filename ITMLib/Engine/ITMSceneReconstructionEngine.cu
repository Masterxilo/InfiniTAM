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
    const THREADPTR(Vector4f) & pt_model //!< in world space
    )
{

    // project point into depth image
    /// X_d, depth camera coordinate system
    Vector4f pt_camera;
    /// \pi(K_dX_d), projection into the depth image
    Vector2f pt_image;
    if (!projectModel(projParams_d, M_d,
        imgSize_d, pt_model, pt_camera, pt_image)) return -1;

    // get measured depth from image, no interpolation
    /// I_d(\pi(K_dX_d))
    const float depth_measure = sampleNearest(depth, pt_image, imgSize_d);
    if (depth_measure <= 0.0) return -1;

    /// I_d(\pi(K_dX_d)) - X_d^(z)          (3)
    float const eta = depth_measure - pt_camera.z;
    // check whether voxel needs updating
    if (eta < -mu) return eta;

    // compute updated SDF value and reliability (number of observations)
    /// D(X), w(X)
    float const oldF = voxel.getSDF();
    int const oldW = voxel.w_depth;

    // newF, normalized for -1 to 1
    float const newF = MIN(1.0f, eta / mu);
    int const newW = 1;

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
    if (!projectModel(projParams_rgb, M_rgb,
        imgSize_rgb, pt_model, pt_camera, pt_image)) return;

    int oldW = (float)voxel.w_color;
    const Vector3f oldC = TO_FLOAT3(voxel.clr);

    /// Like formula (4) for depth
    const Vector3f newC = TO_VECTOR3(interpolateBilinear<Vector4f>(rgb, pt_image, imgSize_rgb));
    int newW = 1;

    updateVoxelColorInformation(
        voxel,
        oldC, oldW, newC, newW);
}


GPU_ONLY static void computeUpdatedVoxelInfo(
    DEVICEPTR(ITMVoxel) & voxel, //!< [in, out] updated voxel
    const THREADPTR(Vector4f) & pt_model) {
    const float eta = computeUpdatedVoxelDepthInfo(voxel, pt_model);

    // Only the voxels within +- 25% mu of the surface get color
    if ((eta > mu) || (fabs(eta / mu) > 0.25f)) return;
    computeUpdatedVoxelColorInfo(voxel, pt_model);
}

/// Determine the blocks around a given depth sample that are currently visible
/// and need to be allocated.
/// Builds hashVisibility and entriesAllocType.
/// \param x,y [in] loop over depth image.
struct buildHashAllocAndVisibleTypePP {
    forEachPixelNoImage_process() {
        // Find 3d position of depth pixel xy, in world coordinates
        const float depth_measure = sampleNearest(depth, x, y, imgSize_d);
        if (depth_measure <= 0 || (depth_measure - mu) < 0 || (depth_measure - mu) < viewFrustum_min || (depth_measure + mu) > viewFrustum_max) return;

        const Vector4f pt_camera_f = depthTo3D(projParams_d, x, y, depth_measure);

        // distance from camera
        float norm = length(pt_camera_f.toVector3());

        // Transform into fractional block coordinates the found point +- mu
        // TODO why /norm? An adhoc fix to not allocate too much when far away and allocate more when nearby?
#define oneOverVoxelBlockWorldspaceSize (1.0f / (voxelSize * SDF_BLOCK_SIZE))
        Vector3f       point = TO_VECTOR3(invM_d * (pt_camera_f * (1.0f - mu / norm))) * oneOverVoxelBlockWorldspaceSize;
        const Vector3f point_e = TO_VECTOR3(invM_d * (pt_camera_f * (1.0f + mu / norm))) * oneOverVoxelBlockWorldspaceSize;

        // We will step along point -> point_e and add all voxel blocks we encounter to the visible list
        // "Create a segment on the line of sight in the range of the T-SDF truncation band"
        Vector3f direction = point_e - point;
        norm = length(direction);
        const int noSteps = (int)ceil(2.0f*norm);

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
};

#include <cuda_runtime.h>

struct IntegrateVoxel {
    static GPU_ONLY void process(const ITMVoxelBlock* vb, ITMVoxel* v, const Vector3i localPos) {
        const Vector3i globalPos = vb->pos.toInt() * SDF_BLOCK_SIZE;
        const Vector4f pt_model = Vector4f(
            (globalPos.toFloat() + localPos.toFloat()) * voxelSize, 1.f);

        /*
        float z = (threadIdx.z) * voxelSize;
        assert(v);
        float eta = (SDF_BLOCK_SIZE / 2)*voxelSize - z;
        v->setSDF(MAX(MIN(1.0f, eta / mu), -1.f));
        */
        computeUpdatedVoxelInfo(*v, pt_model);
    }
};

// Allocation request and setup of global variables part
void ITMSceneReconstructionEngine_ProcessFrame_pre(
    const ITMView * const view,
    Matrix4f M_d
    ) {
    cudaDeviceSynchronize();
    assert(Scene::getCurrentScene());

    imgSize_d = view->depth->noDims;
    imgSize_rgb = view->rgb->noDims;
    ::M_d = M_d;
    M_d.inv(invM_d);

    M_rgb = view->calib->trafo_rgb_to_depth.calib_inv * M_d;
    projParams_d = view->calib->intrinsics_d.projectionParamsSimple.all;
    projParams_rgb = view->calib->intrinsics_rgb.projectionParamsSimple.all;
    depth = view->depth->GetData(MEMORYDEVICE_CUDA);
    rgb = view->rgb->GetData(MEMORYDEVICE_CUDA);

    // allocation request
    forEachPixelNoImage<buildHashAllocAndVisibleTypePP>(imgSize_d);

    cudaDeviceSynchronize();

}

/// Fusion stage of the system
void ITMSceneReconstructionEngine_ProcessFrame(
    const ITMView * const view,
    Matrix4f M_d)
{
    ITMSceneReconstructionEngine_ProcessFrame_pre(
        view, M_d
        );

    // dump ra
    printf("----------\n");
    // test content of requests "allocate planned"
    uchar *entriesAllocType = (uchar *)malloc(SDF_GLOBAL_BLOCK_NUM);
    Vector3s *blockCoords = (Vector3s *)malloc(SDF_GLOBAL_BLOCK_NUM * sizeof(Vector3s));

    cudaMemcpy(entriesAllocType,
        Scene::getCurrentScene()->voxelBlockHash->needsAllocation,
        SDF_GLOBAL_BLOCK_NUM,
        cudaMemcpyDeviceToHost);

    cudaMemcpy(blockCoords,
        Scene::getCurrentScene()->voxelBlockHash->naKey,
        SDF_GLOBAL_BLOCK_NUM * sizeof(VoxelBlockPos),
        cudaMemcpyDeviceToHost);

    for (int targetIdx = 0; targetIdx < SDF_GLOBAL_BLOCK_NUM; targetIdx++) {
        if (entriesAllocType[targetIdx] == 0) continue;
        printf("%d %d %d\n", blockCoords[targetIdx].x, blockCoords[targetIdx].y, blockCoords[targetIdx].z);
    }

    // allocation
    Scene::performCurrentSceneAllocations();

    // camera data integration
    Scene::getCurrentScene()->doForEachAllocatedVoxel<IntegrateVoxel>();
}

