#include "ITMSceneReconstructionEngine.h"
#include "ITMCUDAUtils.h"
#include "ITMLibDefines.h"
#include "ITMPixelUtils.h"
#include "ITMRepresentationAccess.h"
#include "CoordinateSystem.h"
#include "CameraImage.h"

// Reduce passing by using global variables.
// Note that this whole program should be run single-threaded on a
// single GPU and single GPU context, so this is no problem
// TODO use __const__ memory (TODO is passing parameters faster?)

/// current color image
static __managed__ /*const*/ CameraImage<Vector4u>* colorImage = 0;
/// current depth image
static __managed__ /*const*/ DepthImage* depthImage = 0;

/// Fusion Stage - Camera Data Integration
/// \returns \f$\eta\f$, -1 on failure
// Note that the stored T-SDF values are normalized to lie
// in [-1,1] within the truncation band.
GPU_ONLY inline float computeUpdatedVoxelDepthInfo(
    DEVICEPTR(ITMVoxel) &voxel, //!< X
    const THREADPTR(Point) & pt_model //!< in world space
    )
{

    // project point into depth image
    /// X_d, depth camera coordinate system
    const Vector4f pt_camera = Vector4f(
        depthImage->eyeCoordinates->convert(pt_model).location,
        1);
    /// \pi(K_dX_d), projection into the depth image
    Vector2f pt_image;
    if (!depthImage->project(pt_model, pt_image))
        return -1;

    // get measured depth from image, no interpolation
    /// I_d(\pi(K_dX_d))
    auto p = depthImage->getPointForPixel(pt_image.toInt());
    const float depth_measure = p.location.z;

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
    const THREADPTR(Point) & pt_model)
{
    Vector2f pt_image;
    if (!colorImage->project(pt_model, pt_image))
        return;

    int oldW = (float)voxel.w_color;
    const Vector3f oldC = TO_FLOAT3(voxel.clr);

    /// Like formula (4) for depth
    const Vector3f newC = TO_VECTOR3(interpolateBilinear<Vector4f>(colorImage->image->GetData(), pt_image, colorImage->imgSize()));
    int newW = 1;

    updateVoxelColorInformation(
        voxel,
        oldC, oldW, newC, newW);
}


GPU_ONLY static void computeUpdatedVoxelInfo(
    DEVICEPTR(ITMVoxel) & voxel, //!< [in, out] updated voxel
    const THREADPTR(Point) & pt_model) {
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
        // Find 3d position of depth pixel xy, in eye coordinates
        auto pt_camera = depthImage->getPointForPixel(Vector2i(x, y));

        const float depth = pt_camera.location.z;
        if (depth <= 0 || (depth - mu) < 0 || (depth - mu) < viewFrustum_min || (depth + mu) > viewFrustum_max) return;

        // the found point +- mu
        const Vector pt_camera_v = (pt_camera - depthImage->location());
        const float norm = length(pt_camera_v.direction);
        const Vector pt_camera_v_minus_mu = pt_camera_v*(1.0f - mu / norm);
        const Vector pt_camera_v_plus_mu = pt_camera_v*(1.0f + mu / norm);

        // Convert to voxel block coordinates  
        // the initial point pt_camera_v_minus_mu
        Point point = voxelBlockCoordinates->convert(depthImage->location() + pt_camera_v_minus_mu);
        // the direction towards pt_camera_v_plus_mu in voxelBlockCoordinates
        const Vector vector = voxelBlockCoordinates->convert(pt_camera_v_plus_mu - pt_camera_v_minus_mu);

        // We will step along point -> point_e and add all voxel blocks we encounter to the visible list
        // "Create a segment on the line of sight in the range of the T-SDF truncation band"
        const int noSteps = (int)ceil(2.0f* length(vector.direction) ); // make steps smaller than 1, maybe even < 1/2 to really land in all blocks at least once
        const Vector direction = vector * (1.f / (float)(noSteps - 1));

        //add neighbouring blocks
        for (int i = 0; i < noSteps; i++)
        {
            // "take the block coordinates of voxels on this line segment"
            const VoxelBlockPos blockPos = TO_SHORT_FLOOR3(point.location);
            Scene::requestCurrentSceneVoxelBlockAllocation(blockPos);

            point = point + direction;
        }
    }
};

#include <cuda_runtime.h>

struct IntegrateVoxel {
    static GPU_ONLY void process(const ITMVoxelBlock* vb, ITMVoxel* v, const Vector3i localPos) {
        const Vector3i globalPos = vb->pos.toInt() * SDF_BLOCK_SIZE;

        computeUpdatedVoxelInfo(*v, 
            Point(
                CoordinateSystem::global(), 
                (globalPos.toFloat() + localPos.toFloat()) * voxelSize
            ));
    }
};

// Allocation request and setup of global variables part
void FuseView_pre(
    const ITMView * const view,
    Matrix4f M_d
    ) {
    cudaDeviceSynchronize();
    assert(Scene::getCurrentScene());

    Matrix4f M_rgb = view->calib->trafo_rgb_to_depth.calib_inv * M_d;

    auto depthCs = new CoordinateSystem(M_d.getInv());
    depthImage = new DepthImage(view->depth, depthCs, view->calib->intrinsics_d.projectionParamsSimple.all);

    auto colorCs = new CoordinateSystem(M_rgb.getInv());
    colorImage = new CameraImage<Vector4u>(view->rgb, colorCs, view->calib->intrinsics_rgb.projectionParamsSimple.all);
        

    // allocation request
    forEachPixelNoImage<buildHashAllocAndVisibleTypePP>(view->depth->noDims);

    cudaDeviceSynchronize();
}

/// Fusion stage of the system
void FuseView(
    const ITMView * const view,
    Matrix4f M_d)
{
    FuseView_pre(
        view, M_d
        );

    // allocation
    Scene::performCurrentSceneAllocations();

    // camera data integration
    cudaDeviceSynchronize();
    Scene::getCurrentScene()->doForEachAllocatedVoxel<IntegrateVoxel>();


    delete depthImage->eyeCoordinates;
    delete depthImage; depthImage = 0;
    delete colorImage->eyeCoordinates;
    delete colorImage; colorImage = 0;
}

// 222