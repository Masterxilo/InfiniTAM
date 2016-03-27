/// \file c.f. newcombe_etal_ismar2011.pdf

#include "ITMDepthTracker.h"
#include "ITMCUDAUtils.h"
#include "CUDADefines.h"
#include "Cholesky.h"
#include "ITMLibDefines.h"
#include "ITMPixelUtils.h"

using namespace ITMLib::Engine;


struct ITMDepthTracker_KernelParameters {
    ITMDepthTracker::AccuCell *accu;
    float *depth;
    Matrix4f approxInvPose;
    Vector4f *pointsMap;
    Vector4f *normalsMap;
    Vector4f sceneIntrinsics;
    Vector2i sceneImageSize;
    Matrix4f scenePose;
    Vector4f viewIntrinsics;
    Vector2i viewImageSize;
    float distThresh;
};


// device functions
/// \file Depth Tracker, c.f. newcombe_etal_ismar2011.pdf Sensor Pose Estimation
// The current discussion ignores the optimizations/special iterations with 
// rotation estimation only ("At the coarser levels we optimise only for the rotation matrix R.")
// Also 'translation only' is not used for the depth ICP tracker.


/**
Computes
\f{eqnarray*}{
b &:=& n_{k-1}^\top(p_{k-1} - p_k)  \\
A^T &:=& G(u)^T . n_{k-1}\\
\f}

where \f$G(u) = [ [p_k]_\times \;|\; Id ]\f$ a 3 x 6 matrix and \f$A^T\f$ is a 6 x 1 column vector.

\f$p_{k-1}\f$ is the point observed in the last frame in
the direction in which \f$p_k\f$ is observed (projective data association).

\f$n_{k-1}\f$ is the normal that was observed at that location

\f$b\f$ is the point-plane alignment energy for the point under consideration

\param x,y \f$\mathbf u\f$
\return false on failure
\see newcombe_etal_ismar2011.pdf Sensor Pose Estimation
*/
template<TrackerIterationType iterationType>
CPU_AND_GPU inline bool computePerPointGH_Depth_Ab(
    THREADPTR(float) *AT, //!< [out]
    THREADPTR(float) &b,//!< [out]

    const CONSTPTR(int) & x, const CONSTPTR(int) & y,
    const CONSTPTR(float) &depth, //!< \f$D_k(\mathbf u)\f$
    const CONSTPTR(Vector2i) & viewImageSize,  //!< unused
    const CONSTPTR(Vector4f) & viewIntrinsics, //!< K
    const CONSTPTR(Vector2i) & sceneImageSize,
    const CONSTPTR(Vector4f) & sceneIntrinsics, //!< K
    const CONSTPTR(Matrix4f) & T_g_k, //!< \f$T_{g,k}\f$ current estimate
    const CONSTPTR(Matrix4f) & T_km1_g, //!< \f$T_{g, k-1}^{-1}\f$, i.e. \f$T_{k-1,g}\f$
    const CONSTPTR(Vector4f) *pointsMap,//!< of size sceneImageSize, for frame k-1. \f$V_{k-1}\f$
    const CONSTPTR(Vector4f) *normalsMap, //!< of size sceneImageSize, for frame k-1. \f$N_{k-1}\f$
    const float distThresh //!< \f$\epsilon_d\f$
    )
{
    if (depth <= 1e-8f) return false;

    // (1) Grab the corresponding points by projective data association
    // p_k := T_{g,k}V_k(u) = V_k^g(u)
    Vector4f p_k = T_g_k *
        depthTo3D(viewIntrinsics, x, y, depth); // V_k(u) = D_k(u)K^{-1}u 
    p_k.w = 1.0f;

    // hat_u = \pi(K T_{k-1,g} T_{g,k}V_k(u) )
    Vector2f hat_u;
    if (!projectExtraBounds(sceneIntrinsics, sceneImageSize,
        T_km1_g * p_k, // T_{g,k}V_k(u)
        hat_u)) return false;
    // p_km1 := V_{k-1}(\hat u)
    Vector4f p_km1 = interpolateBilinear<Vector4f, WITH_HOLES>(pointsMap, hat_u, sceneImageSize);
    if (!isLegalColor(p_km1)) return false;

    // n_km1 := N_{k-1}(\hat u)
    Vector4f n_km1 = interpolateBilinear<Vector4f, WITH_HOLES>(normalsMap, hat_u, sceneImageSize);
    if (!isLegalColor(n_km1)) return false;

    // d := p_km1 - p_k
    Vector3f d = p_km1.toVector3() - p_k.toVector3();

    // [
    // Projective data assocation rejection test, "\Omega_k(u) != 0"
    // TODO check whether normal matches normal from image, done in the original paper, but does not seem to be required
    if (length2(d) > distThresh) return false;
    // ]

    // (2) Point-plane ICP computations

    // b = n_km1 . (p_km1 - p_k)
    b = dot(n_km1.toVector3(), d);

    // Compute A^T = G(u)^T . n_{k-1}
    // Where G(u) = [ [p_k]_x Id ] a 3 x 6 matrix
    // [v]_x denotes the skew symmetric matrix such that for all w [v]_x w = v \cross w
    int counter = 0;
#define rotationPart() do {\
    AT[counter++] = +p_k.z * n_km1.y - p_k.y * n_km1.z;\
    AT[counter++] = -p_k.z * n_km1.x + p_k.x * n_km1.z;\
    AT[counter++] = +p_k.y * n_km1.x - p_k.x * n_km1.y;} while(false)
#define translationPart() do {\
    AT[counter++] = n_km1.x;\
    AT[counter++] = n_km1.y;\
    AT[counter++] = n_km1.z;} while(false)

    switch (iterationType) {
    case TRACKER_ITERATION_ROTATION: rotationPart(); break;
    case TRACKER_ITERATION_TRANSLATION: translationPart(); break;
    case TRACKER_ITERATION_BOTH: rotationPart(); translationPart(); break;
    }
#undef rotationPart
#undef translationPart

    return true;
}

#define REDUCE_BLOCK_SIZE 256 // must be power of 2. Used for reduction of a sum.
template<TrackerIterationType iterationType>
__device__ void depthTrackerOneLevel_g_rt_device_main(
    ITMDepthTracker::AccuCell *accu,
    float *depth,
    Matrix4f approxInvPose,
    Vector4f *pointsMap,
    Vector4f *normalsMap,
    Vector4f sceneIntrinsics,
    Vector2i sceneImageSize, Matrix4f scenePose, Vector4f viewIntrinsics, Vector2i viewImageSize,
    float distThresh)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

    int locId_local = threadIdx.x + threadIdx.y * blockDim.x;

    __shared__ bool should_prefix; // set to true if any point is valid

    should_prefix = false;
    __syncthreads();

    const bool shortIteration = iterationType != TRACKER_ITERATION_BOTH;
    const int noPara = shortIteration ? 3 : 6;
    const int noParaSQ = shortIteration ? 3 + 2 + 1 : 6 + 5 + 4 + 3 + 2 + 1;
    float A[noPara]; float b;
    bool isValidPoint = false;

    if (x < viewImageSize.x && y < viewImageSize.y)
    {
        isValidPoint = computePerPointGH_Depth_Ab<iterationType>(A, b, x, y, depth[x + y * viewImageSize.x],
            viewImageSize, viewIntrinsics, sceneImageSize, sceneIntrinsics, approxInvPose, scenePose, pointsMap, normalsMap, distThresh);
        if (isValidPoint) should_prefix = true;
    }

    if (!isValidPoint) {
        for (int i = 0; i < noPara; i++) A[i] = 0.0f;
        b = 0.0f;
    }

    __syncthreads();

    if (!should_prefix) return;

    __shared__ float dim_shared1[REDUCE_BLOCK_SIZE];
    __shared__ float dim_shared2[REDUCE_BLOCK_SIZE];
    __shared__ float dim_shared3[REDUCE_BLOCK_SIZE];

    { //reduction for noValidPoints
        warpReduce256<int>(
            isValidPoint,
            dim_shared1,
            locId_local,
            &(accu->noValidPoints));
    }

    { //reduction for energy function value
        warpReduce256<float>(
            b*b,
            dim_shared1,
            locId_local,
            &(accu->f));
    }

    __syncthreads();

    //reduction for nabla
    for (unsigned char paraId = 0; paraId < noPara; paraId += 3)
    {
        dim_shared1[locId_local] = b*A[paraId + 0];
        dim_shared2[locId_local] = b*A[paraId + 1];
        dim_shared3[locId_local] = b*A[paraId + 2];
        __syncthreads();

        if (locId_local < 128) {
            dim_shared1[locId_local] += dim_shared1[locId_local + 128];
            dim_shared2[locId_local] += dim_shared2[locId_local + 128];
            dim_shared3[locId_local] += dim_shared3[locId_local + 128];
        }
        __syncthreads();
        if (locId_local < 64) {
            dim_shared1[locId_local] += dim_shared1[locId_local + 64];
            dim_shared2[locId_local] += dim_shared2[locId_local + 64];
            dim_shared3[locId_local] += dim_shared3[locId_local + 64];
        }
        __syncthreads();

        if (locId_local < 32) {
            warpReduce(dim_shared1, locId_local);
            warpReduce(dim_shared2, locId_local);
            warpReduce(dim_shared3, locId_local);
        }
        __syncthreads();

        if (locId_local == 0) {
            atomicAdd(&(accu->ATb[paraId + 0]), dim_shared1[0]);
            atomicAdd(&(accu->ATb[paraId + 1]), dim_shared2[0]);
            atomicAdd(&(accu->ATb[paraId + 2]), dim_shared3[0]);
        }
    }

    __syncthreads();

    float localHessian[noParaSQ];

    for (unsigned char r = 0, counter = 0; r < noPara; r++)
    {
        for (int c = 0; c <= r; c++, counter++) localHessian[counter] = A[r] * A[c];
    }

    //reduction for hessian
    for (unsigned char paraId = 0; paraId < noParaSQ; paraId += 3)
    {
        dim_shared1[locId_local] = localHessian[paraId + 0];
        dim_shared2[locId_local] = localHessian[paraId + 1];
        dim_shared3[locId_local] = localHessian[paraId + 2];
        __syncthreads();

        if (locId_local < 128) {
            dim_shared1[locId_local] += dim_shared1[locId_local + 128];
            dim_shared2[locId_local] += dim_shared2[locId_local + 128];
            dim_shared3[locId_local] += dim_shared3[locId_local + 128];
        }
        __syncthreads();
        if (locId_local < 64) {
            dim_shared1[locId_local] += dim_shared1[locId_local + 64];
            dim_shared2[locId_local] += dim_shared2[locId_local + 64];
            dim_shared3[locId_local] += dim_shared3[locId_local + 64];
        }
        __syncthreads();

        if (locId_local < 32) {
            warpReduce(dim_shared1, locId_local);
            warpReduce(dim_shared2, locId_local);
            warpReduce(dim_shared3, locId_local);
        }
        __syncthreads();

        if (locId_local == 0) {
            atomicAdd(&(accu->AT_A_tri[paraId + 0]), dim_shared1[0]);
            atomicAdd(&(accu->AT_A_tri[paraId + 1]), dim_shared2[0]);
            atomicAdd(&(accu->AT_A_tri[paraId + 2]), dim_shared3[0]);
        }
    }
}

template<TrackerIterationType iterationType>
KERNEL depthTrackerOneLevel_g_rt_device(ITMDepthTracker_KernelParameters para)
{
    depthTrackerOneLevel_g_rt_device_main<iterationType>(para.accu, para.depth, para.approxInvPose, para.pointsMap, para.normalsMap, para.sceneIntrinsics, para.sceneImageSize, para.scenePose, para.viewIntrinsics, para.viewImageSize, para.distThresh);
}

// host methods

ITMDepthTracker::AccuCell ITMDepthTracker::ComputeGandH(Matrix4f T_g_k_estimate) {
    Vector4f *pointsMap = this->pointsMap->GetData(MEMORYDEVICE_CUDA);
    Vector4f *normalsMap = this->normalsMap->GetData(MEMORYDEVICE_CUDA);
    Vector4f sceneIntrinsics = getViewIntrinsics();
    Vector2i sceneImageSize = this->pointsMap->noDims;
    assert(sceneImageSize == this->normalsMap->noDims);

    float *depth = currentTrackingLevel->depth->GetData(MEMORYDEVICE_CUDA);
    Vector4f viewIntrinsics = currentTrackingLevel->intrinsics;
    Vector2i viewImageSize = currentTrackingLevel->depth->noDims;

    dim3 blockSize(16, 16); // must equal REDUCE_BLOCK_SIZE
    assert(16 * 16 == REDUCE_BLOCK_SIZE);

    dim3 gridSize(
        (int)ceil((float)viewImageSize.x / (float)blockSize.x),
        (int)ceil((float)viewImageSize.y / (float)blockSize.y));

    cudaSafeCall(cudaMemset(accu, 0, sizeof(AccuCell)));

    struct ITMDepthTracker_KernelParameters args;
    args.accu = accu;
    args.depth = depth;
    args.approxInvPose = T_g_k_estimate;
    args.pointsMap = pointsMap;
    args.normalsMap = normalsMap;
    args.sceneIntrinsics = sceneIntrinsics;
    args.sceneImageSize = sceneImageSize;
    args.scenePose = scenePose;
    args.viewIntrinsics = viewIntrinsics;
    args.viewImageSize = viewImageSize;
    args.distThresh = currentTrackingLevel->distanceThreshold;
#define iteration(it) \
			it: LAUNCH_KERNEL(depthTrackerOneLevel_g_rt_device<it>, gridSize, blockSize, args);

    switch (iterationType()) {
        case iteration(TRACKER_ITERATION_ROTATION);
            case iteration(TRACKER_ITERATION_TRANSLATION);
                case iteration(TRACKER_ITERATION_BOTH);
    }
#undef iteration

    cudaDeviceSynchronize(); // for later access of accu
    return *accu;
}



ITMDepthTracker::ITMDepthTracker(
    Vector2i depthImgSize,
    const ITMLowLevelEngine *lowLevelEngine) 
{
    this->lowLevelEngine = lowLevelEngine;

    // Tracking strategy:
    const int noHierarchyLevels = 5;
    const float distThreshStep = depthTrackerICPThreshold / noHierarchyLevels;
    // starting with highest resolution (lowest level, last to be executed)
    trackingLevels.push_back(TrackingLevel(2, TRACKER_ITERATION_BOTH, depthTrackerICPThreshold - distThreshStep * 4, depthImgSize));
    trackingLevels.push_back(TrackingLevel(4, TRACKER_ITERATION_BOTH, depthTrackerICPThreshold - distThreshStep * 3, depthImgSize / 2));
    trackingLevels.push_back(TrackingLevel(6, TRACKER_ITERATION_ROTATION, depthTrackerICPThreshold - distThreshStep * 2, depthImgSize / 4));
    trackingLevels.push_back(TrackingLevel(8, TRACKER_ITERATION_ROTATION, depthTrackerICPThreshold - distThreshStep, depthImgSize / 8));
    trackingLevels.push_back(TrackingLevel(10, TRACKER_ITERATION_ROTATION, depthTrackerICPThreshold, depthImgSize / 16));
    assert(trackingLevels.size() == noHierarchyLevels);

    cudaSafeCall(cudaMallocManaged((void**)&accu, sizeof(AccuCell)));
}

ITMDepthTracker::~ITMDepthTracker(void)
{
    cudaSafeCall(cudaFree(accu));
}

void ITMDepthTracker::ComputeDelta(float *step, float *nabla, float *hessian) const
{
    for (int i = 0; i < 6; i++) step[i] = 0;

    if (shortIteration())
    {
        // Keep only upper 3x3 part of hessian
        float smallHessian[3 * 3];
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) smallHessian[r + c * 3] = hessian[r + c * 6];

        ORUtils::Cholesky::solve(smallHessian, 3, nabla, step);
    }
    else
    {
        ORUtils::Cholesky::solve(hessian, 6, nabla, step);
    }
}

bool ITMDepthTracker::HasConverged(float *step) const
{
    // Compute ||step||_2^2
    float stepLength = 0.0f;
    for (int i = 0; i < 6; i++) stepLength += step[i] * step[i];

    // heuristic? Why /6?
    if (sqrt(stepLength) / 6 < depthTrackerTerminationThreshold) return true; //converged

    return false;
}

Matrix4f ITMDepthTracker::ComputeTinc(const float *delta) const
{
    // step is T_inc, expressed as a parameter vector 
    // (beta, gamma, alpha, tx,ty, tz)
    // beta, gamma, alpha parametrize the rotation axis
    float step[6];

    // Depending on the iteration type, fill in 0 for values that where not computed.
    switch (currentTrackingLevel->iterationType)
    {
    case TRACKER_ITERATION_ROTATION:
        step[0] = (float)(delta[0]); step[1] = (float)(delta[1]); step[2] = (float)(delta[2]);
        step[3] = 0.0f; step[4] = 0.0f; step[5] = 0.0f;
        break;
    case TRACKER_ITERATION_TRANSLATION:
        step[0] = 0.0f; step[1] = 0.0f; step[2] = 0.0f;
        step[3] = (float)(delta[0]); step[4] = (float)(delta[1]); step[5] = (float)(delta[2]);
        break;
    default:
    case TRACKER_ITERATION_BOTH:
        step[0] = (float)(delta[0]); step[1] = (float)(delta[1]); step[2] = (float)(delta[2]);
        step[3] = (float)(delta[3]); step[4] = (float)(delta[4]); step[5] = (float)(delta[5]);
        break;
    }

    // Incremental pose update assuming small angles.
    Matrix4f Tinc;

    Tinc.m00 = 1.0f;		Tinc.m10 = step[2];		Tinc.m20 = -step[1];	Tinc.m30 = step[3];
    Tinc.m01 = -step[2];	Tinc.m11 = 1.0f;		Tinc.m21 = step[0];		Tinc.m31 = step[4];
    Tinc.m02 = step[1];		Tinc.m12 = -step[0];	Tinc.m22 = 1.0f;		Tinc.m32 = step[5];
    Tinc.m03 = 0.0f;		Tinc.m13 = 0.0f;		Tinc.m23 = 0.0f;		Tinc.m33 = 1.0f;
    return Tinc;
}

/// \file c.f. newcombe_etal_ismar2011.pdf, Sensor Pose Estimation section
void ITMDepthTracker::TrackCamera(ITMTrackingState *trackingState, const ITMView *view)
{
    /// Initialize one tracking event base data. Init hierarchy level 0 (finest).
    this->view = view;

    pointsMap = trackingState->pointCloud->locations;
    normalsMap = trackingState->pointCloud->normals;
    scenePose = trackingState->pointCloud->pose_pointCloud->GetM();

    /// Init image hierarchy levels
    trackingLevels[0].depth = view->depth;
    trackingLevels[0].intrinsics = getViewIntrinsics();
    for (int i = 1; i < trackingLevels.size(); i++)
    {
        TrackingLevel* currentLevel = &trackingLevels[i];
        TrackingLevel* previousLevel = &trackingLevels[i - 1];

        lowLevelEngine->FilterSubsampleWithHoles(currentLevel->depth, previousLevel->depth);

        currentLevel->intrinsics = previousLevel->intrinsics * 0.5f;
    }

    // Coarse to fine
    for (int levelId = trackingLevels.size() - 1; levelId >= 0; levelId--)
    {
        currentTrackingLevel = &trackingLevels[levelId];
        if (iterationType() == TRACKER_ITERATION_NONE) continue;

#define T_k_g_estimate trackingState->pose_d
        // T_g_k_estimate caches T_k_g_estimate->GetInvM()
        Matrix4f T_g_k_estimate = T_k_g_estimate->GetInvM();

#define set_T_k_g_estimate(x)\
        T_k_g_estimate->SetFrom(&x);
        T_g_k_estimate = T_k_g_estimate->GetInvM();

#define set_T_k_g_estimate_from_T_g_k_estimate(x) \
        T_k_g_estimate->SetInvM(x);\
		T_k_g_estimate->Coerce(); /* and make sure we've got an SE3*/\
        T_g_k_estimate = T_k_g_estimate->GetInvM();

        // We will 'accept' updates into trackingState->pose_d and T_g_k_estimate
        // before we know whether they actually decrease the energy.
        // When they did not in fact, we will revert to this value that was known to have less energy 
        // than all previous estimates.
        ITMPose least_energy_T_k_g_estimate(*(trackingState->pose_d));

        // Track least energy we measured so far to see whether we improved
        float f_old = 1e20f;

        // current levenberg-marquart style damping parameter, often called mu.
        float lambda = 1.0;

        // Iterate as required
        for (int iterNo = 0; iterNo < currentTrackingLevel->numberOfIterations; iterNo++)
        {
            // [ this takes most time. 
            // Computes f(x) as well as A^TA and A^Tb for next computation of delta_x as
            // (A^TA + lambda * diag(A^TA)) delta_x = A^T b
            // if f decreases, the delta is applied definitely, otherwise x is reset.
            // So we do:
            /*
            x = x_best;
            lambda = 1;
            f_best = infinity

            repeat:
            compute f_new, A^TA_new, A^T b_new

            if (f_new > f_best) {x = x_best; lambda *= 10;}
            else {
            x_best = x;
            A^TA = A^TA_new
            A^Tb = A^Tb_new
            }

            solve (A^TA + lambda * diag(A^TA)) delta_x = A^T b
            x += delta_x;

            */


            // evaluate error function at currently accepted
            // T_g_k_estimate
            // and compute information for next update
            float f_new;
            int noValidPoints;
            float new_sum_ATb[6];
            float new_sum_AT_A[6 * 6];
            noValidPoints = this->ComputeGandH(f_new, new_sum_ATb, new_sum_AT_A, T_g_k_estimate);
            // ]]

            float least_energy_sum_AT_A[6 * 6],
                damped_least_energy_sum_AT_A[6 * 6];
            float least_energy_sum_ATb[6];

            // check if energy actually *increased* with the last update
            // Note: This happens rarely, namely when the blind 
            // gauss-newton step actually leads to an *increase in energy
            // because the damping was too small
            if ((noValidPoints <= 0) || (f_new > f_old)) {
                // If so, revert pose and discard/ignore new_sum_AT_A, new_sum_ATb
                // TODO would it be worthwhile to not compute these when they are not going to be used?
                set_T_k_g_estimate(least_energy_T_k_g_estimate);
                // Increase damping, then solve normal equations again with old matrix (see below)
                lambda *= 10.0f;
            }
            else {
                f_old = f_new;
                least_energy_T_k_g_estimate.SetFrom(T_k_g_estimate);

                // Prepare to solve a new system

                // Preconditioning
                for (int i = 0; i < 6 * 6; ++i) least_energy_sum_AT_A[i] = new_sum_AT_A[i] / noValidPoints;
                for (int i = 0; i < 6; ++i) least_energy_sum_ATb[i] = new_sum_ATb[i] / noValidPoints;

                // Accept and decrease damping
                lambda /= 10.0f;
            }
            // Solve normal equations

            // Apply levenberg-marquart style damping (multiply diagonal of ATA by 1.0f + lambda)
            for (int i = 0; i < 6 * 6; ++i) damped_least_energy_sum_AT_A[i] = least_energy_sum_AT_A[i];
            for (int i = 0; i < 6; ++i) damped_least_energy_sum_AT_A[i + i * 6] *= 1.0f + lambda;

            // compute the update step parameter vector x
            float x[6];
            ComputeDelta(x,
                least_energy_sum_ATb,
                damped_least_energy_sum_AT_A);

            // Apply the corresponding Tinc
            set_T_k_g_estimate_from_T_g_k_estimate(
                /* T_g_k_estimate = */
                ComputeTinc(x) * T_g_k_estimate
                );

            // if step is small, assume it's going to decrease the error and finish
            if (HasConverged(x)) break;
        }
    }

    // Convert T_g_k (k to global) to T_k_g (global to k)
}

