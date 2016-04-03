/** \file c.f. newcombe_etal_ismar2011.pdf
 * T_{g,k} denotes the transformation from frame k's view space to global space
 * T_{k,g} is the inverse
 */
#include "ITMDepthTracker.h"
#include "ITMCUDAUtils.h"
#include "CUDADefines.h"
#include "Cholesky.h"
#include "ITMLibDefines.h"
#include "ITMPixelUtils.h"
#include <vector>

struct AccuCell : public Managed {
    int noValidPoints;
    float f;
    // ATb
    float ATb[6];
    // AT_A (note that this is actually a symmetric matrix, so we could save some effort and memory)
    float AT_A[6][6];
    void reset() {
        memset(this, 0, sizeof(AccuCell));
    }
};

struct TrackingLevel : public Managed {
    /// FilterSubsampleWithHoles result of one level higher
    ITMFloatImage* depth;
    /// Half of the intrinsics of one level higher
    Vector4f intrinsics;

    const float distanceThreshold;
    const int numberOfIterations;
    const TrackerIterationType iterationType;

    TrackingLevel(int numberOfIterations, TrackerIterationType iterationType, float distanceThreshold) :
        numberOfIterations(numberOfIterations), iterationType(iterationType), distanceThreshold(distanceThreshold) {
        depth = new ITMFloatImage(Vector2i(1,1)); // will get correct size from filter subsample
    }
};
// ViewHierarchy, 0 is highest resolution
static std::vector<TrackingLevel*> trackingLevels;
struct ITMDepthTracker_
{
    ITMDepthTracker_() {
        // Tracking strategy:
        const int noHierarchyLevels = 5;
        const float distThreshStep = depthTrackerICPThreshold / noHierarchyLevels;
        // starting with highest resolution (lowest level, last to be executed)
#define iterations
        trackingLevels.push_back(new TrackingLevel(2  iterations, TRACKER_ITERATION_BOTH, depthTrackerICPThreshold - distThreshStep * 4));
        trackingLevels.push_back(new TrackingLevel(4  iterations, TRACKER_ITERATION_BOTH, depthTrackerICPThreshold - distThreshStep * 3));
        trackingLevels.push_back(new TrackingLevel(6  iterations, TRACKER_ITERATION_ROTATION, depthTrackerICPThreshold - distThreshStep * 2));
        trackingLevels.push_back(new TrackingLevel(8  iterations, TRACKER_ITERATION_ROTATION, depthTrackerICPThreshold - distThreshStep));
        trackingLevels.push_back(new TrackingLevel(10 iterations, TRACKER_ITERATION_ROTATION, depthTrackerICPThreshold));
        assert(trackingLevels.size() == noHierarchyLevels);
    }
} _;

static __managed__ /*const*/ TrackingLevel* currentTrackingLevel;

static TrackerIterationType iterationType() {
    return currentTrackingLevel->iterationType;
}
static bool shortIteration() {
    return (iterationType() == TRACKER_ITERATION_ROTATION);
}

static __managed__ /*const*/ AccuCell accu;
static __managed__ /*const*/ float distThresh;//!< \f$\epsilon_d\f$

#include "cameraimage.h"
/// currentTrackingLevel view, 
//static __managed__ /*const*/ Matrix4f T_g_k;//!< \f$T_{g,k}\f$ current estimate, the transformation from frame k's view space to global space
//static __managed__ /*const*/ Vector4f viewIntrinsics;//!< K
//static __managed__ /*const*/ Vector2i viewImageSize;
//static __managed__ /*const*/ float *depth = 0;
static __managed__ DEVICEPTR(DepthImage) * depthImage = 0;



/// In world coordinates, points map, normals map, for frame k-1, \f$V_{k-1}\f$
static __managed__ DEVICEPTR(RayImage) * lastFrameICPMap = 0;

// device functions
/// \file Depth Tracker, c.f. newcombe_etal_ismar2011.pdf Sensor Pose Estimation
// The current discussion ignores the optimizations/special iterations with 
// rotation estimation only ("At the coarser levels we optimise only for the rotation matrix R.")


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
CPU_AND_GPU static inline bool computePerPointGH_Depth_Ab(
    THREADPTR(float) AT[6], //!< [out]
    THREADPTR(float) &b,//!< [out]

    const CONSTPTR(int) & x, const CONSTPTR(int) & y
    )
{
    // p_k := T_{g,k}V_k(u) = V_k^g(u)
    Point V_ku = depthImage->getPointForPixel(Vector2i(x, y));
    if (V_ku.location.z <= 1e-8f) return false;
    assert(V_ku.coordinateSystem == depthImage->eyeCoordinates);
    Point p_k = CoordinateSystem::global()->convert(V_ku);

    // hat_u = \pi(K T_{k-1,g} T_{g,k}V_k(u) )
    Vector2f hat_u;
    if (!lastFrameICPMap->project(
        p_k,
        hat_u,
        EXTRA_BOUNDS))
        return false;

    bool isIllegal = false;
    Ray ray = lastFrameICPMap->getRayForPixelInterpolated(hat_u, isIllegal);
    if (isIllegal) return false;

    // p_km1 := V_{k-1}(\hat u)
    const Point p_km1 = ray.origin;

    // n_km1 := N_{k-1}(\hat u)
    const Vector n_km1 = ray.direction;

    // d := p_km1 - p_k
    const Vector d = p_km1 - p_k;

    assert(CoordinateSystem::global() == d.coordinateSystem);
    assert(CoordinateSystem::global() == n_km1.coordinateSystem);
    assert(CoordinateSystem::global() == p_km1.coordinateSystem);
    assert(CoordinateSystem::global() == p_k.coordinateSystem);
    // [
    // Projective data assocation rejection test, "\Omega_k(u) != 0"
    // TODO check whether normal matches normal from image, done in the original paper, but does not seem to be required
    if (length2(d.direction) > distThresh) return false;
    // ]

    // (2) Point-plane ICP computations

    // b = n_km1 . (p_km1 - p_k)
    b = n_km1.dot(d);

    // Compute A^T = G(u)^T . n_{k-1}
    // Where G(u) = [ [p_k]_x Id ] a 3 x 6 matrix
    // [v]_x denotes the skew symmetric matrix such that for all w, [v]_x w = v \cross w
    int counter = 0;
    {
        const Vector3f pk = p_k.location;
        const Vector3f nkm1 = n_km1.direction;
        // rotationPart
        AT[counter++] = +pk.z * nkm1.y - pk.y * nkm1.z;
        AT[counter++] = -pk.z * nkm1.x + pk.x * nkm1.z;
        AT[counter++] = +pk.y * nkm1.x - pk.x * nkm1.y;
        // translationPart
        AT[counter++] = nkm1.x;
        AT[counter++] = nkm1.y;
        AT[counter++] = nkm1.z;
    }

    return true;
}

#define REDUCE_BLOCK_SIZE 256 // must be power of 2. Used for reduction of a sum.
static KERNEL depthTrackerOneLevel_g_rt_device_main()
{
    int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

    int locId_local = threadIdx.x + threadIdx.y * blockDim.x;

    __shared__ bool should_prefix; // set to true if any point is valid

    should_prefix = false;
    __syncthreads();

    float A[6];
    float b;
    bool isValidPoint = false;

    auto viewImageSize = depthImage->imgSize();
    if (x < viewImageSize.width && y < viewImageSize.height
        )
    {
        isValidPoint = computePerPointGH_Depth_Ab(
            A, b, x, y);
        if (isValidPoint) should_prefix = true;
    }

    if (!isValidPoint) {
        for (int i = 0; i < 6; i++) A[i] = 0.0f;
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
            &(accu.noValidPoints));
    }
#define reduce(what, into) warpReduce256<float>((what),dim_shared1,locId_local,&(into));
    { //reduction for energy function value
        reduce(b*b, accu.f);
    }

    //reduction for nabla
    for (unsigned char paraId = 0; paraId < 6; paraId++)
    {
        reduce(b*A[paraId], accu.ATb[paraId]);
    }

    float AT_A[6][6];
    int counter = 0;
    for (int r = 0; r < 6; r++)
    {
        for (int c = 0; c < 6; c++) {
            AT_A[r][c] = A[r] * A[c];

            //reduction for hessian
            reduce(AT_A[r][c], accu.AT_A[r][c]);
        }
    }
}

// host methods

AccuCell ComputeGandH(Matrix4f T_g_k_estimate) {
    cudaDeviceSynchronize(); // prepare writing to __managed__

    //::depth = currentTrackingLevel->depth->GetData(MEMORYDEVICE_CUDA);
    //::viewIntrinsics = currentTrackingLevel->intrinsics;
    auto viewImageSize = currentTrackingLevel->depth->noDims;
    //::T_g_k = T_g_k_estimate;
    std::auto_ptr<CoordinateSystem> depthCoordinateSystemEstimate(new CoordinateSystem(T_g_k_estimate));
    depthImage = new DepthImage(
        currentTrackingLevel->depth, 
        depthCoordinateSystemEstimate.get(),
        currentTrackingLevel->intrinsics
        );
    assert(depthImage->imgSize() == currentTrackingLevel->depth->noDims);
    ::accu.reset();
    ::distThresh = currentTrackingLevel->distanceThreshold;

    dim3 blockSize(16, 16); // must equal REDUCE_BLOCK_SIZE
    assert(16 * 16 == REDUCE_BLOCK_SIZE);

    dim3 gridSize(
        (int)ceil((float)viewImageSize.x / (float)blockSize.x),
        (int)ceil((float)viewImageSize.y / (float)blockSize.y));

    LAUNCH_KERNEL(depthTrackerOneLevel_g_rt_device_main, gridSize, blockSize);

    cudaDeviceSynchronize(); // for later access of accu
    return accu;
}

/// evaluate error function at the supplied T_g_k_estimate, 
/// compute sum_ATb and sum_AT_A, the system we need to solve to compute the
/// next update step (note: this system is not yet solved and we don't know the new energy yet!)
/// \returns noValidPoints
int ComputeGandH(
    float &f,
    float sum_ATb[6],
    float sum_AT_A[6][6],
    Matrix4f T_g_k_estimate) {
    AccuCell accu = ComputeGandH(T_g_k_estimate);

    memcpy(sum_ATb, accu.ATb, sizeof(float) * 6);
    assert(sum_ATb[4] == accu.ATb[4]);
    memcpy(sum_AT_A, accu.AT_A, sizeof(float) * 6 * 6);
    assert(sum_AT_A[3][4] == accu.AT_A[3][4]);

    // Output energy -- if we have very few points, output some high energy
    f = (accu.noValidPoints > 100) ? sqrt(accu.f) / accu.noValidPoints : 1e5f;

    return accu.noValidPoints;
}

/// Solves hessian.step = nabla
/// \param delta output array of 6 floats 
/// \param hessian 6x6
/// \param delta 3 or 6
/// \param nabla 3 or 6
/// \param shortIteration whether there are only 3 parameters
void ComputeDelta(float step[6], float nabla[6], float hessian[6][6])
{
    for (int i = 0; i < 6; i++) step[i] = 0;

    if (shortIteration())
    {
        // Keep only upper 3x3 part of hessian
        float smallHessian[3][3];
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) smallHessian[r][c] = hessian[r][c];

        ORUtils::Cholesky::solve((float*)smallHessian, 3, nabla, step);
        
        // check
        float result[3];
        matmul((float*)smallHessian, step, result, 3, 3);
        for (int r = 0; r < 3; r++)
            assert(abs(result[r] - nabla[r]) / abs(result[r]) < 0.0001);
    }
    else
    {
        ORUtils::Cholesky::solve((float*)hessian, 6, nabla, step);
    }
}

bool HasConverged(float *step)
{
    // Compute ||step||_2^2
    float stepLength = 0.0f;
    for (int i = 0; i < 6; i++) stepLength += step[i] * step[i];

    // heuristic? Why /6?
    if (sqrt(stepLength) / 6 < depthTrackerTerminationThreshold) return true; //converged

    return false;
}

Matrix4f ComputeTinc(const float *delta)
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
#include <memory>
/** Performing ICP based depth tracking.
Implements the original KinectFusion tracking algorithm.

c.f. newcombe_etal_ismar2011.pdf section "Sensor Pose Estimation"

6-d parameter vector "x" is (beta, gamma, alpha, tx, ty, tz)
*/
/// \file c.f. newcombe_etal_ismar2011.pdf, Sensor Pose Estimation section
void TrackCamera(
    ITMTrackingState *trackingState,
    const ITMView *view)
{
    /// Initialize one tracking event base data. Init hierarchy level 0 (finest).
    cudaDeviceSynchronize(); // prepare writing to __managed__
    auto sceneIntrinsics =
        view->calib->intrinsics_d.projectionParamsSimple.all;

    std::auto_ptr<CoordinateSystem> lastFrameEyeCoordinateSystem(new CoordinateSystem(
        trackingState->pointCloud->pose_pointCloud->GetInvM()
        ));
    lastFrameICPMap = new RayImage(
        trackingState->pointCloud->locations, 
        trackingState->pointCloud->normals,
        CoordinateSystem::global(), // locations and normals are both in world-coordinates

        lastFrameEyeCoordinateSystem.get(),
        sceneIntrinsics
        );

    /// Init image hierarchy levels
    trackingLevels[0]->depth = view->depth;
    trackingLevels[0]->intrinsics = sceneIntrinsics;
    for (int i = 1; i < trackingLevels.size(); i++)
    {
        TrackingLevel* currentLevel = trackingLevels[i];
        TrackingLevel* previousLevel = trackingLevels[i - 1];

        FilterSubsampleWithHoles(currentLevel->depth, previousLevel->depth);
        cudaDeviceSynchronize();

        currentLevel->intrinsics = previousLevel->intrinsics * 0.5f;
    }

    // Coarse to fine
    for (int levelId = trackingLevels.size() - 1; levelId >= 0; levelId--)
    {
        currentTrackingLevel = trackingLevels[levelId];
        if (iterationType() == TRACKER_ITERATION_NONE) continue;

        // T_{k,g} transforms global (g) coordinates to eye or view coordinates of the k-th frame
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
        ITMPose least_energy_T_k_g_estimate(*T_k_g_estimate);

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
            float new_sum_AT_A[6][6];
            noValidPoints = ComputeGandH(f_new, new_sum_ATb, new_sum_AT_A, T_g_k_estimate);
            // ]]

            float least_energy_sum_AT_A[6][6],
                damped_least_energy_sum_AT_A[6][6];
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

                // Preconditioning: Normalize by noValidPoints
                for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) least_energy_sum_AT_A[i][j] = new_sum_AT_A[i][j] / noValidPoints;
                for (int i = 0; i < 6; ++i) least_energy_sum_ATb[i] = new_sum_ATb[i] / noValidPoints;

                // Accept and decrease damping
                lambda /= 10.0f;
            }
            // Solve normal equations

            // Apply levenberg-marquart style damping (multiply diagonal of ATA by 1.0f + lambda)
            for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) damped_least_energy_sum_AT_A[i][j] = least_energy_sum_AT_A[i][j];
            for (int i = 0; i < 6; ++i) damped_least_energy_sum_AT_A[i][i] *= 1.0f + lambda;

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

    delete lastFrameICPMap;
    lastFrameICPMap = 0;
}
// 540        