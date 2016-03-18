/// \file Depth Tracker, c.f. newcombe_etal_ismar2011.pdf Sensor Pose Estimation
// The current discussion ignores the optimizations/special iterations with 
// rotation estimation only ("At the coarser levels we optimise only for the rotation matrix R.")
// Also 'translation only' is not used for the depth ICP tracker.
#pragma once

#include "../../Utils/ITMLibDefines.h"
#include "ITMPixelUtils.h"


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
_CPU_AND_GPU_CODE_ inline bool computePerPointGH_Depth_Ab(
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
    if (p_km1.w < 0.0f) return false;

    // n_km1 := N_{k-1}(\hat u)
    Vector4f n_km1 = interpolateBilinear<Vector4f, WITH_HOLES>(normalsMap, hat_u, sceneImageSize);

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

/// Wrapper for computePerPointGH_Depth_Ab
/// \param AT_b a 6 x 1 column vector (3 x 1)
/// \param AT_A a 6 x 6 matrix (3 x 3) [because this will be a symmetric matrix, we store only the unique elements in the lower right triangular part of it] // woot, so smart
/// \returns whether u is a valid point (\f$\Omega(u) \neq \mbox{null}\f$)
/// \param x,y \f$u\f$
template<TrackerIterationType iterationType>
_CPU_AND_GPU_CODE_ inline bool computePerPointGH_Depth(
    THREADPTR(float) *AT_b,
    THREADPTR(float) *AT_A_tri,
    THREADPTR(float) &localF,
    const CONSTPTR(int) & x, const CONSTPTR(int) & y,
    const CONSTPTR(float) &depth,
    const CONSTPTR(Vector2i) & viewImageSize,
    const CONSTPTR(Vector4f) & viewIntrinsics,
    const CONSTPTR(Vector2i) & sceneImageSize,
    const CONSTPTR(Vector4f) & sceneIntrinsics,
    const CONSTPTR(Matrix4f) & T_g_k_estimate,
    const CONSTPTR(Matrix4f) & scenePose,
    const CONSTPTR(Vector4f) *pointsMap,
    const CONSTPTR(Vector4f) *normalsMap,
    const float distThresh)
{
    const int noPara = iterationType == TRACKER_ITERATION_BOTH ? 6 : 3;
    // Column vector
    float AT[noPara];
	float b;

    if (!computePerPointGH_Depth_Ab<iterationType>(
        AT, b, x, y, depth, 
        viewImageSize, viewIntrinsics, sceneImageSize, sceneIntrinsics, 
        T_g_k_estimate, scenePose, pointsMap, normalsMap, distThresh))
        return false;

    // apply ||.||_2^2 to b to obtain local contribution to cost function f
    localF = b * b;

    // Compute AT_b and AT_A
    for (int r = 0, counter = 0; r < noPara; r++)
	{
        AT_b[r] = AT[r] * b;

        for (int c = 0; c <= r; c++)
            AT_A_tri[counter++] = AT[r] * AT[c]; // row fixed, column counting to the right
	}

	return true;
}

