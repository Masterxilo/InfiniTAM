if (depth <= 1e-8f) return false;

// (1) Grab the corresponding points by projective data association
// p_k := T_{g,k}V_k(u) = V_k^g(u)
Vector4f p_k = T_g_k *
depthTo3D(viewIntrinsics, x, y, depth); // V_k(u) = D_k(u)K^{-1}u 
p_k.w = 1.0f;

// hat_u = \pi(K T_{k-1,g} T_{g,k}V_k(u) )
Vector2f hat_u;
if (!projectExtraBounds(sceneIntrinsics, sceneImageSize,
    T_km1_g * p_k, // T_{k-1,g}V_k^g(u)
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