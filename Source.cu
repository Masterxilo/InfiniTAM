/// c.f. chapter "Lighting Estimation with Signed Distance Fields"

#include "matrix.h"
#include "itmlibdefines.h"
#include "image.h" 
#include "cholesky.h" 
#include "cudadefines.h" 
#include "itmrepresentationaccess.h" 
#include "ITMLibDefines.h"
#include <array>
using namespace ORUtils;
using namespace ITMLib;
using namespace ITMLib::Objects;

class LightingModel {

public:
    static const int b2 = 9;

    float getReflectedIrradiance(float albedo, Vector3f normal) {
        float o = 0;
        for (int m = 0; m < b2; m++) {
            o += l[m] * sphericalHarmonicHi(m, n);
        }
        return albedo * o;
    }

    static LightingModel constructAsSolution(
        MatrixSQX<float, b2> AtA, VectorX<float, b2> Atb) {
        // original paper uses svd
        std::array<float, b2> l;
        Cholesky::solve(AtA.m, b2, Atb, l.data());

        return LightingModel(l);
    }
    
    static float sphericalHarmonicHi(int i, Vector3f n) {
        assert(i >= 0 && i < b2);
        switch (i) {
        case 0: return 1.f;
        case 1: return n.y;
        case 2: return n.z;
        case 3: return n.x;
        case 4: return n.x * n.y;
        case 5: return n.y * n.z;
        case 6: return -n.x * n.x - n.y * n.y + 2 * n.z * n.z;
        case 7: return n.z * n.x;
        case 8: return n.x - n.y * n.y;
        default: assert(false);
        }
        return 0;
    }

private:
    LightingModel(std::array<float, b2>& l) : l(l){}

    const std::array<float, b2> l;
    
};

/// compute H0(n(v_i)) ... H8(n(v_i))
/// for voxel v_i given by pos
/// 8 == b^2
/// \returns false when this cannot be computed
GPU_ONLY bool computeLightingEstimate(
    const ITMVoxelBlock* const localVBA,
    const typename ITMVoxelBlockHash::IndexData * const voxelIndex,
    const Vector3i pos, //!< [in]
    float* H_n_out //!< [out] H0(n(v_i)) ... H8(n(v_i))
    ) {
    bool isFound;
    Vector3f n = computeSingleNormalFromSDFByForwardDifference(localVBA, voxelIndex, pos, isFound);
    if (!isFound) return false;

    for (int i = 0; i < LightingModel::b2; i++) {
        H_n_out[i] = LightingModel::sphericalHarmonicHi(i, n);
    }
}

/// compute the matrix a_i a_i^T
// for the voxel i, where a_i is one column of A^T
GPU_ONLY bool computeLightingEstimateAtColumn(
    const ITMVoxelBlock* const localVBA,
    const typename ITMVoxelBlockHash::IndexData * const voxelIndex,
    const Vector3i pos//!< [in]
    ) {
    // TODO handle isFound

}


class LightingEstimation {
public:
private:
};