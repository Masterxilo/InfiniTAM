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


#include "lightingmodel.h"


/// compute H0(n(v_i)) ... H8(n(v_i))
/// for voxel v_i given by pos
/// 8 == b^2
/// \returns false when this cannot be computed
GPU_ONLY bool computeLightingEstimate(
    const ITMVoxelBlock* const localVBA,
    const typename ITMVoxelBlockHash::IndexData * const voxelIndex,
    const Vector3i pos, //!< [in]
    float H_n_out[LightingModel::b2] //!< [out] H0(n(v_i)) ... H8(n(v_i))
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
    const Vector3i pos,//!< [in]
    MatrixSQX<float, LightingModel::b2>& ai_aiT,//!< [out]
    float ai_bi[LightingModel::b2]//!< [out] a_i * b_i (a_i is a column of A^T)
    ) {
    // TODO handle isFound
    // compute ai
    float ai[LightingModel::b2];
    computeLightingEstimate(localVBA, voxelIndex,
        pos, ai);

    // compute bi
    bool isFound;
    ITMVoxel v = readVoxel(localVBA, voxelIndex, pos, isFound);
    float bi = v.intensity() / v.luminanceAlbedo; // I(v) / a(v)

    // compute ai * bi vector (these are later summed up to give A^T * b)
    for (int i = 0; i < LightingModel::b2; i++) {
        ai_bi[i] = ai[i] * bi;
    }

    // each output column is the vector ai, multiplied by ai[column]
    for (int column = 0; column < LightingModel::b2; column++) {
        for (int row = 0; row < LightingModel::b2; row++) {
            ai_aiT.at(column, row) = ai[row] * ai[column];
        }
    }
}

///#define REDUCE_BLOCK_SIZE 256 // must be power of 2. Used for reduction of a sum.


/// launch this such that each thread handles one voxel (3-d)
/// and each thread block handles one voxel block (1-d)
/// we locally sum up A^TA and A^Tb and then put it into the global sum
KERNEL reductionLightingEstimate(
    const ITMVoxelBlock* const localVBA,
    const typename ITMVoxelBlockHash::IndexData * const voxelIndex,

    MatrixSQX<float, LightingModel::b2>* sum_AtA,
    float sum_Atb[LightingModel::b2],
    ) {
    assert(blockDim.x == blockDim.y && blockDim.y == blockDim.z && blockDim.x == sdf_voxel_block_size);
    assert(blockIdx.x < SDF_LOCAL_BLOCK_NUM);
    assert(gridDim.x == SDF_LOCAL_BLOCK_NUM);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);
    
    ITMVoxelBlock* vb = localVBA[blockIdx.x];
    if (vb->pos == illegal) return;

    Vector3i pos = vb->pos * sdf_voxel_block_size + 
        Vector3i(threadIdx.x, threadIdx.y, threadIdx.z)
        ;


    // local computation
    MatrixSQX<float, LightingModel::b2>& ai_aiT;
    float ai_bi[LightingModel::b2];

    computeLightingEstimateAtColumn(
        localVBA,
        voxelIndex,
        pos,//!< [in]
        ai_aiT,//!< [out]
        ai_bi);//!< [out] a_i * b_i (a_i is a column of A^T)

    // reduction within block
    __syncthreads();
    __shared__ MatrixSQX<float, LightingModel::b2> shared_sum_AtA[sdf_block_size3 / 2];
    __shared__ MatrixSQX<float, LightingModel::b2> shared_sum_Atb[sdf_block_size3 / 2][LightingModel::b2];
    
    __syncthreads();
    // integrate with total sum (only done by thread 0)
        if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) return;

    for (int i = 0; i < LightingModel::b2; i++) {
        atomicAdd(&sum_Atb[i], shared_sum_Atb[0][i]);
            

        for (int j = 0; j < LightingModel::b2; j++) {
            atomicAdd(&sum_AtA.at(j, i), ai_aiT.at(j, i));
        }
    }
}


class LightingEstimation {
public:
private:
};