/// c.f. chapter "Lighting Estimation with Signed Distance Fields"

#include "itmlibdefines.h"
#include "cudadefines.h" 
#include "lightingmodel.h"
#include "Scene.h"
#include "itmrepresentationaccess.h" 
#include "constructAndSolve.h"
/*

#include "matrix.h"
#include "image.h"
#include "cholesky.h"

#include "ITMLibDefines.h"
#include <array>


//
*/
/// for constructAndSolve
struct ConstructLightingModelEquationRow {
    // Amount of columns, should be small
    static const unsigned int m = LightingModel::b2;

    /*not really needed */
    struct ExtraData {
        // User specified payload to be summed up alongside:
        uint count;

        // Empty constructor must generate neutral element
        CPU_AND_GPU ExtraData() : count(0) {}

        static GPU_ONLY ExtraData add(const ExtraData& l, const ExtraData& r) {
            ExtraData o;
            o.count = l.count + r.count;
            return o;
        }
        static GPU_ONLY void atomicAdd(DEVICEPTR(ExtraData&) result, const ExtraData& integrand) {
            ::atomicAdd(&result.count, integrand.count);
        }
    };

    /// should be executed with (blockIdx.x/2) == valid localVBA index (0 ignored) 
    /// annd blockIdx.y,z from 0 to 1 (parts of one block)
    /// and threadIdx <==> voxel localPos / 2..
    static GPU_ONLY bool generate(const uint i, VectorX<float, m>& out_ai, float& out_bi/*[1]*/, ExtraData& extra_count /*not really needed */) {
        const uint blockSequenceId = blockIdx.x/2;
        if (blockSequenceId == 0) return false; // unused
        assert(blockSequenceId < SDF_LOCAL_BLOCK_NUM);

        assert(blockSequenceId < Scene::getCurrentScene()->voxelBlockHash->getLowestFreeSequenceNumber());

        assert(threadIdx.x < SDF_BLOCK_SIZE / 2 && 
            threadIdx.y < SDF_BLOCK_SIZE / 2 &&
            threadIdx.z < SDF_BLOCK_SIZE / 2);

        assert(blockIdx.y <= 1);
        assert(blockIdx.z <= 1);
        // voxel position
        const Vector3i localPos = Vector3i(threadIdx_xyz) + Vector3i(blockIdx.x % 2, blockIdx.y % 2, blockIdx.z % 2) * 4;

        assert(localPos.x >= 0  &&
               localPos.y >= 0 &&
               localPos.z >= 0);
        assert(localPos.x < SDF_BLOCK_SIZE  &&
            localPos.y < SDF_BLOCK_SIZE &&
            localPos.z < SDF_BLOCK_SIZE );

        ITMVoxelBlock* voxelBlock = Scene::getCurrentScene()->getVoxelBlockForSequenceNumber(blockSequenceId);

        const ITMVoxel* voxel = voxelBlock->getVoxel(localPos);
        const Vector3i globalPos = (voxelBlock->pos.toInt() * SDF_BLOCK_SIZE + localPos);
        /*
        const Vector3i globalPos = vb->pos.toInt() * SDF_BLOCK_SIZE;

        const THREADPTR(Point) & voxel_pt_world =  Point(
            CoordinateSystem::global(),
            (globalPos.toFloat() + localPos.toFloat()) * voxelSize
            ));

        .toFloat();
        Vector3f worldPos = CoordinateSystems::global()->convert(globalPos);
        */
        const float worldSpaceDistanceToSurface = abs(voxel->getSDF() * mu);
        assert(worldSpaceDistanceToSurface <= mu);

        // Is this voxel within the truncation band? Otherwise discard this term (as unreliable for lighting calculation)
        if (worldSpaceDistanceToSurface > t_shell) return false;

        // return if we cannot compute the normal
        bool found = true;
        const Vector3f normal = computeSingleNormalFromSDFByForwardDifference(globalPos, found);
        if (!found) return false;
        assert(abs(length(normal) - 1) < 0.01);

        // i-th (voxel-th) row of A shall contain H_{0..b^2-1}(n(v))
        for (int i = 0; i < LightingModel::b2; i++) {
            out_ai[i] = LightingModel::sphericalHarmonicHi(i, normal);
        }

        // corresponding entry of b is I(v) / a(v)
        out_bi = voxel->intensity() / voxel->luminanceAlbedo;
        assert(out_bi >= 0 && out_bi <= 1);

        // TODO not really needed
        extra_count.count = 1;
        return true;
    }

};

// todo should we really discard the existing lighting model the next time? maybe we could use it as an initialization
// when solving
LightingModel estimateLightingModel() {
    assert(Scene::getCurrentScene());
    // Maximum number of entries

    const int validBlockNum = Scene::getCurrentScene()->voxelBlockHash->getLowestFreeSequenceNumber();

    auto gridDim = dim3(validBlockNum * 2, 2, 2); 
    auto blockDim = dim3(SDF_BLOCK_SIZE / 2, SDF_BLOCK_SIZE / 2, SDF_BLOCK_SIZE / 2); // cannot use full SDF_BLOCK_SIZE: too much shared data (in reduction)

    const int n = validBlockNum * SDF_BLOCK_SIZE3; // maximum number of entries: total amount of currently allocated voxels (unlikely)
    assert(n == volume(gridDim) * volume(blockDim));

    ConstructLightingModelEquationRow::ExtraData extra_count;
    auto l_harmonicCoefficients = constructAndSolve<ConstructLightingModelEquationRow>(n, gridDim, blockDim, extra_count);
    assert(extra_count.count <= n); // sanity check
    assert(l_harmonicCoefficients.size() == LightingModel::b2);

    std::array<float, LightingModel::b2> l_harmonicCoefficients_a;
    for (int i = 0; i < LightingModel::b2; i++)
        l_harmonicCoefficients_a[i] = assertFinite( l_harmonicCoefficients[i] );
    
    LightingModel lightingModel(l_harmonicCoefficients_a);
    return lightingModel;
}

void estimateLightingModel_() {
    estimateLightingModel();
}