#include "itmlibdefines.h"
#include "itmrepresentationaccess.h"
#include "LightingModel.h"
/// c.f. Geometry Refinement and Albedo Estimation on Signed Distance Fields
const float wGradientShading, wRegularize, wStabilize, wAlbedoRegularize;

GPU_ONLY Vector3f B(const LightingModel& lm, Vector3i v) {
    Vector3f normal = readVoxel();
    lm.getReflectedIrradiance();
}

/// computes refinement energy for given voxel
GPU_ONLY float ERefine(Vector3i v_pos) {
    return wgradientShading * EGradientShading()
        + wRegularize * ERegularize()
        + wStabilize * EStabilize()
        + wAlbedoRegularize * EAlbedoRegularize();
}

/// Refines the geometry and estimates albedo on the scene (== current hierarchy level)
class GeometryRefinement {
    // initialize albedo to uniform white

};