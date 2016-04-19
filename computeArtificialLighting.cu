#include "computeArtificialLighting.h"
#include "Scene.h"
#include "itmrepresentationaccess.h"
static __managed__ Vector3f lightNormal;
struct ComputeLighting {
    doForEachAllocatedVoxel_process() {
        // skip voxels without computable normal
        bool found = true;
        const Vector3f normal = computeSingleNormalFromSDFByForwardDifference(globalPos, found);
        if (!found) return;

        float cos = max(0.f, dot(normal, lightNormal));

        v->clr = Vector3u(cos * 255, cos * 255, cos * 255);
        v->w_color = 1;
    }
};

void computeArtificialLighting(Vector3f lightNormal) {
    cudaDeviceSynchronize();
    assert(abs(length(lightNormal) - 1) < 0.1);
    assert(Scene::getCurrentScene());

    ::lightNormal = lightNormal;
    Scene::getCurrentScene()->doForEachAllocatedVoxel<ComputeLighting>();

    cudaDeviceSynchronize();
}


void computeArtificialLighting_() {
    computeArtificialLighting(Vector3f(0, 1, 0));
}

int* v = (int*)computeArtificialLighting_;