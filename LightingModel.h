
#include "ITMLibDefines.h"

using namespace ORUtils;
#include <array>
#include "cholesky.h" 
class LightingModel {

public:
    static const int b2 = 9;

    float getReflectedIrradiance(float albedo, Vector3f normal) const {
        assert(albedo >= 0);
        float o = 0;
        for (int m = 0; m < b2; m++) {
            o += l[m] * sphericalHarmonicHi(m, normal);
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