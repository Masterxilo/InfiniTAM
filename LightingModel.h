#pragma once
#include "ITMLibDefines.h"

using namespace ORUtils;
#include <array>
#include "cholesky.h" 

class LightingModel {

public:
    static const int b2 = 9;

    /// \f[a \sum_{m = 1}^{b^2} l_m H_m(n)\f]
    /// \f$v\f$ is some voxel (inside the truncation band)
    // TODO wouldn't the non-refined voxels interfere with the updated, refined voxels, 
    // if we just cut them off hard from the computation?
    float getReflectedIrradiance(float albedo, //!< \f$a(v)\f$
        Vector3f normal //!< \f$n(v)\f$
        ) const {
        assert(albedo >= 0);
        float o = 0;
        for (int m = 0; m < b2; m++) {
            o += l[m] * sphericalHarmonicHi(m, normal);
        }
        return albedo * o;
    }

    // original paper uses svd to compute the solution to the linear system, but how this is done should not matter
    LightingModel(std::array<float, b2>& l) : l(l){ 
        assert(l[0] > 0); // constant term should be positive - otherwise the lighting will be negative in some places (?)
    }
    LightingModel(const LightingModel& m) : l(m.l){}

    static CPU_AND_GPU float sphericalHarmonicHi(int i, Vector3f n) {
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

    const std::array<float, b2> l;

};