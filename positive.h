#include "itmlibdefines.h"
#include <float.h>
/**
Any non-infinite, non-NAN, non-negative float.
IEEE -0 is also prohibited (even though
assert(_fpclass(-0.f) == _FPCLASS_NZ);
assert(-0.f >= 0.f);
holds)
*/
class positive {
private:
    const float value;

public:
    /// Utility: -0 is converted to +0
    static float nz2pz(float x) {
        if (_fpclass(x) == _FPCLASS_NZ)
            return +0.f;
        return x;
    }
    /// Value verified at runtime if assertions are enabled
    explicit positive(float x) : value(nz2pz(x)) {
        assert(_fpclass(value) == _FPCLASS_PD || _fpclass(value) == _FPCLASS_PN || _fpclass(value) == _FPCLASS_PZ);
    }

    // Trick to avoid initialization with anything but float: declare but dont define for all basic types:
    explicit positive(unsigned char);
    explicit positive(char);
    explicit positive(unsigned short);
    explicit positive(short);
    explicit positive(unsigned int);
    explicit positive(int);
    explicit positive(unsigned long);
    explicit positive(long);
    /* explicit positive(unsigned long long);
    explicit positive(long long);*/
    explicit positive(double);
    explicit positive(long double);
    explicit positive(wchar_t);
};
