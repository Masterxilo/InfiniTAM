#pragma once // TODO make functions inline or otherwise avoid redefinition in other modules (or use only from one module)

#include "matrix.h"
#include "vector.h"
#include "cudadefines.h"

#include <iostream>
using namespace std;
/// Exchange information
__managed__ long long transform_reduce_resultMemory[100 * 100]; // use only in one module

template <typename T>
CPU_AND_GPU T& transform_reduce_result() {
    assert(sizeof(T) <= sizeof(transform_reduce_resultMemory));
    return *(T*)transform_reduce_resultMemory;
}

/// Constructor must define
/// Constructor::ExtraData(), add, atomicAdd
/// static const uint Constructor::m
/// bool Constructor::generate(const uint i, VectorX<float, m>& , float& bi)
template<typename Constructor>
struct AtA_Atb_Add {
    static const uint m = Constructor::m;
    typedef typename MatrixSQX<float, m> AtA;
    typedef typename VectorX<float, m> Atb;
    typedef typename Constructor::ExtraData ExtraData;

    struct ElementType {
        typename AtA _AtA;
        typename Atb _Atb;
        typename ExtraData _extraData;
        CPU_AND_GPU ElementType() {} // uninitialized on purpose
        CPU_AND_GPU ElementType(AtA _AtA, Atb _Atb, ExtraData _extraData) : _AtA(_AtA), _Atb(_Atb), _extraData(_extraData) {}
    };

    static GPU_ONLY bool generate(const uint i, ElementType& out) {
        // Some threads contribute zero
        VectorX<float, m> ai; float bi;
        if (!Constructor::generate(i, ai, bi, out._extraData)) return false;

        // Construct ai_aiT, ai_bi
        out._AtA = MatrixSQX<float, m>::make_aaT(ai);
        out._Atb = ai * bi;
        return true;
    }

    static CPU_AND_GPU ElementType neutralElement() {
        return ElementType(
            AtA::make_zeros(),
            Atb::make_zeros(),
            ExtraData());
    }

    static GPU_ONLY ElementType operate(ElementType & l, ElementType & r) {
        return ElementType(
            l._AtA + r._AtA,
            l._Atb + r._Atb,
            ExtraData::add(l._extraData, r._extraData)
            );
    }

    static GPU_ONLY void atomicOperate(DEVICEPTR(ElementType&) result, ElementType & integrand) {
        for (int r = 0; r < m*m; r++)
            atomicAdd(
            &result._AtA[r],
            integrand._AtA[r]);

        for (int r = 0; r < m; r++)
            atomicAdd(
            &result._Atb[r],
            integrand._Atb[r]);

        ExtraData::atomicAdd(result._extraData, integrand._extraData);
    }
};



CPU_AND_GPU unsigned int toLinearId(const dim3 dim, const uint3 id) {
    return dim.x * dim.y * id.z + dim.x * id.y + id.x;
}
CPU_AND_GPU unsigned int toLinearId2D(const dim3 dim, const uint3 id) {
    assert(dim.z == 1);
    return toLinearId(dim, id);
}

GPU_ONLY uint linear_threadIdx() {
    return toLinearId(blockDim, threadIdx);
}
GPU_ONLY uint linear_blockIdx() {
    return toLinearId(gridDim, blockIdx);
}

CPU_AND_GPU unsigned int volume(dim3 d) {
    return d.x*d.y*d.z;
}

GPU_ONLY uint linear_global_threadId() {
    return linear_blockIdx() * volume(blockDim) + linear_threadIdx();
}

const int MAX_REDUCE_BLOCK_SIZE = 4*4*4; // TODO this actually depends on shared memory demand of Constructor (::m, ExtraData etc.) -- template-specialize on it?

template<class Constructor>
KERNEL transform_reduce_if_device(const uint n) {
    const uint tid = linear_threadIdx();
    assert(tid < MAX_REDUCE_BLOCK_SIZE, "tid %d", tid);
    const uint i = linear_global_threadId();

    const uint REDUCE_BLOCK_SIZE = volume(blockDim);

    // Whether this thread block needs to compute a prefix sum
    __shared__ bool shouldPrefix;
    shouldPrefix = false;
    __syncthreads();

    __shared__ typename Constructor::ElementType reduced_elements[MAX_REDUCE_BLOCK_SIZE]; // this is pretty heavy on shared memory!

    typename Constructor::ElementType& ei = reduced_elements[tid];
    if (i >= n || !Constructor::generate(i, ei)) {
        ei = Constructor::neutralElement();
    }
    else
        shouldPrefix = true;

    __syncthreads();

    if (!shouldPrefix) return;
    // only if at least one thread in the thread block gets here do we do the prefix sum.

    // tree reduction into reduced_elements[0]
    for (int offset = REDUCE_BLOCK_SIZE / 2; offset >= 1; offset /= 2) {
        if (tid >= offset) return;

        reduced_elements[tid] = Constructor::operate(reduced_elements[tid], reduced_elements[tid + offset]);

        __syncthreads();
    }

    assert(tid == 0);

    // Sum globally, using atomics
    auto& result = transform_reduce_result<Constructor::ElementType>();
    Constructor::atomicOperate(result, reduced_elements[0]);
}

CPU_AND_GPU bool isPowerOf2(unsigned int x) {
#if GPU_CODE
    return __popc(x) <= 1;
#else
    return __popcnt(x) <= 1;
#endif
}

/**
Constructor must provide:
* Constructor::ElementType
* Constructor::generate(i) which will be called with i from 0 to n and may return false causing its result to be replaced with
* CPU_AND_GPU Constructor::neutralElement()
* Constructor::operate and atomicOperate define the binary operation

Constructor::generate is run once in each CUDA thread.
The division into threads *can* be manually specified -- doing so will not significantly affect the outcome if the Constructor is agnostic to threadIdx et.al.
If gridDim and/or blockDim are nonzero, it will be checked for conformance with n (must be bigger than or equal).
gridDim can be left 0,0,0 in which case it is computed as ceil(n/volume(blockDim)),1,1.

volume(blockDim) must be a power of 2 (for reduction) and <= MAX_REDUCE_BLOCK_SIZE

Both gridDim and blockDim default to one-dimension.
*/
template<class Constructor>
Constructor::ElementType
transform_reduce_if(const uint n, dim3 gridDim = dim3(0, 0, 0), dim3 blockDim = dim3(0, 0, 0)) {
    // Configure kernel scheduling
    if (gridDim.x == 0) {
        assert(gridDim.y == gridDim.z && gridDim.z == 0);

        if (blockDim.x == 0) {
            assert(blockDim.y == blockDim.z && blockDim.z == 0);
            blockDim = dim3(MAX_REDUCE_BLOCK_SIZE, 1, 1); // default to one-dimension
        }

        gridDim = dim3(ceil(n / volume(blockDim)), 1, 1); // default to one-dimension
    }
    assert(isPowerOf2(volume(blockDim)));

    assert(volume(gridDim)*volume(blockDim) >= n, "must have enough threads to generate each element");
    assert(volume(blockDim) > 0);
    assert(volume(blockDim) <= MAX_REDUCE_BLOCK_SIZE);

    // Set up storage for result
    transform_reduce_result<typename Constructor::ElementType>() = Constructor::neutralElement();

    LAUNCH_KERNEL(
        (transform_reduce_if_device<Constructor>),
        gridDim, blockDim,
        n);
    cudaDeviceSynchronize();

    return transform_reduce_result<Constructor::ElementType>();
}

/**
Build A^T A and A^T b where A is <n x m and b has <n elements.

Row i (0-based) of A and b[i] are generated by bool Constructor::generate(uint i, VectorX<float, m> out_ai, float& out_bi).
It is thrown away if generate returns false.
*/
template<class Constructor>
AtA_Atb_Add<Constructor>::ElementType construct(const uint n, dim3 gridDim = dim3(0, 0, 0), dim3 blockDim = dim3(0, 0, 0)) {
    assert(Constructor::m < 100);
    return transform_reduce_if<AtA_Atb_Add<Constructor>>(n, gridDim, blockDim);
}

/// \see construct
template<class Constructor>
AtA_Atb_Add<Constructor>::Atb constructAndSolve(int n, dim3 gridDim, dim3 blockDim, Constructor::ExtraData& out_extra_sum) {
    auto result = construct<Constructor>(n, gridDim, blockDim);
    out_extra_sum = result._extraData;
    cout << result._AtA << endl;
    cout << result._Atb << endl;

    return Cholesky::solve(result._AtA, result._Atb);
}