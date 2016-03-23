#pragma once

#ifndef __CUDACC__
#error can only be included by .cu files
#endif

#ifndef __CUDA_ARCH__
#define __CUDACC_RTC__ // hack to calm intellisense about cuda intrinsics (__syncthreads et. al)
#endif

#ifndef _WIN64
#error cudaMallocManaged requires 64 bits. Get rid of legacy 32 bit ;)
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 500
#error Always use the latest cuda arch. Old versions dont support any amount of thread blocks being submitted at once.
#endif

#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#pragma comment(lib,"cudart")

// Some macros to identify cuda concepts (e.g. pointer/function types)
#define GPU_ONLY __device__
#define GPU(mem) mem
#define KERNEL __global__ void
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define CPU_AND_GPU __device__	// for CUDA device code
#else
#define CPU_AND_GPU  // alternatively: __host__ __device__, note that this is different for functions using dynamic parallelism!
#endif

#ifndef __CUDACC__
// hack to make intellisense shut up
#define LAUNCH_KERNEL(...) ((void)0)

#else
#if UNIT_TESTING

#define LAUNCH_KERNEL(kernelFunction, gridDim, blockDim, ...) {\
cudaSafeCall(cudaGetLastError());\
kernelFunction << <gridDim, blockDim >> >(__VA_ARGS__);\
cudaSafeCall(cudaGetLastError());\
cudaSafeCall(cudaDeviceSynchronize());\
}

#else

#define LAUNCH_KERNEL(kernelFunction, gridDim, blockDim, ...) {\
cudaSafeCall(cudaGetLastError());\
kernelFunction << <gridDim, blockDim >> >(__VA_ARGS__);\
cudaSafeCall(cudaGetLastError());\
cudaSafeCall(cudaDeviceSynchronize()); /*TODO you do not want this in release code/it alters the execution order 
but it catches kernel errors (early)*/\
}
#endif

#endif

#include "cudaSafeCall.h"
#include "MyAssert.h"