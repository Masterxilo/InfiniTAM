// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include <cstdio>

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define CPU_AND_GPU __device__	// for CUDA device code
#else
#define CPU_AND_GPU 
#endif

// Documenting types of pointers/references
// pointer is valid only for one thread
#define THREADPTR(x) x
// pointer can be dereferenced on gpu
#define DEVICEPTR(x) x
// __shared__ memory
#define THREADGROUPPTR(x) x
// __const__ memory
#define CONSTPTR(x) x