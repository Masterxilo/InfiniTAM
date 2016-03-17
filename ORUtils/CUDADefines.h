// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#pragma comment( lib, "cuda.lib" )
#pragma comment( lib, "cudart.lib" )

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  include <windows.h>
#endif

#ifndef ORcudaSafeCall
#define ORcudaSafeCall(err) ORUtils::__cudaSafeCall(err, __FILE__, __LINE__)

namespace ORUtils {

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
		printf("%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
                file, line, cudaGetErrorString(err) );
        exit(-1);
    }
}

}

#endif

