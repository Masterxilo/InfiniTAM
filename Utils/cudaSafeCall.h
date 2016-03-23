#pragma once

#include <stdio.h>
#include <string> // wstring 
extern void logger(std::string s);

#include "CUDADefines.h"
#if UNIT_TESTING
#include "CppUnitTest.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
extern std::wstring _errmsg;
#endif

// Automatically wrap some functions in cudaSafeCall
#ifdef __CUDACC__ // hack to hide these from intellisense
#define cudaDeviceSynchronize(...) cudaSafeCall(cudaDeviceSynchronize(__VA_ARGS__))
#define cudaMalloc(...) cudaSafeCall(cudaMalloc(__VA_ARGS__))
#define cudaFree(...) cudaSafeCall(cudaFree(__VA_ARGS__))
#define cudaMallocManaged(...) cudaSafeCall(cudaMallocManaged(__VA_ARGS__))
#endif

// cudaSafeCall is an expression that evaluates to 
// 0 when err is cudaSuccess (0), such that cudaSafeCall(cudaSafeCall(cudaSuccess)) will not block
// this is important because we might have legacy code that explicitly does 
// cudaSafeCall(cudaDeviceSynchronize());
// but we extended cudaDeviceSynchronize to include this already, giving
// cudaSafeCall(cudaSafeCall(cudaDeviceSynchronize()))

// it debug-breaks and returns 
bool cudaSafeCallImpl(cudaError err, const char * const expr, const char * const file, const int line);
#if UNIT_TESTING
#define cudaSafeCall(err)\
    !(cudaSafeCallImpl((cudaError)(err), #err, __FILE__, __LINE__) || ([]() {if (!IsDebuggerPresent()) {Assert::IsTrue(false, _errmsg.c_str());} else DebugBreak(); return true;})() )
#else
// If err is cudaSuccess, cudaSafeCallImpl will return true, early-out of || will make DebugBreak not evaluated.
// The final expression will be 0.
// Otherwise we evaluate debug break, which returns true as well and then return 0.
#define cudaSafeCall(err) \
    !(cudaSafeCallImpl((cudaError)(err), #err, __FILE__, __LINE__) || ([]() {DebugBreak(); return true;})() )
#endif
