// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMDepthTracker_CUDA.h"
#include "ITMCUDAUtils.h"
#include "../../DeviceAgnostic/ITMDepthTracker.h"
#include "../../../../ORUtils/CUDADefines.h"

using namespace ITMLib::Engine;


struct ITMDepthTracker_KernelParameters {
	ITMDepthTracker_CUDA::AccuCell *accu;
	float *depth;
	Matrix4f approxInvPose;
	Vector4f *pointsMap;
	Vector4f *normalsMap;
	Vector4f sceneIntrinsics;
	Vector2i sceneImageSize;
	Matrix4f scenePose;
	Vector4f viewIntrinsics;
	Vector2i viewImageSize;
	float distThresh;
};


// device functions

template<TrackerIterationType iterationType>
__device__ void depthTrackerOneLevel_g_rt_device_main(
    ITMDepthTracker::AccuCell *accu,
    float *depth,
    Matrix4f approxInvPose,
    Vector4f *pointsMap,
    Vector4f *normalsMap,
    Vector4f sceneIntrinsics,
    Vector2i sceneImageSize, Matrix4f scenePose, Vector4f viewIntrinsics, Vector2i viewImageSize,
    float distThresh)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

    int locId_local = threadIdx.x + threadIdx.y * blockDim.x;

    __shared__ float dim_shared1[256];
    __shared__ float dim_shared2[256];
    __shared__ float dim_shared3[256];
    __shared__ bool should_prefix;

    should_prefix = false;
    __syncthreads();

    const bool shortIteration = iterationType != TRACKER_ITERATION_BOTH;
    const int noPara = shortIteration ? 3 : 6;
    const int noParaSQ = shortIteration ? 3 + 2 + 1 : 6 + 5 + 4 + 3 + 2 + 1;
    float A[noPara]; float b;
    bool isValidPoint = false;

    if (x < viewImageSize.x && y < viewImageSize.y)
    {
        isValidPoint = computePerPointGH_Depth_Ab<iterationType>(A, b, x, y, depth[x + y * viewImageSize.x],
            viewImageSize, viewIntrinsics, sceneImageSize, sceneIntrinsics, approxInvPose, scenePose, pointsMap, normalsMap, distThresh);
        if (isValidPoint) should_prefix = true;
    }

    if (!isValidPoint) {
        for (int i = 0; i < noPara; i++) A[i] = 0.0f;
        b = 0.0f;
    }

    __syncthreads();

    if (!should_prefix) return;

    { //reduction for noValidPoints
        dim_shared1[locId_local] = isValidPoint;
        __syncthreads();

        if (locId_local < 128) dim_shared1[locId_local] += dim_shared1[locId_local + 128];
        __syncthreads();
        if (locId_local < 64) dim_shared1[locId_local] += dim_shared1[locId_local + 64];
        __syncthreads();

        if (locId_local < 32) warpReduce(dim_shared1, locId_local);

        if (locId_local == 0) atomicAdd(&(accu->noValidPoints), (int)dim_shared1[locId_local]);
    }

    { //reduction for energy function value
        dim_shared1[locId_local] = b*b;
        __syncthreads();

        if (locId_local < 128) dim_shared1[locId_local] += dim_shared1[locId_local + 128];
        __syncthreads();
        if (locId_local < 64) dim_shared1[locId_local] += dim_shared1[locId_local + 64];
        __syncthreads();

        if (locId_local < 32) warpReduce(dim_shared1, locId_local);

        if (locId_local == 0) atomicAdd(&(accu->f), dim_shared1[locId_local]);
    }

    __syncthreads();

    //reduction for nabla
    for (unsigned char paraId = 0; paraId < noPara; paraId += 3)
    {
        dim_shared1[locId_local] = b*A[paraId + 0];
        dim_shared2[locId_local] = b*A[paraId + 1];
        dim_shared3[locId_local] = b*A[paraId + 2];
        __syncthreads();

        if (locId_local < 128) {
            dim_shared1[locId_local] += dim_shared1[locId_local + 128];
            dim_shared2[locId_local] += dim_shared2[locId_local + 128];
            dim_shared3[locId_local] += dim_shared3[locId_local + 128];
        }
        __syncthreads();
        if (locId_local < 64) {
            dim_shared1[locId_local] += dim_shared1[locId_local + 64];
            dim_shared2[locId_local] += dim_shared2[locId_local + 64];
            dim_shared3[locId_local] += dim_shared3[locId_local + 64];
        }
        __syncthreads();

        if (locId_local < 32) {
            warpReduce(dim_shared1, locId_local);
            warpReduce(dim_shared2, locId_local);
            warpReduce(dim_shared3, locId_local);
        }
        __syncthreads();

        if (locId_local == 0) {
            atomicAdd(&(accu->ATb[paraId + 0]), dim_shared1[0]);
            atomicAdd(&(accu->ATb[paraId + 1]), dim_shared2[0]);
            atomicAdd(&(accu->ATb[paraId + 2]), dim_shared3[0]);
        }
    }

    __syncthreads();

    float localHessian[noParaSQ];

    for (unsigned char r = 0, counter = 0; r < noPara; r++)
    {
        for (int c = 0; c <= r; c++, counter++) localHessian[counter] = A[r] * A[c];
    }

    //reduction for hessian
    for (unsigned char paraId = 0; paraId < noParaSQ; paraId += 3)
    {
        dim_shared1[locId_local] = localHessian[paraId + 0];
        dim_shared2[locId_local] = localHessian[paraId + 1];
        dim_shared3[locId_local] = localHessian[paraId + 2];
        __syncthreads();

        if (locId_local < 128) {
            dim_shared1[locId_local] += dim_shared1[locId_local + 128];
            dim_shared2[locId_local] += dim_shared2[locId_local + 128];
            dim_shared3[locId_local] += dim_shared3[locId_local + 128];
        }
        __syncthreads();
        if (locId_local < 64) {
            dim_shared1[locId_local] += dim_shared1[locId_local + 64];
            dim_shared2[locId_local] += dim_shared2[locId_local + 64];
            dim_shared3[locId_local] += dim_shared3[locId_local + 64];
        }
        __syncthreads();

        if (locId_local < 32) {
            warpReduce(dim_shared1, locId_local);
            warpReduce(dim_shared2, locId_local);
            warpReduce(dim_shared3, locId_local);
        }
        __syncthreads();

        if (locId_local == 0) {
            atomicAdd(&(accu->AT_A_tri[paraId + 0]), dim_shared1[0]);
            atomicAdd(&(accu->AT_A_tri[paraId + 1]), dim_shared2[0]);
            atomicAdd(&(accu->AT_A_tri[paraId + 2]), dim_shared3[0]);
        }
    }
}

template<TrackerIterationType iterationType>
__global__ void depthTrackerOneLevel_g_rt_device(ITMDepthTracker_KernelParameters para)
{
    depthTrackerOneLevel_g_rt_device_main<iterationType>(para.accu, para.depth, para.approxInvPose, para.pointsMap, para.normalsMap, para.sceneIntrinsics, para.sceneImageSize, para.scenePose, para.viewIntrinsics, para.viewImageSize, para.distThresh);
}

// host methods

ITMDepthTracker_CUDA::ITMDepthTracker_CUDA(
    Vector2i depthImgSize,
    float distThresh,
    float terminationThreshold,
    const ITMLowLevelEngine *lowLevelEngine)
    :ITMDepthTracker(depthImgSize, distThresh, terminationThreshold, lowLevelEngine)
{
	ITMSafeCall(cudaMallocManaged((void**)&accu, sizeof(AccuCell)));
}

ITMDepthTracker_CUDA::~ITMDepthTracker_CUDA(void)
{
    ITMSafeCall(cudaFree(accu));
}

ITMDepthTracker::AccuCell ITMDepthTracker_CUDA::ComputeGandH(Matrix4f T_g_k_estimate) {
    Vector4f *pointsMap = this->pointsMap->GetData(MEMORYDEVICE_CUDA);
	Vector4f *normalsMap = this->normalsMap->GetData(MEMORYDEVICE_CUDA);
	Vector4f sceneIntrinsics = getViewIntrinsics();
    Vector2i sceneImageSize = this->pointsMap->noDims;
    assert(sceneImageSize == this->normalsMap->noDims);

    float *depth = currentTrackingLevel->depth->GetData(MEMORYDEVICE_CUDA);
    Vector4f viewIntrinsics = currentTrackingLevel->intrinsics;
    Vector2i viewImageSize = currentTrackingLevel->depth->noDims;

	dim3 blockSize(16, 16);
	dim3 gridSize(
        (int)ceil((float)viewImageSize.x / (float)blockSize.x), 
        (int)ceil((float)viewImageSize.y / (float)blockSize.y));

    ITMSafeCall(cudaMemset(accu, 0, sizeof(AccuCell)));

	struct ITMDepthTracker_KernelParameters args;
    args.accu = accu;
	args.depth = depth;
    args.approxInvPose = T_g_k_estimate;
	args.pointsMap = pointsMap;
	args.normalsMap = normalsMap;
	args.sceneIntrinsics = sceneIntrinsics;
	args.sceneImageSize = sceneImageSize;
	args.scenePose = scenePose;
	args.viewIntrinsics = viewIntrinsics;
	args.viewImageSize = viewImageSize;
    args.distThresh = currentTrackingLevel->distanceThreshold;

#define iteration(it) \
			it: depthTrackerOneLevel_g_rt_device<it> << <gridSize, blockSize >> >(args);

    switch (iterationType()) {
        case iteration(TRACKER_ITERATION_ROTATION);
        case iteration(TRACKER_ITERATION_TRANSLATION);
        case iteration(TRACKER_ITERATION_BOTH);
    }
#undef iteration

    cudaDeviceSynchronize(); // for later access of accu
    return *accu;
}


