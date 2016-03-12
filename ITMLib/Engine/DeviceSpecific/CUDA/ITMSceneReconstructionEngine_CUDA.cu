// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMSceneReconstructionEngine_CUDA.h"
#include "ITMCUDAUtils.h"
#include "../../DeviceAgnostic/ITMSceneReconstructionEngine.h"
#include "../../../Objects/ITMRenderState_VH.h"


using namespace ITMLib::Engine;

// device functions

template<class TVoxel, bool stopIntegratingAtMaxW, bool approximateIntegration>
__global__ void integrateIntoScene_device(TVoxel *voxelArray, const ITMPlainVoxelArray::ITMVoxelArrayInfo *arrayInfo,
    const Vector4u *rgb, Vector2i rgbImgSize, const float *depth, Vector2i depthImgSize, Matrix4f M_d, Matrix4f M_rgb, Vector4f projParams_d,
    Vector4f projParams_rgb, float _voxelSize, float mu, int maxW)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    Vector4f pt_model; int locId;

    locId = x + y * arrayInfo->size.x + z * arrayInfo->size.x * arrayInfo->size.y;

    if (stopIntegratingAtMaxW) if (voxelArray[locId].w_depth == maxW) return;
    //	if (approximateIntegration) if (voxelArray[locId].w_depth != 0) return;

    pt_model.x = (float)(x + arrayInfo->offset.x) * _voxelSize;
    pt_model.y = (float)(y + arrayInfo->offset.y) * _voxelSize;
    pt_model.z = (float)(z + arrayInfo->offset.z) * _voxelSize;
    pt_model.w = 1.0f;

    ComputeUpdatedVoxelInfo<TVoxel::hasColorInformation, TVoxel>::compute(voxelArray[locId], pt_model, M_d, projParams_d, M_rgb, projParams_rgb, mu, maxW, depth, depthImgSize, rgb, rgbImgSize);
}

template<class TVoxel, bool stopIntegratingAtMaxW, bool approximateIntegration>
__global__ void integrateIntoScene_device(TVoxel *localVBA, const ITMHashEntry *hashTable, int *visibleEntryIDs,
    const Vector4u *rgb, Vector2i rgbImgSize, const float *depth, Vector2i depthImgSize, Matrix4f M_d, Matrix4f M_rgb, Vector4f projParams_d,
    Vector4f projParams_rgb, float voxelSize, float mu, int maxW)
{
    // one thread block for each voxel block
    Vector3i globalPos;
    int entryId = visibleEntryIDs[blockIdx.x];

    const ITMHashEntry &currentHashEntry = hashTable[entryId];

    if (currentHashEntry.ptr < 0) return;

    globalPos = currentHashEntry.pos.toInt() * SDF_BLOCK_SIZE;

    TVoxel *localVoxelBlock = &(localVBA[currentHashEntry.ptr * SDF_BLOCK_SIZE3]);

    // one thread for each voxel
    int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;
    integrateVoxel(x, y, z,
        stopIntegratingAtMaxW, globalPos, localVoxelBlock, voxelSize,
        M_d, projParams_d, M_rgb, projParams_rgb, mu, maxW, depth, depthImgSize, rgb, rgbImgSize);
}

__global__ void buildHashAllocAndVisibleType_device(uchar *entriesAllocType, uchar *entriesVisibleType, Vector4s *blockCoords, const float *depth,
    Matrix4f invM_d, Vector4f projParams_d, float mu, Vector2i _imgSize, float _voxelSize, ITMHashEntry *hashTable, float viewFrustum_min,
    float viewFrustum_max)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > _imgSize.x - 1 || y > _imgSize.y - 1) return;

    buildHashAllocAndVisibleTypePP(entriesAllocType, entriesVisibleType, x, y, blockCoords, depth, invM_d,
        projParams_d, mu, _imgSize, _voxelSize, hashTable, viewFrustum_min, viewFrustum_max);
}

__global__ void setToType3(uchar *entriesVisibleType, int *visibleEntryIDs, int noVisibleEntries)
{
    int entryId = threadIdx.x + blockIdx.x * blockDim.x;
    if (entryId > noVisibleEntries - 1) return;
    entriesVisibleType[visibleEntryIDs[entryId]] = 3;
}

__global__ void allocateVoxelBlocksList_device(int *voxelAllocationList, int *excessAllocationList, ITMHashEntry *hashTable, int noTotalEntries,
    AllocationTempData *allocData, uchar *entriesAllocType, uchar *entriesVisibleType, Vector4s *blockCoords)
{
    int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (targetIdx > noTotalEntries - 1) return;

    allocateVoxelBlock(targetIdx,
        voxelAllocationList,
        excessAllocationList,
        hashTable,
        allocData,

        entriesAllocType,
        entriesVisibleType,
        blockCoords);
}

__global__ void reAllocateSwappedOutVoxelBlocks_device(int *voxelAllocationList, ITMHashEntry *hashTable, int noTotalEntries,
    AllocationTempData *allocData, /*int *noAllocatedVoxelEntries,*/ uchar *entriesVisibleType)
{
    int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (targetIdx > noTotalEntries - 1) return;

    reAllocateSwappedOutVoxelBlock(voxelAllocationList, targetIdx, entriesVisibleType, hashTable, allocData);
}

template<bool useSwapping>
__global__ void buildVisibleList_device(ITMHashEntry *hashTable, ITMHashSwapState *swapStates, int noTotalEntries,
    int *visibleEntryIDs, AllocationTempData *allocData, uchar *entriesVisibleType,
    Matrix4f M_d, Vector4f projParams_d, Vector2i depthImgSize, float voxelSize)
{
    int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (targetIdx > noTotalEntries - 1) return;

    __shared__ bool shouldPrefix;
    shouldPrefix = false;
    __syncthreads();

    bool visible = visibilityTestIfNeeded(
        targetIdx, entriesVisibleType, useSwapping, hashTable, swapStates,
        M_d, projParams_d, depthImgSize, voxelSize
        );

    if (visible) shouldPrefix = true;

    __syncthreads();

    if (shouldPrefix)
    {
        int offset = computePrefixSum_device<int>(visible, &allocData->noVisibleEntries, blockDim.x * blockDim.y, threadIdx.x);
        if (offset != -1) visibleEntryIDs[offset] = targetIdx;
    }
}

// host methods

template<class TVoxel>
ITMSceneReconstructionEngine_CUDA<TVoxel,ITMVoxelBlockHash>::ITMSceneReconstructionEngine_CUDA(void) 
{
	ITMSafeCall(cudaMalloc((void**)&allocationTempData_device, sizeof(AllocationTempData)));
	ITMSafeCall(cudaMallocHost((void**)&allocationTempData_host, sizeof(AllocationTempData)));

	int noTotalEntries = ITMVoxelBlockHash::noTotalEntries;
	ITMSafeCall(cudaMalloc((void**)&entriesAllocType_device, noTotalEntries));
	ITMSafeCall(cudaMalloc((void**)&blockCoords_device, noTotalEntries * sizeof(Vector4s)));
}

template<class TVoxel>
ITMSceneReconstructionEngine_CUDA<TVoxel,ITMVoxelBlockHash>::~ITMSceneReconstructionEngine_CUDA(void) 
{
	ITMSafeCall(cudaFreeHost(allocationTempData_host));
	ITMSafeCall(cudaFree(allocationTempData_device));
	ITMSafeCall(cudaFree(entriesAllocType_device));
	ITMSafeCall(cudaFree(blockCoords_device));
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel,ITMVoxelBlockHash>::ResetScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene)
{
	int numBlocks = scene->index.getNumAllocatedVoxelBlocks();
	int blockSize = scene->index.getVoxelBlockSize();

    // Reset all voxels in all voxel blocks
	TVoxel *voxelBlocks_ptr = scene->localVBA.GetVoxelBlocks();
	memsetKernel<TVoxel>(voxelBlocks_ptr, TVoxel(), numBlocks * blockSize);

    // Reset voxel allocation list
	int *vbaAllocationList_ptr = scene->localVBA.GetAllocationList();
	fillArrayKernel<int>(vbaAllocationList_ptr, numBlocks);
	scene->localVBA.lastFreeBlockId = numBlocks - 1;

    // Reset hash entries
    ITMHashEntry tmpEntry = ITMHashEntry::createIllegalEntry();
	ITMHashEntry *hashEntry_ptr = scene->index.GetEntries();
	memsetKernel<ITMHashEntry>(hashEntry_ptr, tmpEntry, scene->index.noTotalEntries);

    // Reset excess allocation list
	int *excessList_ptr = scene->index.GetExcessAllocationList();
	fillArrayKernel<int>(excessList_ptr, SDF_EXCESS_LIST_SIZE);
	scene->index.SetLastFreeExcessListId(SDF_EXCESS_LIST_SIZE - 1);
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel, ITMVoxelBlockHash>::AllocateSceneFromDepth(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view, 
	const ITMTrackingState *trackingState, const ITMRenderState *renderState, bool onlyUpdateVisibleList)
{
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, invM_d;
	Vector4f projParams_d, invProjParams_d;

	ITMRenderState_VH *renderState_vh = (ITMRenderState_VH*)renderState;

	M_d = trackingState->pose_d->GetM(); M_d.inv(invM_d);

	projParams_d = view->calib->intrinsics_d.projectionParamsSimple.all;
	invProjParams_d = projParams_d;
	invProjParams_d.x = 1.0f / invProjParams_d.x;
	invProjParams_d.y = 1.0f / invProjParams_d.y;

	float mu = scene->sceneParams->mu;

	float *depth = view->depth->GetData(MEMORYDEVICE_CUDA);
	int *voxelAllocationList = scene->localVBA.GetAllocationList();
	int *excessAllocationList = scene->index.GetExcessAllocationList();
	ITMHashEntry *hashTable = scene->index.GetEntries();


    // [swapping[
	ITMHashSwapState *swapStates = scene->useSwapping ? scene->globalCache->GetSwapStates(true) : 0;
    bool useSwapping = scene->useSwapping;
    // ]swapping]

	int noTotalEntries = scene->index.noTotalEntries;

	int *visibleEntryIDs = renderState_vh->GetVisibleEntryIDs();
	uchar *entriesVisibleType = renderState_vh->GetEntriesVisibleType();

	dim3 cudaBlockSizeHV(16, 16);
	dim3 gridSizeHV((int)ceil((float)depthImgSize.x / (float)cudaBlockSizeHV.x), (int)ceil((float)depthImgSize.y / (float)cudaBlockSizeHV.y));

	dim3 cudaBlockSizeAL(256, 1);
	dim3 gridSizeAL((int)ceil((float)noTotalEntries / (float)cudaBlockSizeAL.x));

	dim3 cudaBlockSizeVS(256, 1);
	dim3 gridSizeVS((int)ceil((float)renderState_vh->noVisibleEntries / (float)cudaBlockSizeVS.x));

	float oneOverVoxelSize = 1.0f / (voxelSize * SDF_BLOCK_SIZE);

	AllocationTempData *tempData = (AllocationTempData*)allocationTempData_host;
	tempData->noAllocatedVoxelEntries = scene->localVBA.lastFreeBlockId;
	tempData->noAllocatedExcessEntries = scene->index.GetLastFreeExcessListId();
	tempData->noVisibleEntries = 0;
	ITMSafeCall(cudaMemcpyAsync(allocationTempData_device, tempData, sizeof(AllocationTempData), cudaMemcpyHostToDevice));

	ITMSafeCall(cudaMemsetAsync(entriesAllocType_device, 0, sizeof(unsigned char)* noTotalEntries));

	if (gridSizeVS.x > 0) setToType3 << <gridSizeVS, cudaBlockSizeVS >> > (entriesVisibleType, visibleEntryIDs, renderState_vh->noVisibleEntries);

	buildHashAllocAndVisibleType_device << <gridSizeHV, cudaBlockSizeHV >> >(entriesAllocType_device, entriesVisibleType, 
		blockCoords_device, depth, invM_d, invProjParams_d, mu, depthImgSize, oneOverVoxelSize, hashTable,
		scene->sceneParams->viewFrustum_min, scene->sceneParams->viewFrustum_max);

	if (onlyUpdateVisibleList) useSwapping = false;
	if (!onlyUpdateVisibleList)
	{
		allocateVoxelBlocksList_device << <gridSizeAL, cudaBlockSizeAL >> >(voxelAllocationList, excessAllocationList, hashTable,
			noTotalEntries, (AllocationTempData*)allocationTempData_device, entriesAllocType_device, entriesVisibleType,
			blockCoords_device);
	}

	if (useSwapping)
		buildVisibleList_device<true> << <gridSizeAL, cudaBlockSizeAL >> >(hashTable, swapStates, noTotalEntries, visibleEntryIDs,
			(AllocationTempData*)allocationTempData_device, entriesVisibleType, M_d, projParams_d, depthImgSize, voxelSize);
	else
		buildVisibleList_device<false> << <gridSizeAL, cudaBlockSizeAL >> >(hashTable, swapStates, noTotalEntries, visibleEntryIDs,
			(AllocationTempData*)allocationTempData_device, entriesVisibleType, M_d, projParams_d, depthImgSize, voxelSize);

	if (useSwapping)
	{
		reAllocateSwappedOutVoxelBlocks_device << <gridSizeAL, cudaBlockSizeAL >> >(voxelAllocationList, hashTable, noTotalEntries, 
			(AllocationTempData*)allocationTempData_device, entriesVisibleType);
	}

	ITMSafeCall(cudaMemcpy(tempData, allocationTempData_device, sizeof(AllocationTempData), cudaMemcpyDeviceToHost));
	renderState_vh->noVisibleEntries = tempData->noVisibleEntries;
	scene->localVBA.lastFreeBlockId = tempData->noAllocatedVoxelEntries;
	scene->index.SetLastFreeExcessListId(tempData->noAllocatedExcessEntries);
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel, ITMVoxelBlockHash>::IntegrateIntoScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState)
{
	Vector2i rgbImgSize = view->rgb->noDims;
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, M_rgb;
	Vector4f projParams_d, projParams_rgb;

	ITMRenderState_VH *renderState_vh = (ITMRenderState_VH*)renderState;

	M_d = trackingState->pose_d->GetM();
	if (TVoxel::hasColorInformation) M_rgb = view->calib->trafo_rgb_to_depth.calib_inv * M_d;

	projParams_d = view->calib->intrinsics_d.projectionParamsSimple.all;
	projParams_rgb = view->calib->intrinsics_rgb.projectionParamsSimple.all;

	float mu = scene->sceneParams->mu; int maxW = scene->sceneParams->maxW;

	float *depth = view->depth->GetData(MEMORYDEVICE_CUDA);
	Vector4u *rgb = view->rgb->GetData(MEMORYDEVICE_CUDA);
	TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
	ITMHashEntry *hashTable = scene->index.GetEntries();

	int *visibleEntryIDs = renderState_vh->GetVisibleEntryIDs();

	dim3 cudaBlockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
	dim3 gridSize(renderState_vh->noVisibleEntries);

#define integrateIntoScene_d(stopIntegratingAtMaxW, approximateIntegration) \
    integrateIntoScene_device<TVoxel, stopIntegratingAtMaxW, approximateIntegration> << <gridSize, cudaBlockSize >> >(\
    localVBA, hashTable, visibleEntryIDs,\
        rgb, rgbImgSize, depth, depthImgSize, M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW)

	if (scene->sceneParams->stopIntegratingAtMaxW)
		if (trackingState->requiresFullRendering)
            integrateIntoScene_d(true, false);
		else
            integrateIntoScene_d(true, true);
	else
		if (trackingState->requiresFullRendering)
            integrateIntoScene_d(false, false);
		else
            integrateIntoScene_d(false, true);
}

// plain voxel array

template<class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel,ITMPlainVoxelArray>::ResetScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene)
{
	int numBlocks = scene->index.getNumAllocatedVoxelBlocks();
	int blockSize = scene->index.getVoxelBlockSize();

	TVoxel *voxelBlocks_ptr = scene->localVBA.GetVoxelBlocks();
	memsetKernel<TVoxel>(voxelBlocks_ptr, TVoxel(), numBlocks * blockSize);
	int *vbaAllocationList_ptr = scene->localVBA.GetAllocationList();
	fillArrayKernel<int>(vbaAllocationList_ptr, numBlocks);
	scene->localVBA.lastFreeBlockId = numBlocks - 1;
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel, ITMPlainVoxelArray>::AllocateSceneFromDepth(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState, bool onlyUpdateVisibleList)
{
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel, ITMPlainVoxelArray>::IntegrateIntoScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState)
{
	Vector2i rgbImgSize = view->rgb->noDims;
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, M_rgb;
	Vector4f projParams_d, projParams_rgb;

	M_d = trackingState->pose_d->GetM();
	if (TVoxel::hasColorInformation) M_rgb = view->calib->trafo_rgb_to_depth.calib_inv * M_d;

	projParams_d = view->calib->intrinsics_d.projectionParamsSimple.all;
	projParams_rgb = view->calib->intrinsics_rgb.projectionParamsSimple.all;

	float mu = scene->sceneParams->mu; int maxW = scene->sceneParams->maxW;

	float *depth = view->depth->GetData(MEMORYDEVICE_CUDA);
	Vector4u *rgb = view->rgb->GetData(MEMORYDEVICE_CUDA);
	TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
	const ITMPlainVoxelArray::ITMVoxelArrayInfo *arrayInfo = scene->index.getIndexData();

	dim3 cudaBlockSize(8, 8, 8);
	dim3 gridSize(scene->index.getVolumeSize().x / cudaBlockSize.x, scene->index.getVolumeSize().y / cudaBlockSize.y, scene->index.getVolumeSize().z / cudaBlockSize.z);

	if (scene->sceneParams->stopIntegratingAtMaxW) {
		if (trackingState->requiresFullRendering)
			integrateIntoScene_device < TVoxel, true, false> << <gridSize, cudaBlockSize >> >(localVBA, arrayInfo,
				rgb, rgbImgSize, depth, depthImgSize, M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
		else
			integrateIntoScene_device < TVoxel, true, true> << <gridSize, cudaBlockSize >> >(localVBA, arrayInfo,
				rgb, rgbImgSize, depth, depthImgSize, M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
	}
	else
	{
		if (trackingState->requiresFullRendering)
			integrateIntoScene_device < TVoxel, false, false> << <gridSize, cudaBlockSize >> >(localVBA, arrayInfo,
				rgb, rgbImgSize, depth, depthImgSize, M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
		else
			integrateIntoScene_device < TVoxel, false, true> << <gridSize, cudaBlockSize >> >(localVBA, arrayInfo,
				rgb, rgbImgSize, depth, depthImgSize, M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
	}
}



template class ITMLib::Engine::ITMSceneReconstructionEngine_CUDA<ITMVoxel, ITMVoxelIndex>;

