// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMSwappingEngine_CUDA.h"
#include "ITMCUDAUtils.h"
#include "../../DeviceAgnostic/ITMSwappingEngine.h"
#include "../../../Objects/ITMRenderState_VH.h"

using namespace ITMLib::Engine;


__global__ void buildListToSwapIn_device(int *neededEntryIDs, int *noNeededEntries, ITMHashSwapState *swapStates, int noTotalEntries)
{
    int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (targetIdx > noTotalEntries - 1) return;

    __shared__ bool shouldPrefix;

    shouldPrefix = false;
    __syncthreads();

    bool isNeededId = (swapStates[targetIdx].state == HSS_HOST_AND_ACTIVE_NOT_COMBINED);

    if (isNeededId) shouldPrefix = true;
    __syncthreads();

    if (shouldPrefix)
    {
        int offset = computePrefixSum_device<int>(isNeededId, noNeededEntries, blockDim.x * blockDim.y, threadIdx.x);
        if (offset != -1 && offset < SDF_TRANSFER_BLOCK_NUM) neededEntryIDs[offset] = targetIdx;
    }
}

__global__ void buildListToSwapOut_device(int *neededEntryIDs, int *noNeededEntries, ITMHashSwapState *swapStates,
    ITMHashEntry *hashTable, uchar *entriesVisibleType, int noTotalEntries)
{
    int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (targetIdx > noTotalEntries - 1) return;

    __shared__ bool shouldPrefix;

    shouldPrefix = false;
    __syncthreads();

    ITMHashSwapState &swapState = swapStates[targetIdx];

    bool isNeededId = (swapState.state == HSS_ACTIVE &&
        hashTable[targetIdx].isAllocatedAndActive() && entriesVisibleType[targetIdx] == 0);

    if (isNeededId) shouldPrefix = true;
    __syncthreads();

    if (shouldPrefix)
    {
        int offset = computePrefixSum_device<int>(isNeededId, noNeededEntries, blockDim.x * blockDim.y, threadIdx.x);
        if (offset != -1 && offset < SDF_TRANSFER_BLOCK_NUM) neededEntryIDs[offset] = targetIdx;
    }
}

/// After blocks have been swapped out, mark them as such
template<class TVoxel>
__global__ void cleanMemory_device(int *voxelAllocationList, int *noAllocatedVoxelEntries, ITMHashSwapState *swapStates,
    ITMHashEntry *hashTable, TVoxel *localVBA, int *neededEntryIDs_local, int noNeededEntries)
{
    int locId = threadIdx.x + blockIdx.x * blockDim.x;

    if (locId > noNeededEntries - 1) return;

    int entryDestId = neededEntryIDs_local[locId];

    swapStates[entryDestId].state = 0;

    int vbaIdx = atomicAdd(&noAllocatedVoxelEntries[0], 1);
    if (vbaIdx < SDF_LOCAL_BLOCK_NUM - 1)
    {
        voxelAllocationList[vbaIdx + 1] = hashTable[entryDestId].ptr;
        hashTable[entryDestId].setSwappedOut();
    }
}

/// Integrate the data of all voxel blocks in the transfer buffer
/// into the live local voxel block array.
template<class TVoxel>
__global__ void integrateOldIntoActiveData_device(
    TVoxel *localVBA,
    ITMHashSwapState *swapStates,
    TVoxel *syncedVoxelBlocks_local,
    int *neededEntryIDs_local,
    ITMHashEntry *hashTable,
    int maxW)
{
    // One thread block per voxel block
    int entryDestId = neededEntryIDs_local[blockIdx.x];

    TVoxel *srcVB = syncedVoxelBlocks_local + blockIdx.x * SDF_BLOCK_SIZE3;
    TVoxel *dstVB = localVBA + hashTable[entryDestId].ptr * SDF_BLOCK_SIZE3;

    // One thread per voxel: integrate/combine the information
    int vIdx = threadIdx.x + threadIdx.y * SDF_BLOCK_SIZE + threadIdx.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

    CombineVoxelInformation<TVoxel::hasColorInformation, TVoxel>::compute(srcVB[vIdx], dstVB[vIdx], maxW);

    if (vIdx == 0) swapStates[entryDestId].state = HSS_ACTIVE;
}



/// Copy to transfer buffer the voxel block mentioned in neededEntryIDs_local[blockIdx.x]
template<class TVoxel>
__global__ void moveActiveDataToTransferBuffer_device(
    TVoxel *syncedVoxelBlocks_local,//!< [out] receives the voxel data
    bool *hasSyncedData_local, //!< [out] hasSyncedData_local[blockIdx.x] will be set to true
    const int * const neededEntryIDs_local, //!< [in] hash indices of entries that should be copied over
    ITMHashEntry * const hashTable, //!< [in] local voxel block array
    TVoxel * const localVBA //!< [in] local voxel block array
    )
{
    // One thread block per voxel block
    ITMHashEntry &hashEntry = hashTable[neededEntryIDs_local[blockIdx.x]];

    TVoxel *dstVB = syncedVoxelBlocks_local + blockIdx.x * SDF_BLOCK_SIZE3;
    TVoxel *srcVB = localVBA + hashEntry.ptr * SDF_BLOCK_SIZE3;

    // One thread per voxel: Copy it
    int vIdx = threadIdx.x + threadIdx.y * SDF_BLOCK_SIZE + threadIdx.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
    dstVB[vIdx] = srcVB[vIdx];
    srcVB[vIdx] = TVoxel();

    // only thread 0 sets this:
    if (vIdx == 0) hasSyncedData_local[blockIdx.x] = true;
}

template<class TVoxel>
ITMSwappingEngine_CUDA<TVoxel,ITMVoxelBlockHash>::ITMSwappingEngine_CUDA(void)
{
	ITMSafeCall(cudaMalloc((void**)&noAllocatedVoxelEntries_device, sizeof(int)));
	ITMSafeCall(cudaMalloc((void**)&noNeededEntries_device, sizeof(int)));
}

template<class TVoxel>
ITMSwappingEngine_CUDA<TVoxel,ITMVoxelBlockHash>::~ITMSwappingEngine_CUDA(void)
{
	ITMSafeCall(cudaFree(noAllocatedVoxelEntries_device));
	ITMSafeCall(cudaFree(noNeededEntries_device));
}

/// Receive requested ids in transfer buffer from GPU.
/// Send out the corresponding elements.
template<class TVoxel>
int ITMSwappingEngine_CUDA<TVoxel,ITMVoxelBlockHash>::LoadFromGlobalMemory(ITMScene<TVoxel,ITMVoxelBlockHash> *scene)
{
    // Renaming [[
	ITMGlobalCache<TVoxel> *globalCache = scene->globalCache;

	ITMHashSwapState *swapStates = globalCache->GetSwapStates(true);

    // _local means on gpu, _global is longterm (the swapped-out storage)
    TVoxel *syncedVoxelBlocks_local = globalCache->transferBuffer_device->syncedVoxelBlocks;
    bool *hasSyncedData_local = globalCache->transferBuffer_device->hasSyncedData;
    int *neededEntryIDs_local = globalCache->transferBuffer_device->neededEntryIDs;

    TVoxel *syncedVoxelBlocks_global = globalCache->transferBuffer_host->syncedVoxelBlocks;
    bool *hasSyncedData_global = globalCache->transferBuffer_host->hasSyncedData;
    int *neededEntryIDs_global = globalCache->transferBuffer_host->neededEntryIDs;
    // ]]

	dim3 blockSize(256);
	dim3 gridSize((int)ceil((float)scene->index.noTotalEntries / (float)blockSize.x));

    // Query the device, asking which blocks it needs.
    // "Build list of required voxel blocks"
	ITMSafeCall(cudaMemset(noNeededEntries_device, 0, sizeof(int)));

	buildListToSwapIn_device << <gridSize, blockSize >> >(neededEntryIDs_local, noNeededEntries_device, swapStates,
		scene->globalCache->noTotalEntries);

    // Query transfer buffer size and contents (neededEntryIDs_local)
    // "Copy list to long term storage host"
	int noNeededEntries;
	ITMSafeCall(cudaMemcpy(&noNeededEntries, noNeededEntries_device, sizeof(int), cudaMemcpyDeviceToHost));

    if (noNeededEntries <= 0) return noNeededEntries;
    noNeededEntries = MIN(noNeededEntries, SDF_TRANSFER_BLOCK_NUM);
    //  copy neededEntryIDs_local into neededEntryIDs_global here
	ITMSafeCall(cudaMemcpy(neededEntryIDs_global, neededEntryIDs_local, sizeof(int) * noNeededEntries, cudaMemcpyDeviceToHost));

    FillTransferBuffer(globalCache, noNeededEntries);

    // Send transfer buffer
    // "copy voxel transfer buffer to active memory"
	ITMSafeCall(cudaMemcpy(hasSyncedData_local, hasSyncedData_global, sizeof(bool) * noNeededEntries, cudaMemcpyHostToDevice));
	ITMSafeCall(cudaMemcpy(syncedVoxelBlocks_local, syncedVoxelBlocks_global, sizeof(TVoxel) *SDF_BLOCK_SIZE3 * noNeededEntries, cudaMemcpyHostToDevice));

	return noNeededEntries;
}

/// Swap in & Integrate.
/// Receive from global memory the requested blocks and integrate them
template<class TVoxel>
void ITMSwappingEngine_CUDA<TVoxel, ITMVoxelBlockHash>::IntegrateGlobalIntoLocal(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, ITMRenderState *renderState)
{
	ITMGlobalCache<TVoxel> *globalCache = scene->globalCache;

	ITMHashEntry *hashTable = scene->index.GetEntries();

	ITMHashSwapState *swapStates = globalCache->GetSwapStates(true);

    TVoxel *syncedVoxelBlocks_local = globalCache->transferBuffer_device->syncedVoxelBlocks;
    int *neededEntryIDs_local = globalCache->transferBuffer_device->neededEntryIDs;

	TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();


	int maxW = scene->sceneParams->maxW;

    // Swap-in: Let host upload transfer buffer, populated according to requested ids.
	int noNeededEntries = this->LoadFromGlobalMemory(scene);

    // "Integrate transferred data into active representation"
	if (noNeededEntries > 0) {
		dim3 blockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
		dim3 gridSize(noNeededEntries);

        printf("transferring %d voxel blocks from host to device\n", noNeededEntries);
		integrateOldIntoActiveData_device << <gridSize, blockSize >> >(localVBA, swapStates, syncedVoxelBlocks_local,
			neededEntryIDs_local, hashTable, maxW);
	}
}

/// Swap Out
template<class TVoxel>
void ITMSwappingEngine_CUDA<TVoxel, ITMVoxelBlockHash>::SaveToGlobalMemory(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, ITMRenderState *renderState)
{
	ITMGlobalCache<TVoxel> *globalCache = scene->globalCache;

	ITMHashSwapState *swapStates = globalCache->GetSwapStates(true);

	ITMHashEntry *hashTable = scene->index.GetEntries();
	uchar *entriesVisibleType = ((ITMRenderState_VH*)renderState)->GetEntriesVisibleType();
	
    TVoxel *syncedVoxelBlocks_local = globalCache->transferBuffer_device->syncedVoxelBlocks;
    bool *hasSyncedData_local = globalCache->transferBuffer_device->hasSyncedData;
    int *neededEntryIDs_local = globalCache->transferBuffer_device->neededEntryIDs;

    TVoxel *syncedVoxelBlocks_global = globalCache->transferBuffer_host->syncedVoxelBlocks; 
    bool *hasSyncedData_global = globalCache->transferBuffer_host->hasSyncedData;
    int *neededEntryIDs_global = globalCache->transferBuffer_host->neededEntryIDs;

	TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
	int *voxelAllocationList = scene->localVBA.GetAllocationList();

	int noTotalEntries = globalCache->noTotalEntries;
	
	dim3 blockSize, gridSize;
	int noNeededEntries;

    // "Build list of inactive voxel blocks"
	{
		blockSize = dim3(256);
		gridSize = dim3((int)ceil((float)scene->index.noTotalEntries / (float)blockSize.x));

		ITMSafeCall(cudaMemset(noNeededEntries_device, 0, sizeof(int)));

		buildListToSwapOut_device << <gridSize, blockSize >> >(neededEntryIDs_local, noNeededEntries_device, swapStates,
			hashTable, entriesVisibleType, noTotalEntries);

		ITMSafeCall(cudaMemcpy(&noNeededEntries, noNeededEntries_device, sizeof(int), cudaMemcpyDeviceToHost));
	}

    if (noNeededEntries <= 0) return;

    // "Populate transfer buffer according to list"
    printf("transferring %d voxel blocks from device to host\n", noNeededEntries);
	noNeededEntries = MIN(noNeededEntries, SDF_TRANSFER_BLOCK_NUM);
	{
		blockSize = dim3(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
		gridSize = dim3(noNeededEntries);

		moveActiveDataToTransferBuffer_device << <gridSize, blockSize >> >(syncedVoxelBlocks_local, hasSyncedData_local,
			neededEntryIDs_local, hashTable, localVBA);
	}

    // "Delete inactive voxel block from voxel block array"
	{
		blockSize = dim3(256);
		gridSize = dim3((int)ceil((float)noNeededEntries / (float)blockSize.x));

		ITMSafeCall(cudaMemcpy(noAllocatedVoxelEntries_device, &scene->localVBA.lastFreeBlockId, sizeof(int), cudaMemcpyHostToDevice));

		cleanMemory_device << <gridSize, blockSize >> >(voxelAllocationList, noAllocatedVoxelEntries_device, swapStates, hashTable, localVBA,
			neededEntryIDs_local, noNeededEntries);

		ITMSafeCall(cudaMemcpy(&scene->localVBA.lastFreeBlockId, noAllocatedVoxelEntries_device, sizeof(int), cudaMemcpyDeviceToHost));
		scene->localVBA.lastFreeBlockId = MAX(scene->localVBA.lastFreeBlockId, 0);
		scene->localVBA.lastFreeBlockId = MIN(scene->localVBA.lastFreeBlockId, SDF_LOCAL_BLOCK_NUM);
	}

    // "Copy transfer buffer to long term storage host"
	ITMSafeCall(cudaMemcpy(neededEntryIDs_global, neededEntryIDs_local, sizeof(int) * noNeededEntries, cudaMemcpyDeviceToHost));
	ITMSafeCall(cudaMemcpy(hasSyncedData_global, hasSyncedData_local, sizeof(bool) * noNeededEntries, cudaMemcpyDeviceToHost));
	ITMSafeCall(cudaMemcpy(syncedVoxelBlocks_global, syncedVoxelBlocks_local, sizeof(TVoxel) *SDF_BLOCK_SIZE3 * noNeededEntries, cudaMemcpyDeviceToHost));

    // "Write transferred blocks to long term storage"
	for (int entryId = 0; entryId < noNeededEntries; entryId++)
	{
		if (hasSyncedData_global[entryId])
			globalCache->SetStoredData(neededEntryIDs_global[entryId], syncedVoxelBlocks_global + entryId * SDF_BLOCK_SIZE3);
	}
}

template class ITMLib::Engine::ITMSwappingEngine_CUDA<ITMVoxel, ITMVoxelIndex>;
