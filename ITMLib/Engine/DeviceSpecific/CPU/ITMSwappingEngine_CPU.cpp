// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMSwappingEngine_CPU.h"
#include "../../DeviceAgnostic/ITMSwappingEngine.h"
#include "../../../Objects/ITMRenderState_VH.h"

using namespace ITMLib::Engine;

template<class TVoxel>
int ITMSwappingEngine_CPU<TVoxel, ITMVoxelBlockHash>::LoadFromGlobalMemory(ITMScene<TVoxel, ITMVoxelBlockHash> *scene)
{
	ITMGlobalCache<TVoxel> *globalCache = scene->globalCache;

	ITMHashSwapState *swapStates = globalCache->GetSwapStates(false);

    TVoxel *syncedVoxelBlocks_local = globalCache->transferBuffer_host->syncedVoxelBlocks;
    bool *hasSyncedData_local = globalCache->transferBuffer_host->hasSyncedData;
    int *neededEntryIDs_local = globalCache->transferBuffer_host->neededEntryIDs;

    TVoxel *syncedVoxelBlocks_global = globalCache->transferBuffer_host->syncedVoxelBlocks;
    bool *hasSyncedData_global = globalCache->transferBuffer_host->hasSyncedData;
    int *neededEntryIDs_global = globalCache->transferBuffer_host->neededEntryIDs;

	int noTotalEntries = globalCache->noTotalEntries;

	int noNeededEntries = 0;
	for (int entryId = 0; entryId < noTotalEntries; entryId++)
	{
		if (noNeededEntries >= SDF_TRANSFER_BLOCK_NUM) break;
        if (swapStates[entryId].state == HSS_HOST_AND_ACTIVE_NOT_COMBINED)
		{
			neededEntryIDs_local[noNeededEntries] = entryId;
			noNeededEntries++;
		}
	}

	// would copy neededEntryIDs_local into neededEntryIDs_global here

    FillTransferBuffer(globalCache, noNeededEntries);

	// would copy syncedVoxelBlocks_global and hasSyncedData_global 
    // to syncedVoxelBlocks_local and hasSyncedData_local here

	return noNeededEntries;
}

template<class TVoxel>
void ITMSwappingEngine_CPU<TVoxel, ITMVoxelBlockHash>::IntegrateGlobalIntoLocal(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, ITMRenderState *renderState)
{
	ITMGlobalCache<TVoxel> *globalCache = scene->globalCache;

	ITMHashEntry *hashTable = scene->index.GetEntries();

	ITMHashSwapState *swapStates = globalCache->GetSwapStates(false);

    TVoxel *syncedVoxelBlocks_local = globalCache->transferBuffer_host->syncedVoxelBlocks;
    bool *hasSyncedData_local = globalCache->transferBuffer_host->hasSyncedData;
    int *neededEntryIDs_local = globalCache->transferBuffer_host->neededEntryIDs;

	TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();

	int noNeededEntries = this->LoadFromGlobalMemory(scene);

	int maxW = scene->sceneParams->maxW;

    // for each voxel block
	for (int i = 0; i < noNeededEntries; i++)
	{
		int entryDestId = neededEntryIDs_local[i];

		if (hasSyncedData_local[i])
		{
			TVoxel *srcVB = syncedVoxelBlocks_local + i * SDF_BLOCK_SIZE3;
			TVoxel *dstVB = localVBA + hashTable[entryDestId].ptr * SDF_BLOCK_SIZE3;

            // for each voxel: integrate the information
			for (int vIdx = 0; vIdx < SDF_BLOCK_SIZE3; vIdx++)
			{
				CombineVoxelInformation<TVoxel::hasColorInformation, TVoxel>::compute(srcVB[vIdx], dstVB[vIdx], maxW);
			}
		}

        swapStates[entryDestId].state = HSS_ACTIVE;
	}
}

template<class TVoxel>
void ITMSwappingEngine_CPU<TVoxel, ITMVoxelBlockHash>::SaveToGlobalMemory(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, ITMRenderState *renderState)
{
	ITMGlobalCache<TVoxel> *globalCache = scene->globalCache;

	ITMHashSwapState *swapStates = globalCache->GetSwapStates(false);

	ITMHashEntry *hashTable = scene->index.GetEntries();
	uchar *entriesVisibleType = ((ITMRenderState_VH*)renderState)->GetEntriesVisibleType();

    TVoxel *syncedVoxelBlocks_local = globalCache->transferBuffer_host->syncedVoxelBlocks;
    bool *hasSyncedData_local       = globalCache->transferBuffer_host->hasSyncedData;
    int *neededEntryIDs_local       = globalCache->transferBuffer_host->neededEntryIDs;

    TVoxel *syncedVoxelBlocks_global = globalCache->transferBuffer_host->syncedVoxelBlocks;
    bool *hasSyncedData_global       = globalCache->transferBuffer_host->hasSyncedData;
    int *neededEntryIDs_global       = globalCache->transferBuffer_host->neededEntryIDs;

	TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
	int *voxelAllocationList = scene->localVBA.GetAllocationList();

	int noTotalEntries = globalCache->noTotalEntries;
	
	int noNeededEntries = 0;
	int noAllocatedVoxelEntries = scene->localVBA.lastFreeBlockId;

    // Check all allocated, invisible entries in the hash table (voxel blocks)
    // and copy them to the transfer buffer
	for (int entryDestId = 0; entryDestId < noTotalEntries; entryDestId++)
	{
		if (noNeededEntries >= SDF_TRANSFER_BLOCK_NUM) break;

		int localPtr = hashTable[entryDestId].ptr;
		ITMHashSwapState &swapState = swapStates[entryDestId];

		if (swapState.state == HSS_ACTIVE && localPtr >= 0 && entriesVisibleType[entryDestId] == 0)
		{
			TVoxel *localVBALocation = localVBA + localPtr * SDF_BLOCK_SIZE3;

            // Copy to transfer buffer
			neededEntryIDs_local[noNeededEntries] = entryDestId;
			hasSyncedData_local[noNeededEntries] = true;
			memcpy(syncedVoxelBlocks_local + noNeededEntries * SDF_BLOCK_SIZE3, localVBALocation, SDF_BLOCK_SIZE3 * sizeof(TVoxel));
            noNeededEntries++;

			swapStates[entryDestId].state = 0;

            // Free the corresponding local vba entry as allocatable 
            // and mark the hashTable entry as swapped out
			int vbaIdx = noAllocatedVoxelEntries;
			if (vbaIdx < SDF_BUCKET_NUM - 1)
			{
				noAllocatedVoxelEntries++;
				voxelAllocationList[vbaIdx + 1] = localPtr;
                hashTable[entryDestId].setSwappedOut();

				for (int i = 0; i < SDF_BLOCK_SIZE3; i++) localVBALocation[i] = TVoxel();
			}

		}
	}

	scene->localVBA.lastFreeBlockId = noAllocatedVoxelEntries;

	// would copy neededEntryIDs_local, hasSyncedData_local and syncedVoxelBlocks_local into *_global here

	if (noNeededEntries > 0)
	{
		for (int entryId = 0; entryId < noNeededEntries; entryId++)
		{
			if (hasSyncedData_global[entryId])
				globalCache->SetStoredData(neededEntryIDs_global[entryId], syncedVoxelBlocks_global + entryId * SDF_BLOCK_SIZE3);
		}
	}
}

template class ITMLib::Engine::ITMSwappingEngine_CPU<ITMVoxel, ITMVoxelIndex>;
