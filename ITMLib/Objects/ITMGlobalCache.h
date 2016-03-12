// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include <stdlib.h>
#include <stdio.h>

#include "../Utils/ITMLibDefines.h"
#ifndef COMPILE_WITHOUT_CUDA
#include "../../ORUtils/CUDADefines.h"
#endif

namespace ITMLib
{
	namespace Objects
	{
        /**
        Storage for swapped-out voxel blocks in global memory.
        */
		template<class TVoxel>
		class ITMGlobalCache
		{
		private:
			bool *hasStoredData;
            /// Large buffer for SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE voxel blocks,
            /// enough to accommodate all blocks that could possibly exist
			TVoxel *storedVoxelBlocks;
            inline TVoxel *GetStoredVoxelBlockPtr(int address) { 
                return storedVoxelBlocks + address * SDF_BLOCK_SIZE3;
            }

			ITMHashSwapState *swapStates_host, *swapStates_device;

            /// Transfer buffer. At most SDF_TRANSFER_BLOCK_NUM entries.
            struct TransferBuffer {
                bool *hasSyncedData;
                TVoxel *syncedVoxelBlocks;
                int *neededEntryIDs;
            };
            TransferBuffer* transferBuffer_host;
            TransferBuffer* transferBuffer_device;

			bool *hasSyncedData_host, *hasSyncedData_device;
			TVoxel *syncedVoxelBlocks_host, *syncedVoxelBlocks_device;
            int *neededEntryIDs_host, *neededEntryIDs_device;
            
		public:
            /** Copy the voxel block (SDF_BLOCK_SIZE3 * TVoxel voxels) at 
            data to storedVoxelBlocks + address */
			inline void SetStoredData(int address, TVoxel *data) 
			{ 
				hasStoredData[address] = true; 
                memcpy(GetStoredVoxelBlockPtr(address), data, sizeof(TVoxel::VoxelBlock));
			}
			inline bool HasStoredData(int address) const { return hasStoredData[address]; }
            /// noTotalEntries many
            inline TVoxel *GetStoredVoxelBlock(int address) { return GetStoredVoxelBlockPtr(address); }
            /// noTotalEntries many
			ITMHashSwapState *GetSwapStates(bool useGPU) { return useGPU ? swapStates_device : swapStates_host; }
			
            /// Transfer buffer. At most SDF_TRANSFER_BLOCK_NUM entries.
			bool *GetHasSyncedData(bool useGPU) const { return useGPU ? hasSyncedData_device : hasSyncedData_host; }

            /// Transfer buffer. At most SDF_TRANSFER_BLOCK_NUM entries.
            TVoxel *GetSyncedVoxelBlocks(bool useGPU) const { return useGPU ? syncedVoxelBlocks_device : syncedVoxelBlocks_host; }

            /// Transfer buffer. At most SDF_TRANSFER_BLOCK_NUM entries.
            int *GetNeededEntryIDs(bool useGPU) { return useGPU ? neededEntryIDs_device : neededEntryIDs_host; }

			const int noTotalEntries; 

			ITMGlobalCache() : noTotalEntries(SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE)
			{	
				hasStoredData = (bool*)malloc(noTotalEntries * sizeof(bool));
                size_t storageBytes = noTotalEntries * sizeof(TVoxel) * SDF_BLOCK_SIZE3;
                storedVoxelBlocks = (TVoxel*)malloc(storageBytes);
                printf("ITMGlobalCache storedVoxelBlocks uses %d MB\n", storageBytes >> 20);
                printf("sizeof(TVoxel) = %d \n", sizeof(TVoxel));
                printf("sizeof(TVoxel::VoxelBlock) = %d \n", sizeof(TVoxel::VoxelBlock));
                
				memset(hasStoredData, 0, noTotalEntries);

				swapStates_host = (ITMHashSwapState *)malloc(noTotalEntries * sizeof(ITMHashSwapState));
				memset(swapStates_host, 0, sizeof(ITMHashSwapState) * noTotalEntries);

				syncedVoxelBlocks_host = (TVoxel *)malloc(SDF_TRANSFER_BLOCK_NUM * sizeof(TVoxel) * SDF_BLOCK_SIZE3);
				hasSyncedData_host = (bool*)malloc(SDF_TRANSFER_BLOCK_NUM * sizeof(bool));
				neededEntryIDs_host = (int*)malloc(SDF_TRANSFER_BLOCK_NUM * sizeof(int));
#ifndef COMPILE_WITHOUT_CUDA
				ITMSafeCall(cudaMalloc((void**)&swapStates_device, noTotalEntries * sizeof(ITMHashSwapState)));
				ITMSafeCall(cudaMemset(swapStates_device, 0, noTotalEntries * sizeof(ITMHashSwapState)));

				ITMSafeCall(cudaMalloc((void**)&syncedVoxelBlocks_device, SDF_TRANSFER_BLOCK_NUM * sizeof(TVoxel) * SDF_BLOCK_SIZE3));
				ITMSafeCall(cudaMalloc((void**)&hasSyncedData_device, SDF_TRANSFER_BLOCK_NUM * sizeof(bool)));

				ITMSafeCall(cudaMalloc((void**)&neededEntryIDs_device, SDF_TRANSFER_BLOCK_NUM * sizeof(int)));
#endif
			}

			void SaveToFile(char *fileName) const
			{
				TVoxel *storedData = storedVoxelBlocks;

				FILE *f = fopen(fileName, "wb");

				fwrite(hasStoredData, sizeof(bool), noTotalEntries, f);
				for (int i = 0; i < noTotalEntries; i++)
				{
					fwrite(storedData, sizeof(TVoxel) * SDF_BLOCK_SIZE3, 1, f);
					storedData += SDF_BLOCK_SIZE3;
				}

				fclose(f);
			}

			void ReadFromFile(char *fileName)
			{
				TVoxel *storedData = storedVoxelBlocks;
				FILE *f = fopen(fileName, "rb");

				size_t tmp = fread(hasStoredData, sizeof(bool), noTotalEntries, f);
				if (tmp == (size_t)noTotalEntries) {
					for (int i = 0; i < noTotalEntries; i++)
					{
						fread(storedData, sizeof(TVoxel) * SDF_BLOCK_SIZE3, 1, f);
						storedData += SDF_BLOCK_SIZE3;
					}
				}

				fclose(f);
			}

			~ITMGlobalCache(void) 
			{
				free(hasStoredData);
				free(storedVoxelBlocks);

				free(swapStates_host);

				free(hasSyncedData_host);
				free(syncedVoxelBlocks_host);
				free(neededEntryIDs_host);
#ifndef COMPILE_WITHOUT_CUDA
				ITMSafeCall(cudaFree(swapStates_device));
				ITMSafeCall(cudaFree(syncedVoxelBlocks_device));
				ITMSafeCall(cudaFree(hasSyncedData_device));
				ITMSafeCall(cudaFree(neededEntryIDs_device));
#endif
			}
		};
	}
}
