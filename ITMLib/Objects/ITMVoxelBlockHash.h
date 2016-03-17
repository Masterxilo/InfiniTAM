// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once


#include "..\Engine\DeviceSpecific\CUDA\ITMCUDAUtils.h"

#include "../Utils/ITMLibDefines.h"

#include "../../ORUtils/MemoryBlock.h"

namespace ITMLib
{
	namespace Objects
	{
		/** \brief
		This is the central class for the voxel block hash
		implementation. 

        Accessed via scene->index.
		*/
		class ITMVoxelBlockHash
		{
		public:
			typedef ITMHashEntry IndexData;

            /// Caches a single voxel block.
			struct IndexCache {
				Vector3i blockPos;
				int blockPtr;
				_CPU_AND_GPU_CODE_ IndexCache(void) : blockPos(0x7fffffff), blockPtr(-1) {}
			};

			/** Maximum number of total entries. */
			static const CONSTPTR(int) noTotalEntries = SDF_GLOBAL_BLOCK_NUM;
			static const CONSTPTR(int) voxelBlockSize = SDF_BLOCK_SIZE3;

		private:

			/** The actual data in the hash table. */
			ORUtils::MemoryBlock<ITMHashEntry> hashEntries;


			const MemoryDeviceType memoryType;

		public:
            /// Allocation list generating sequential ids
            // Implemented as a countdown semaphore in CUDA unified memory
            class ExcessAllocationList : public Managed
            {
            private:
                /// This index is considered free 
                /// Return it and change it when allocation is requested.
                int lastFreeEntry;
                const int capacity;
            public:
                ExcessAllocationList(int capacity) : capacity(capacity) {
                    Reset();
                }

                __device__ int Allocate() {
                    return atomicSub(&lastFreeEntry, 1);
                }

                void Reset() {
                    cudaDeviceSynchronize(); // make sure lastFreeEntry is accessible (this is a managed memory structure - the memory might be locked)
                    lastFreeEntry = capacity - 1;
                }
            } * excessAllocationList;
            
			ITMVoxelBlockHash(MemoryDeviceType memoryType) :
                memoryType(memoryType),
                hashEntries(noTotalEntries, memoryType)
            {
                excessAllocationList = new ExcessAllocationList(SDF_EXCESS_LIST_SIZE);
            }

            ~ITMVoxelBlockHash() {
                delete excessAllocationList;
            }

			/** Get the list of actual entries in the hash table. */
			const ITMHashEntry *GetEntries(void) const { return hashEntries.GetData(memoryType); }
			ITMHashEntry *GetEntries(void) { return hashEntries.GetData(memoryType); }

			const IndexData *getIndexData(void) const { return hashEntries.GetData(memoryType); }
			IndexData *getIndexData(void) { return hashEntries.GetData(memoryType); }

			
			/** Maximum number of locally allocated entries. */
			int getNumAllocatedVoxelBlocks(void) { return SDF_LOCAL_BLOCK_NUM; }
            /// Amount of voxels per block
			int getVoxelBlockSize(void) { return SDF_BLOCK_SIZE3; }
		};
	}
}
