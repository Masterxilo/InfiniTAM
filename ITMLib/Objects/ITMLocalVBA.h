// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "..\Engine\DeviceSpecific\CUDA\ITMCUDAUtils.h"
#include <stdlib.h>

#include "../Utils/ITMLibDefines.h"
#include "../../ORUtils/MemoryBlock.h"

namespace ITMLib
{
	namespace Objects
	{
		/** \brief
		Stores the actual voxel content that is referred to by a
		ITMLib::Objects::ITMHashTable.
		*/
		class ITMLocalVBA
		{
		private:
            ORUtils::MemoryBlock<ITMVoxelBlock> *voxelBlocks;

		public:
            inline ITMVoxelBlock *GetVoxelBlocks(void) { return voxelBlocks->GetData(MEMORYDEVICE_CUDA); }
            inline const ITMVoxelBlock *GetVoxelBlocks(void) const { return voxelBlocks->GetData(MEMORYDEVICE_CUDA); }


            /// Allocation list generating sequential ids
            // Implemented as a countdown semaphore in CUDA unified memory
            // Filled from the top
            class VoxelAllocationList : public Managed 
            {
            private:
                /// This index is considered free in the list
                int lastFreeEntry;
                const int capacity;
                ORUtils::MemoryBlock<int> _allocationList;
                /// Lists numbers that will eventually be assigned, becoming indices into the local voxel block array
                int* allocationList;

            public:
                VoxelAllocationList(int capacity) : 
                    capacity(capacity), _allocationList(capacity, MEMORYDEVICE_CUDA) {
                    allocationList = _allocationList.GetData(MEMORYDEVICE_CUDA);
                    Reset();
                }

                /// Returns a free ptr in the local voxel block array
                /// to be used as the .ptr attribut
                // Assume single threaded
                // TODO return 0 when nothing can be allocated (full)
                __device__ int Allocate() {
                    int ptr;

                    // Must atomically decrease lastFreeEntry, but retrieve the value at the previous
                    int newlyReservedEntry = atomicSub(&lastFreeEntry, 1);

                    ptr = allocationList[newlyReservedEntry];
                    allocationList[newlyReservedEntry] = -1; // now illegal - updating this is not strictly necessary

                    return ptr;
                }

                __device__ void Free(int ptr) {
                    // TODO dont accept when list is full already or when this was never allocated
                    // that is, when lastFreeEntry >= SDF_LOCAL_BLOCK_NUM - 1
                    // ^^ should never happen

                    int newlyFreedEntry = atomicAdd(&lastFreeEntry, 1) + 1; // atomicAdd returns the last value, but the new value
                    // is what is newly free. We should not read the changed lastFreeEntry as it might have changed yet again!

                    allocationList[newlyFreedEntry] = ptr;
                }

                void Reset() {
                    cudaDeviceSynchronize(); // make sure lastFreeEntry is accessible (this is a managed memory structure - the memory might be locked)
                    lastFreeEntry = capacity - 1;

                    // allocationList initially contains [0:capacity-1]
                    fillArrayKernel<int>(allocationList, capacity);
                }
            } *voxelAllocationList;


            ITMLocalVBA(int noBlocks) 
			{
                voxelBlocks = new ORUtils::MemoryBlock<ITMVoxelBlock>(noBlocks, MEMORYDEVICE_CUDA);
                voxelAllocationList = new VoxelAllocationList(noBlocks);
			}

			~ITMLocalVBA(void)
			{
				delete voxelBlocks;
                delete voxelAllocationList;
			}
		};
	}
}
