// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

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
		template<class TVoxel>
		class ITMLocalVBA
		{
		private:
			ORUtils::MemoryBlock<TVoxel> *voxelBlocks;

            const MemoryDeviceType memoryType;

		public:
			inline TVoxel *GetVoxelBlocks(void) { return voxelBlocks->GetData(memoryType); }
			inline const TVoxel *GetVoxelBlocks(void) const { return voxelBlocks->GetData(memoryType); }

			const int allocatedSize;

            /// Allocation list generating sequential ids
            // Implemented as a countdown semaphore in CUDA unified memory
            // Filled from the top
            class VoxelAllocationList : Managed {
            private:
                /// This index is considered free in the list
                int lastFreeEntry;
                const int capacity;
                ORUtils::MemoryBlock<int> _allocationList;
                int* allocationList;

            public:
                VoxelAllocationList(int capacity, MemoryDeviceType memoryType) : 
                    capacity(capacity), _allocationList(capacity, memoryType) {
                    allocationList = _allocationList.GetData(memoryType);
                    Reset();
                }

#if !defined(COMPILE_WITHOUT_CUDA) && defined(__CUDACC__)
                __device__ int Allocate() {
                    return atomicSub(&lastFreeEntry, 1);
                }

                __device__ void Free(int ptr) {
                    return atomicSub(&lastFreeEntry, 1);
                }
#else
                /// Returns a free ptr in the local voxel block array
                // Assume single threaded
                int Allocate() {
                    int ptr = allocationList[lastFreeEntry];
                    allocationList[lastFreeEntry] = -1; // now illegal - updating this is not strictly necessary
                    lastFreeEntry--;
                    return ptr;
                }

                void Free(int ptr) {
                    // TODO dont accept when list is full already or when this was never allocated
                    printf("voxel block %d freed\n", ptr);
                    lastFreeEntry++;
                    allocationList[lastFreeEntry] = ptr;
                }
#endif
                void Reset() {
                    lastFreeEntry = capacity - 1;
#if !defined(COMPILE_WITHOUT_CUDA) && defined(__CUDACC__)
                    fillArrayKernel<int>(allocationList, capacity);
#else
                    for (int i = 0; i < capacity; ++i) allocationList[i] = i;
#endif
                }
            } *voxelAllocationList;


            ITMLocalVBA(MemoryDeviceType memoryType, int noBlocks, int blockSize) :
                memoryType(memoryType),
                allocatedSize(noBlocks * blockSize)
			{
				voxelBlocks = new ORUtils::MemoryBlock<TVoxel>(allocatedSize, memoryType);
                voxelAllocationList = new VoxelAllocationList(noBlocks, memoryType);
			}

			~ITMLocalVBA(void)
			{
				delete voxelBlocks;
                delete voxelAllocationList;
			}

			// Suppress the default copy constructor and assignment operator
			ITMLocalVBA(const ITMLocalVBA&);
			ITMLocalVBA& operator=(const ITMLocalVBA&);
		};
	}
}
