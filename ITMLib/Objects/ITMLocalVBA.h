// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMCUDAUtils.h"
#include "ITMLibDefines.h"
#include "MemoryBlock.h"

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
            public:
                /// This index is considered free in the list
                int lastFreeEntry;

                VoxelAllocationList() {
                    Reset();
                }

                /// Returns a free ptr in the local voxel block array
                /// to be used as the .ptr attribut
                // Assume single threaded
                // TODO return 0 when nothing can be allocated (full)
                __device__ int Allocate() {
                    // Must atomically decrease lastFreeEntry, but retrieve the value at the previous
                    int newlyReservedEntry = atomicSub(&lastFreeEntry, 1);
                    return newlyReservedEntry;
                }

                void Reset() {
                    cudaSafeCall(cudaDeviceSynchronize()); // make sure lastFreeEntry is accessible (this is a managed memory structure - the memory might be locked)
                    lastFreeEntry = SDF_LOCAL_BLOCK_NUM - 1;
                    cudaDeviceSynchronize(); // commit lastFreeEntry (TODO: needed?)
                }
            } *voxelAllocationList;


            ITMLocalVBA() 
			{
                voxelBlocks = new ORUtils::MemoryBlock<ITMVoxelBlock>(SDF_LOCAL_BLOCK_NUM, MEMORYDEVICE_CUDA);
                voxelAllocationList = new VoxelAllocationList();
			}

			~ITMLocalVBA(void)
			{
				delete voxelBlocks;
                delete voxelAllocationList;
			}
		};
	}
}
