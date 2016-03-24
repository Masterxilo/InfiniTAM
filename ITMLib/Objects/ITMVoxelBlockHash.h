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
                // Cached key
                VoxelBlockPos blockPos;
                // Offset into the voxel block array, caches a .ptr of HashEntry
				int blockPtr;
				CPU_AND_GPU IndexCache(void) : blockPos(0x7fffffff), blockPtr(-1) {}
			};

			/** Maximum number of total entries. */
			static const CONSTPTR(int) voxelBlockSize = SDF_BLOCK_SIZE3;

		private:

			/** The actual data in the hash table. */
			ORUtils::MemoryBlock<ITMHashEntry> hashEntries;

		public:
            /// Allocation list generating sequential ids
            // Implemented as a countdown semaphore in CUDA unified memory
            // TODO we might as well count up.
            // TODO we dont currently check for underflow, clients are assumed to test returned values
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
                    cudaDeviceSynchronize(); // make sure lastFreeEntry is accessible (this is a managed memory structure - the memory might be pagelocked)
                    lastFreeEntry = capacity - 1;
                }
            } * excessAllocationList; // Must be allocated with new for Managed memory management to kick in.
            
			ITMVoxelBlockHash() :
                hashEntries(SDF_GLOBAL_BLOCK_NUM, MEMORYDEVICE_CUDA)
            {
                excessAllocationList = new ExcessAllocationList(SDF_EXCESS_LIST_SIZE);
            }

            ~ITMVoxelBlockHash() {
                delete excessAllocationList;
            }

			/** Get the list of actual entries in the hash table. */
            const ITMHashEntry *GetEntries(void) const { return hashEntries.GetData(MEMORYDEVICE_CUDA); }
            ITMHashEntry *GetEntries(void) { return hashEntries.GetData(MEMORYDEVICE_CUDA); }

		};
	}
}
