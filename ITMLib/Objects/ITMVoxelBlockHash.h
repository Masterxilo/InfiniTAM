// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#ifndef __METALC__
#include <stdlib.h>
#endif

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

			/** Identifies which entries of the overflow
			list are allocated. This is used if too
			many hash collisions caused the buckets to
			overflow.
			*/
			ORUtils::MemoryBlock<int> excessAllocationList;
			int lastFreeExcessListId;

			const MemoryDeviceType memoryType;

		public:
			ITMVoxelBlockHash(MemoryDeviceType memoryType) :
                memoryType(memoryType),
                hashEntries(noTotalEntries, memoryType),
                excessAllocationList(SDF_EXCESS_LIST_SIZE, memoryType)
			{
            }

			/** Get the list of actual entries in the hash table. */
			const ITMHashEntry *GetEntries(void) const { return hashEntries.GetData(memoryType); }
			ITMHashEntry *GetEntries(void) { return hashEntries.GetData(memoryType); }

			const IndexData *getIndexData(void) const { return hashEntries.GetData(memoryType); }
			IndexData *getIndexData(void) { return hashEntries.GetData(memoryType); }

			/** Get the list that identifies which entries of the
			overflow list are allocated. This is used if too
			many hash collisions caused the buckets to overflow.
			*/
			const int *GetExcessAllocationList(void) const { return excessAllocationList.GetData(memoryType); }
            int *GetExcessAllocationList(void) { return excessAllocationList.GetData(memoryType); }
			int GetLastFreeExcessListId(void) { return lastFreeExcessListId; }
			void SetLastFreeExcessListId(int lastFreeExcessListId) { this->lastFreeExcessListId = lastFreeExcessListId; }

			/** Maximum number of locally allocated entries. */
			int getNumAllocatedVoxelBlocks(void) { return SDF_LOCAL_BLOCK_NUM; }
            /// Amount of voxels per block
			int getVoxelBlockSize(void) { return SDF_BLOCK_SIZE3; }
		};
	}
}
