// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Utils/ITMLibDefines.h"

#include "../Objects/ITMScene.h"
#include "../Objects/ITMView.h"
#include "../Objects/ITMRenderState.h"

using namespace ITMLib::Objects;

namespace ITMLib
{
	namespace Engine
	{
		/** \brief
			Interface to engines that swap data in and out of the
			device/GPU ("local/active") memory to some other
            'long-term' storage space ("global memory") host.
			*/
		template<class TVoxel, class TIndex>
		class ITMSwappingEngine
		{
		public:
            /// Swap in & Integrate
            /// swapping: CPU -> GPU
			virtual void IntegrateGlobalIntoLocal(ITMScene<TVoxel, TIndex> *scene, ITMRenderState *renderState) = 0;

            /// Swap Out
            /// swapping: GPU -> CPU
			virtual void SaveToGlobalMemory(ITMScene<TVoxel, TIndex> *scene, ITMRenderState *renderState) = 0;

			virtual ~ITMSwappingEngine(void) { }

        protected:
            /// Fill transfer buffer
            /// According to the first noNeededEntries in neededEntryIDs_global
            /// "populate transfer buffer from long term storage"
            void FillTransferBuffer(ITMGlobalCache<TVoxel> *globalCache,
                int noNeededEntries) {
                int *neededEntryIDs_global = globalCache->transferBuffer_host->neededEntryIDs;
                globalCache->transferBuffer_host->clearHasSynchedDataAndBlocks_host(noNeededEntries);
                for (int i = 0; i < noNeededEntries; i++)
                {
                    int entryId = neededEntryIDs_global[i];

                    if (globalCache->HasStoredData(entryId))
                    {
                        globalCache->transferBuffer_host->setSyncedVoxelBlockFrom_host(
                            i,
                            globalCache->GetStoredVoxelBlock(entryId));
                    }
                }
            }
		};
	}
}
