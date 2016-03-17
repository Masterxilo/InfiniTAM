// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Utils/ITMLibDefines.h"

#include "ITMSceneParams.h"
#include "ITMLocalVBA.h"

namespace ITMLib
{
	namespace Objects
	{
		/** \brief
		Represents the 3D world model as a hash of small voxel
		blocks
		*/
		template<class TVoxel, //!< any of the ITMVoxel_ classes
        class TIndex //!< 
        >
		class ITMScene
		{
		public:

			/** Scene parameters like voxel size etc. */
			const ITMSceneParams *sceneParams;

			/** Hash table to reference the 8x8x8 blocks */
			TIndex index;

			/** Current local content of the 8x8x8 voxel blocks -- stored host or device */
			ITMLocalVBA<TVoxel> localVBA;

			ITMScene(const ITMSceneParams *sceneParams, MemoryDeviceType memoryType)
				: index(memoryType), localVBA(memoryType, index.getNumAllocatedVoxelBlocks(), index.getVoxelBlockSize())
			{
				this->sceneParams = sceneParams;
			}
		};
	}
}