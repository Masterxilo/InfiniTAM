// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMLibDefines.h"

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
		class ITMScene
		{
		public:

			/** Scene parameters like voxel size etc. */
			const ITMSceneParams *sceneParams;

			/** Hash table to reference the 8x8x8 blocks */
			ITMVoxelBlockHash index;

			/** Current local content of the 8x8x8 voxel blocks */
			ITMLocalVBA localVBA;

			ITMScene(const ITMSceneParams *sceneParams) 
			{
				this->sceneParams = sceneParams;
			}
		};
	}
}