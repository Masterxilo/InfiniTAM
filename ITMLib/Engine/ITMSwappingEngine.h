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
			GPU ("local") memory to some other storage space ("global memory").
			*/
		template<class TVoxel, class TIndex>
		class ITMSwappingEngine
		{
		public:

			virtual void IntegrateGlobalIntoLocal(ITMScene<TVoxel, TIndex> *scene, ITMRenderState *renderState) = 0;

			virtual void SaveToGlobalMemory(ITMScene<TVoxel, TIndex> *scene, ITMRenderState *renderState) = 0;

			virtual ~ITMSwappingEngine(void) { }
		};
	}
}
