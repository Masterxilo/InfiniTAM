// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMDenseMapper.h"

#include "../Objects/ITMRenderState_VH.h"

#include "../ITMLib.h"

using namespace ITMLib::Engine;

template<class TVoxel, class TIndex>
ITMDenseMapper<TVoxel, TIndex>::ITMDenseMapper(const ITMLibSettings *settings)
{
	switch (settings->deviceType)
	{
    case ITMLibSettings::DEVICE_CPU:
#ifdef COMPILE_WITHOUT_CUDA
		scen#endifeRecoEngine = new ITMSceneReconstructionEngine_CPU<TVoxel,TIndex>();
#endif
        break;
	case ITMLibSettings::DEVICE_CUDA:
#ifndef COMPILE_WITHOUT_CUDA
		sceneRecoEngine = new ITMSceneReconstructionEngine_CUDA<TVoxel,TIndex>();
#endif
		break;
	}
}

template<class TVoxel, class TIndex>
ITMDenseMapper<TVoxel,TIndex>::~ITMDenseMapper()
{
	delete sceneRecoEngine;
}

template<class TVoxel, class TIndex>
void ITMDenseMapper<TVoxel,TIndex>::ResetScene(ITMScene<TVoxel,TIndex> *scene)
{
	sceneRecoEngine->ResetScene(scene);
}

/// Fusion stage of the system
template<class TVoxel, class TIndex>
void ITMDenseMapper<TVoxel,TIndex>::ProcessFrame(const ITMView *view, const ITMTrackingState *trackingState, ITMScene<TVoxel,TIndex> *scene, ITMRenderState *renderState)
{
	// allocation & visible list update
	sceneRecoEngine->AllocateSceneFromDepth(scene, view, trackingState, renderState);

	// camera data integration
	sceneRecoEngine->IntegrateIntoScene(scene, view, trackingState, renderState);
}

template<class TVoxel, class TIndex>
void ITMDenseMapper<TVoxel,TIndex>::UpdateVisibleList(const ITMView *view, const ITMTrackingState *trackingState, ITMScene<TVoxel,TIndex> *scene, ITMRenderState *renderState)
{
	sceneRecoEngine->AllocateSceneFromDepth(scene, view, trackingState, renderState, true);
}

template class ITMLib::Engine::ITMDenseMapper<ITMVoxel, ITMVoxelIndex>;
