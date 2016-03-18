// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../ITMSceneReconstructionEngine.h"

namespace ITMLib
{
	namespace Engine
    {
		class ITMSceneReconstructionEngine_CUDA : public ITMSceneReconstructionEngine
		{
		private:
			void *allocationTempData_device;
			void *allocationTempData_host;
			unsigned char *entriesAllocType_device;
			Vector4s *blockCoords_device;

		public:
			void ResetScene(ITMScene *scene);

			void AllocateSceneFromDepth(
                ITMScene *scene, 
                const ITMView *view, 
                const ITMTrackingState *trackingState,
				ITMRenderState *renderState);

			void IntegrateIntoScene(
                ITMScene *scene,
                const ITMView *view,
                const ITMTrackingState *trackingState,
				const ITMRenderState *renderState);

			ITMSceneReconstructionEngine_CUDA(void);
			~ITMSceneReconstructionEngine_CUDA(void);
		};
	}
}
