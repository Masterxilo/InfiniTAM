// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../ITMVisualisationEngine.h"

struct RenderingBlock;

namespace ITMLib
{
	namespace Engine
	{

        template<class TVoxel, class TIndex>
        class ITMVisualisationEngine_CUDA : public ITMVisualisationEngine < TVoxel, TIndex >
        {};

		template<class TVoxel>
		class ITMVisualisationEngine_CUDA<TVoxel, ITMVoxelBlockHash> : public ITMVisualisationEngine < TVoxel, ITMVoxelBlockHash >
		{
		private:
			RenderingBlock *renderingBlockList_device;
			uint *noTotalBlocks_device;
			int *noVisibleEntries_device;
		public:
			explicit ITMVisualisationEngine_CUDA(ITMScene<TVoxel, ITMVoxelBlockHash> *scene);
			~ITMVisualisationEngine_CUDA(void);

            void FindVisibleBlocks(
                const ITMPose *pose,
                const ITMIntrinsics *intrinsics,
                ITMRenderState *renderState //!< [out] initializes visibleEntryIDs(), noVisibleEntries, entriesVisibleType
                ) const;

            /// render for tracking
            void CreateExpectedDepths(const ITMPose *pose, const ITMIntrinsics *intrinsics, ITMRenderState *renderState) const;
			void RenderImage(const ITMPose *pose, const ITMIntrinsics *intrinsics, const ITMRenderState *renderState, 
				ITMUChar4Image *outputImage, IITMVisualisationEngine::RenderImageType type = IITMVisualisationEngine::RENDER_SHADED_GREYSCALE) const;
			

            void FindSurface(
                const ITMPose *pose,
                const ITMIntrinsics *intrinsics,
                ITMRenderState *renderState //!< [out] initializes raycastResult
                ) const; 

            void CreateICPMaps(const ITMView *view, ITMTrackingState *trackingState, ITMRenderState *renderState) const;
			
			ITMRenderState* CreateRenderState(const Vector2i & imgSize) const;
		};
	}
}
