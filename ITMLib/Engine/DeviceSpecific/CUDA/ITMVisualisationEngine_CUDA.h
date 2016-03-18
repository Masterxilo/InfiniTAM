// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../ITMVisualisationEngine.h"

struct RenderingBlock;

namespace ITMLib
{
	namespace Engine
	{
		class ITMVisualisationEngine_CUDA: public ITMVisualisationEngine
		{
		private:
			int *noVisibleEntries_device;

            RenderingBlock *renderingBlockList_device;
            uint *noTotalBlocks_device;
            /** Given scene, pose and intrinsics, create an estimate
            of the minimum and maximum depths at each pixel of
            an image.

            Called by rendering methods (CreateICPMaps, RenderImage).

            Creates the list of RenderingBlocks
            */
            void CreateExpectedDepths(
                const ITMPose *pose,
                const ITMIntrinsics *intrinsics,
                ITMRenderState *renderState //!< [out] initializes renderingRangeImage
                ) const;

		public:
			explicit ITMVisualisationEngine_CUDA(ITMScene *scene);
			~ITMVisualisationEngine_CUDA(void);

            void FindVisibleBlocks(
                const ITMPose *pose,
                const ITMIntrinsics *intrinsics,
                ITMRenderState *renderState //!< [out] initializes visibleEntryIDs(), noVisibleEntries, entriesVisibleType
                ) const;

            /// render for tracking
			void RenderImage(const ITMPose *pose, const ITMIntrinsics *intrinsics,
                ITMRenderState *renderState, //!< [in, out] uses visibility information, builds renderingRangeImage for one-time use
                ITMUChar4Image *outputImage, ITMVisualisationEngine::RenderImageType type = ITMVisualisationEngine::RENDER_SHADED_GREYSCALE) const;
			
            void CreateICPMaps(
                const ITMIntrinsics * intrinsics_d,
                ITMTrackingState *trackingState, // [in, out] builds trackingState->pointCloud, renders from trackingState->pose_d 
                ITMRenderState *renderState //!< [in, out] uses visibility information, builds renderingRangeImage for one-time use
                ) const;
			
			ITMRenderState* CreateRenderState(const Vector2i & imgSize) const;
		};
	}
}
