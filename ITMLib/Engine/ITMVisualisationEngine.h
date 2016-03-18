// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Utils/ITMLibDefines.h"

#include "../Objects/ITMScene.h"
#include "../Objects/ITMView.h"
#include "../Objects/ITMTrackingState.h"
#include "../Objects/ITMRenderState.h"

using namespace ITMLib::Objects;

namespace ITMLib
{
	namespace Engine
    {

        /** \brief
        Interface to engines helping with the visualisation of
        the results from the rest of the library.

        This is also used internally to get depth estimates for the
        raycasting done for the trackers. The basic idea there is
        to project down a scene of 8x8x8 voxel
        blocks and look at the bounding boxes. The projection
        provides an idea of the possible depth range for each pixel
        in an image, which can be used to speed up raycasting
        operations.
        */
        class ITMVisualisationEngine
        {
        protected:
            const ITMScene *scene;
            ITMVisualisationEngine(const ITMScene *scene)
            {
                this->scene = scene;
            }


		public:
			enum RenderImageType
			{
				RENDER_SHADED_GREYSCALE,
				RENDER_COLOUR_FROM_VOLUME,
				RENDER_COLOUR_FROM_NORMAL
			};

            virtual ~ITMVisualisationEngine(void) {}

            /// Heatmap style color gradient for depth
            static void DepthToUchar4(ITMUChar4Image *dst, const ITMFloatImage *src);

			/** Given a scene, pose and intrinsics, compute the
			visible subset of the scene and store it in an
			appropriate visualisation state object, created
			previously using allocateInternalState().
			*/
			virtual void FindVisibleBlocks(
                const ITMPose *pose, 
                const ITMIntrinsics *intrinsics,
                ITMRenderState *renderState //!< [out] initializes visibleEntryIDs(), noVisibleEntries, entriesVisibleType
                ) const = 0;

			/** This will render an image using raycasting. */
			virtual void RenderImage(
                const ITMPose *pose, 
                const ITMIntrinsics *intrinsics,
				ITMRenderState *renderState, 
                ITMUChar4Image *outputImage, 
                RenderImageType type = RENDER_SHADED_GREYSCALE) const = 0;

			/** Create an image of reference points and normals as
			required by the ITMLib::Engine::ITMDepthTracker classes.
			*/
			virtual void CreateICPMaps(const ITMView *view, ITMTrackingState *trackingState, 
				ITMRenderState *renderState) const = 0;

			/** Creates a render state, containing rendering info
			for the scene.
			*/
			virtual ITMRenderState* CreateRenderState(const Vector2i & imgSize) const = 0;
		};
	}
}
