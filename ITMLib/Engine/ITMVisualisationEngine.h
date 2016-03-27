// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMLibDefines.h"
#include "ITMView.h"
#include "ITMTrackingState.h"
#include "ITMRenderState.h"

using namespace ITMLib::Objects;

namespace ITMLib
{
	namespace Engine
    {
        /*!private!*/
        struct RenderingBlock {
            Vector2s upperLeft;
            Vector2s lowerRight;
            Vector2f zRange;
        };

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
        private:
            RenderingBlock *renderingBlockList_device;

            /** Given scene, pose and intrinsics, create an estimate
            of the minimum and maximum depths at each pixel of
            an image.

            Called by rendering methods (CreateICPMaps, RenderImage).
            */
            void CreateExpectedDepths();
            void Common(
                const ITMPose *pose,
                const ITMIntrinsics *intrinsics,
                ITMRenderState *renderState);

		public:
			enum RenderImageType
			{
				RENDER_SHADED_GREYSCALE,
				RENDER_COLOUR_FROM_VOLUME,
				RENDER_COLOUR_FROM_NORMAL
			};

            explicit ITMVisualisationEngine();
            virtual ~ITMVisualisationEngine(void);

            /// Heatmap style color gradient for depth
            static void DepthToUchar4(ITMUChar4Image *dst, const ITMFloatImage *src);

            /** Creates a render state, 
            */
            ITMRenderState* CreateRenderState(const Vector2i & imgSize) const;
            
			/** This will render an image using raycasting. */
            void RenderImage(
                const ITMPose *pose,
                const ITMIntrinsics *intrinsics,
                ITMRenderState *renderState, //!< [in, out] builds renderingRangeImage for one-time use
                ITMUChar4Image *outputImage,
                RenderImageType type = RENDER_SHADED_GREYSCALE);

			/** Create an image of reference points and normals as
			required by the ITMLib::Engine::ITMDepthTracker classes.
			*/
            void CreateICPMaps(
                ITMTrackingState * const trackingState, // [in, out] builds trackingState->pointCloud, renders from trackingState->pose_d 
                const ITMIntrinsics * const intrinsics_d,
                ITMRenderState *const renderState //!< [in, out] builds renderingRangeImage for one-time use
                );

		};
	}
}
