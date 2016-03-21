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
        protected:
        private:


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

            const ITMScene *scene;


		public:
			enum RenderImageType
			{
				RENDER_SHADED_GREYSCALE,
				RENDER_COLOUR_FROM_VOLUME,
				RENDER_COLOUR_FROM_NORMAL
			};

            explicit ITMVisualisationEngine(ITMScene *scene);
            virtual ~ITMVisualisationEngine(void);

            /// Heatmap style color gradient for depth
            static void DepthToUchar4(ITMUChar4Image *dst, const ITMFloatImage *src);

            /** Creates a render state, containing rendering info
            for the scene.
            */
            ITMRenderState* CreateRenderState(const Vector2i & imgSize) const;

			/** Given a scene, pose and intrinsics, compute the
			visible subset of the scene and store it in an
			appropriate visualisation state object, created
			previously using allocateInternalState().
			*/
            void FindVisibleBlocks(
                const ITMPose *pose, 
                const ITMIntrinsics *intrinsics,
                ITMRenderState *renderState //!< [out] initializes visibleEntryIDs(), noVisibleEntries, entriesVisibleType
                ) const;

			/** This will render an image using raycasting. */
            void RenderImage(const ITMPose *pose, const ITMIntrinsics *intrinsics,
                ITMRenderState *renderState, //!< [in, out] uses visibility information, builds renderingRangeImage for one-time use
                ITMUChar4Image *outputImage,
                RenderImageType type = RENDER_SHADED_GREYSCALE) const;

			/** Create an image of reference points and normals as
			required by the ITMLib::Engine::ITMDepthTracker classes.
			*/
            void CreateICPMaps(
                const ITMIntrinsics * intrinsics_d,
                ITMTrackingState *trackingState, // [in, out] builds trackingState->pointCloud, renders from trackingState->pose_d 
                ITMRenderState *renderState //!< [in, out] uses visibility information, builds renderingRangeImage for one-time use
                ) const;

		};
	}
}
