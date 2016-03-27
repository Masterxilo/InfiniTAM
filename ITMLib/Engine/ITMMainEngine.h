// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMLib.h"
#include "ITMLibSettings.h"
#include "ITMLowLevelEngine.h"
#include "Scene.h"

/** \mainpage
    This is the API reference documentation for InfiniTAM. For a general
    overview additional documentation can be found in the included Technical
    Report.

    For use of ITMLib in your own project, the class
    @ref ITMLib::Engine::ITMMainEngine should be the main interface and entry
    point to the library.
*/

namespace ITMLib
{
	namespace Engine
	{
		/** \brief
		    Main engine, that instantiates all the other engines and
		    provides a simplified interface to them.

		    This class is the main entry point to the ITMLib library
		    and basically performs the whole KinectFusion algorithm.
		    It stores the latest image internally, as well as the 3D
		    world model and additionally it keeps track of the camera
		    pose.

		    The intended use is as follows:
		    -# Create an ITMMainEngine specifying the internal settings,
		       camera parameters and image sizes
		    -# Get the pointer to the internally stored images with
		       @ref GetView() and write new image information to that
		       memory
		    -# Call the method @ref ProcessFrame() to track the camera
		       and integrate the new information into the world model
		    -# Optionally access the rendered reconstruction or another
		       image for visualisation using @ref GetImage()
		    -# Iterate the above three steps for each image in the
		       sequence

		    To access the internal information, look at the member
		    variables @ref trackingState and @ref scene.
		*/
		class ITMMainEngine
		{
		private:
			const ITMLibSettings *settings;

			bool fusionActive, mainProcessingActive;

			ITMLowLevelEngine *lowLevelEngine;
			ITMVisualisationEngine *visualisationEngine;

            ITMViewBuilder *viewBuilder;
            ITMSceneReconstructionEngine *sceneRecoEngine;

			ITMTracker *tracker;

			ITMView *view;
			ITMTrackingState *trackingState;

			ITMScene* scene;
            Scene* sscene;
            /// Describes the tracked camera position and possibly a frame rendered from that position
			ITMRenderState *renderState_live;

            /// Describes a free camera position and frame rendered from that position
			ITMRenderState *renderState_freeview;

		public:

			enum GetImageType
			{
				InfiniTAM_IMAGE_ORIGINAL_RGB,
				InfiniTAM_IMAGE_ORIGINAL_DEPTH,

				InfiniTAM_IMAGE_FREECAMERA_SHADED,
				InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME,
				InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL,

				InfiniTAM_IMAGE_UNKNOWN
			};

			/// Gives access to the current input frame
			ITMView* GetView() { return view; }

			/// Gives access to the current camera pose and additional tracking information
			ITMTrackingState* GetTrackingState(void) { return trackingState; }

			/// Gives access to the internal world representation
			ITMScene* GetScene(void) { return scene; }

            /// Gives access to the internal world representation
            void ResetScene(void) { sceneRecoEngine->ResetScene(scene); }

			/// Process a frame with rgb and depth images and optionally a corresponding imu measurement.
            /// Key method.
			void ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage);

			void GetImage(
                ITMUChar4Image * const out, //!< [in] must be allocated on cuda and host. On exit, host version will be requested image, cuda image undefined. Dimensions must not change from call to call.
                const GetImageType getImageType, 
                const ITMPose * const pose = NULL, //!< used for InfiniTAM_IMAGE_FREECAMERA_... image type
                const ITMIntrinsics * const intrinsics = NULL //!< used for InfiniTAM_IMAGE_FREECAMERA_... image type
                );

			explicit ITMMainEngine(
                const ITMLibSettings *settings,
                const ITMRGBDCalib *calib, 
                Vector2i imgSize_rgb, 
                Vector2i imgSize_d
                );
			~ITMMainEngine();
		};
	}
}

