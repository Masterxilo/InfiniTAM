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
	ITMView *view;

public:
    Scene* scene;
	/// Gives access to the current input frame
	ITMView* GetView() { return view; }

	/// Process a frame with rgb and depth images and optionally a corresponding imu measurement.
    /// Key method.
	void ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage);

	void GetImage(
        ITMUChar4Image * const out, //!< [in] must be allocated on cuda and host. On exit, host version will be requested image, cuda image undefined. Dimensions must not change from call to call.

        ITMFloatImage * const outDepth,
        const ITMPose * const pose, //!< used for InfiniTAM_IMAGE_FREECAMERA_... image type
        const ITMIntrinsics * const intrinsics,
        std::string shader 
        );

	explicit ITMMainEngine(
        const ITMRGBDCalib *calib
        );
	~ITMMainEngine();
};

