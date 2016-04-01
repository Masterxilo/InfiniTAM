#pragma once

#include "ITMIntrinsics.h"
#include "ITMExtrinsics.h"
#include "ITMDisparityCalib.h"

/** \brief
	Represents the joint RGBD calibration parameters.
*/
class ITMRGBDCalib
{
public:
	/// Intrinsic parameters of the RGB camera.
	ITMIntrinsics intrinsics_rgb;

	/// Intrinsic parameters of the depth camera.
	ITMIntrinsics intrinsics_d;
			
	/** @brief
		Extrinsic calibration between RGB and depth
		cameras.

		This transformation takes points from the RGB
		camera coordinate system to the depth camera
		coordinate system.

        M_d = trafo_rgb_to_depth.calib * M_rgb

        M_rgb = trafo_rgb_to_depth.calib_inv * M_d

        where M_d transforms from world to depth camera coords,
        and M_rgb transforms from world to color camera coords.


	*/
	ITMExtrinsics trafo_rgb_to_depth;
			
	/// Calibration information to compute depth from disparity images.
	ITMDisparityCalib disparityCalib;
};
