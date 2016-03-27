// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once


#include "ITMLibDefines.h"
#define ITMFloatImage ORUtils::Image<float>
#define ITMFloat2Image ORUtils::Image<Vector2f>
#define ITMFloat4Image ORUtils::Image<Vector4f>
#define ITMShortImage ORUtils::Image<short>
#define ITMShort3Image ORUtils::Image<Vector3s>
#define ITMShort4Image ORUtils::Image<Vector4s>
#define ITMUShortImage ORUtils::Image<ushort>
#define ITMUIntImage ORUtils::Image<uint>
#define ITMIntImage ORUtils::Image<int>
#define ITMUCharImage ORUtils::Image<uchar>
#define ITMUChar4Image ORUtils::Image<Vector4u>
#define ITMBoolImage ORUtils::Image<bool>

#include "ITMRGBDCalib.h"
#include "ITMCalibIO.h"


namespace ITMLib
{
	namespace Objects
	{
		/** \brief
		    Represents a single "view", i.e. RGB and depth images along
		    with all intrinsic and relative calibration information
		*/
		class ITMView
		{
		public:
			/// Intrinsic calibration information for the view.
			ITMRGBDCalib *calib;

			/// RGB colour image.
			ITMUChar4Image *rgb; 

			/// Float valued depth image converted from disparity image, 
            /// if available according to @ref inputImageType.
			ITMFloatImage *depth;

			ITMView(const ITMRGBDCalib *calibration, Vector2i imgSize_rgb, Vector2i imgSize_d, bool useGPU)
			{
				this->calib = new ITMRGBDCalib(*calibration);
				this->rgb = new ITMUChar4Image(imgSize_rgb, true, useGPU);
				this->depth = new ITMFloatImage(imgSize_d, true, useGPU);
			}

			virtual ~ITMView(void)
			{
				delete calib;
				delete rgb;
				delete depth;
			}
		};
	}
}

