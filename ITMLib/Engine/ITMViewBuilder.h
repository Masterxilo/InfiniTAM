// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Utils/ITMLibDefines.h"
#include "../Utils/ITMLibSettings.h"

#include "../Objects/ITMView.h"
#include "../Objects/ITMRGBDCalib.h"

#include "../Utils/ITMLibDefines.h"

using namespace ITMLib::Objects;

namespace ITMLib
{
	namespace Engine
	{
		/** Takes a disparity image and turns it into a depth image, 
		*/
		class ITMViewBuilder
		{
		protected:
			const ITMRGBDCalib *calib;
            /// rawDepthImage
			ITMShortImage *shortImage;
			ITMFloatImage *floatImage;

        public:
			void ConvertDisparityToDepth(ITMFloatImage *depth_out, const ITMShortImage *disp_in, const ITMIntrinsics *depthIntrinsics,
				Vector2f disparityCalibParams);

			void UpdateView(ITMView **view, ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage);


			ITMViewBuilder(const ITMRGBDCalib *calib)
			{
				this->calib = calib;
				this->shortImage = NULL;
				this->floatImage = NULL;
			}

			virtual ~ITMViewBuilder()
			{
				if (this->shortImage != NULL) delete this->shortImage;
				if (this->floatImage != NULL) delete this->floatImage;
			}
		};
	}
}

