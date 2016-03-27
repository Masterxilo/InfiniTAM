// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMLibDefines.h"
#include "ITMLibSettings.h"

#include "ITMView.h"
#include "ITMRGBDCalib.h"

#include "ITMLibDefines.h"

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
            const ITMRGBDCalib * const calib;
            ITMShortImage *rawDepthImage;

        public:
			void UpdateView(ITMView **view, ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage);


            ITMViewBuilder(const ITMRGBDCalib *calib) : calib(calib), rawDepthImage(0) {}

			virtual ~ITMViewBuilder()
			{
                if (this->rawDepthImage != NULL) delete this->rawDepthImage;
			}
		};
	}
}

