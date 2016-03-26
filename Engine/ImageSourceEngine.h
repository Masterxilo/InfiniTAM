// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../ITMLib/ITMLib.h"

namespace InfiniTAM
{
	namespace Engine
	{
        /** Represents a source of depth & color image data, together with the calibration of those two cameras. */
		class ImageSourceEngine
		{
		public:
			ITMRGBDCalib calib;

			ImageSourceEngine(const char *calibFilename);
			virtual ~ImageSourceEngine() {}

			virtual bool hasMoreImages(void) = 0;
			virtual void getImages(ITMUChar4Image *rgb, ITMShortImage *rawDepth) = 0;
			virtual Vector2i getDepthImageSize(void) = 0;
			virtual Vector2i getRGBImageSize(void) = 0;
		};

		class ImageFileReader : public ImageSourceEngine
		{
		private:
			static const int BUF_SIZE = 2048;
			char rgbImageMask[BUF_SIZE];
			char depthImageMask[BUF_SIZE];

			ITMUChar4Image *cached_rgb;
			ITMShortImage *cached_depth;

			void loadIntoCache();
			int cachedFrameNo;
			int currentFrameNo;
		public:

			ImageFileReader(const char *calibFilename, const char *rgbImageMask, const char *depthImageMask);
			~ImageFileReader();

			bool hasMoreImages(void);
            /// images must exist on cpu and be of correct dimensions
			void getImages(ITMUChar4Image *rgb, ITMShortImage *rawDepth);
			Vector2i getDepthImageSize(void);
			Vector2i getRGBImageSize(void);
		};
	}
}

