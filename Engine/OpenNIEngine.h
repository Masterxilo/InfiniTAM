// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ImageSourceEngine.h"

#if (!defined USING_CMAKE) && (defined _MSC_VER)
#pragma comment(lib, "OpenNI2")
#endif

namespace InfiniTAM
{
	namespace Engine
    {
        /** Open Natural Interaction:  open source software project focused on certifying and improving interoperability of natural user interfaces and organic user interfaces for Natural Interaction (NI) devices, applications that use those devices and middleware that facilitates access and use of such devices.[1]
PrimeSense, who was founding member of OpenNI, shutdown the original OpenNI project when it was acquired by Apple on November 24, 2013, since then Occipital and other former partners of PrimeSense are still keeping a forked version of OpenNI 2 (OpenNI version 2) active as an open source software*/
        class OpenNIEngine : public ImageSourceEngine
		{
		private:
			class PrivateData;
			PrivateData *data;
			Vector2i imageSize_rgb, imageSize_d;
			bool colorAvailable, depthAvailable;
		public:
			OpenNIEngine(const char *calibFilename, const char *deviceURI = NULL, const bool useInternalCalibration = false,
				Vector2i imageSize_rgb = Vector2i(640, 480), Vector2i imageSize_d = Vector2i(640, 480));
			~OpenNIEngine();

			bool hasMoreImages(void);
			void getImages(ITMUChar4Image *rgb, ITMShortImage *rawDepth);
			Vector2i getDepthImageSize(void);
			Vector2i getRGBImageSize(void);
		};
	}
}

