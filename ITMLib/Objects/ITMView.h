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


/** \brief
	Represents a single "view", i.e. RGB and depth images along
	with all intrinsic and relative calibration information
*/
class ITMView
{
    ITMShortImage * const rawDepthImageGPU;

public:
	/// Intrinsic calibration information for the view.
    ITMRGBDCalib const * const calib;

	/// RGB colour image.
    ITMUChar4Image * const rgb;

	/// Float valued depth image converted from disparity image, 
    /// if available according to @ref inputImageType.
    ITMFloatImage * const depth;

	ITMView(const ITMRGBDCalib *calibration) :
        calib(new ITMRGBDCalib(*calibration)),
        rgb(new ITMUChar4Image()),
        depth(new ITMFloatImage()),
        rawDepthImageGPU(new ITMShortImage()) {
	}

    static std::string depthConversionType;
    void ITMView::Update(
        ITMUChar4Image *rgbImage,
        ITMShortImage *rawDepthImage);

	virtual ~ITMView(void)
	{
		delete calib;
		delete rgb;
        delete depth;
        delete rawDepthImageGPU;
	}
};

