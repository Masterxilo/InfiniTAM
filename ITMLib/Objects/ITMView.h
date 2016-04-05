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
#include "cameraimage.h"
#include "ITMRenderState.h"

class ITMView;
extern __managed__ ITMView * currentView = 0;

/** \brief
	Represents a single "view", i.e. RGB and depth images along
	with all intrinsic, relative and extrinsic calibration information
*/
class ITMView : Managed {
    /// RGB colour image.
    ITMUChar4Image * const rgbData;

    /// Float valued depth image converted from disparity image, 
    /// if available according to @ref inputImageType.
    ITMFloatImage * const depthData;

    ITMRenderState* tempICPRenderState;
    ITMTrackingState* tempTrackingState;

    Vector2i imgSize_d() const {
        assert(depthImage->imgSize().area() > 1);
        return depthImage->imgSize();
    }
public:
    ITMRenderState* getRenderState() {
        if (!tempICPRenderState) tempICPRenderState = new ITMRenderState(imgSize_d());
        return tempICPRenderState;
    }
    ITMTrackingState* getTrackingState() {
        if (!tempTrackingState) tempTrackingState = new ITMTrackingState(imgSize_d());
        return tempTrackingState;
    }

	/// Intrinsic calibration information for the view.
    ITMRGBDCalib const * const calib;

	/// RGB colour image.
    CameraImage<Vector4u> * colorImage;

	/// Float valued depth image converted from disparity image, 
    /// if available according to @ref inputImageType.
    DepthImage * depthImage;

    void ChangePose(Matrix4f M_d) {
        // TODO delete old ones!
        auto depthCs = new CoordinateSystem(M_d.getInv());
        depthImage = new DepthImage(
            depthData,
            depthCs, 
            calib->intrinsics_d.projectionParamsSimple.all);

        Matrix4f M_rgb = calib->trafo_rgb_to_depth.calib_inv * M_d;
        auto colorCs = new CoordinateSystem(M_rgb.getInv());
        colorImage = new CameraImage<Vector4u>(
            rgbData,
            colorCs,
            calib->intrinsics_rgb.projectionParamsSimple.all);
    }

	ITMView(const ITMRGBDCalib *calibration) :
        tempICPRenderState(0),
        calib(new ITMRGBDCalib(*calibration)),
        rgbData(new ITMUChar4Image()),
        depthData(new ITMFloatImage()) {

        M_d = trackingState->pose_d->GetM();
        Matrix4f M_rgb = view->calib->trafo_rgb_to_depth.calib_inv * M_d;

        auto depthCs = new CoordinateSystem(M_d.getInv());
        depthImage = new DepthImage(view->depth, depthCs, view->calib->intrinsics_d.projectionParamsSimple.all);

        auto colorCs = new CoordinateSystem(M_rgb.getInv());
        colorImage = new CameraImage<Vector4u>(view->rgb, colorCs, view->calib->intrinsics_rgb.projectionParamsSimple.all);

	}

    static std::string depthConversionType;

    /// prepare image and turn it into a world-scale depth image (if it is a disparity image)
    void ITMView::ChangeImages(
        ITMUChar4Image *rgbImage,
        ITMShortImage *rawDepthImage);
};

