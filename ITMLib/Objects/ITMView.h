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

class ITMView;
#ifndef ITMVIEW_ // AVOID "MULTIPLE DEFINITION"
extern __managed__ ITMView * currentView; 
#endif
/** \brief
	Represents a single "view", i.e. RGB and depth images along
	with all intrinsic, relative and extrinsic calibration information
*/
class ITMView : public Managed {
    /// RGB colour image.
    ITMUChar4Image * const rgbData;

    /// Float valued depth image converted from disparity image, 
    /// if available according to @ref inputImageType.
    ITMFloatImage * const depthData;
    ITMShortImage * const rawDepthImageGPU;

    Vector2i imgSize_d() const {
        assert(depthImage->imgSize().area() > 1);
        return depthImage->imgSize();
    }
public:

	/// Intrinsic calibration information for the view.
    ITMRGBDCalib const * const calib;

	/// RGB colour image.
    CameraImage<Vector4u> * const colorImage;

	/// Float valued depth image converted from disparity image, 
    /// if available according to @ref inputImageType.
    DepthImage * const depthImage;

    void ChangePose(Matrix4f M_d) {
        assert(abs(M_d.GetR().det() - 1) < 0.00001);
        // TODO delete old ones!
        auto depthCs = new CoordinateSystem(M_d.getInv());
        depthImage->eyeCoordinates = depthCs;
            
        Matrix4f M_rgb = calib->trafo_rgb_to_depth.calib_inv * M_d;
        auto colorCs = new CoordinateSystem(M_rgb.getInv());
        colorImage->eyeCoordinates = colorCs;
    }

	ITMView(const ITMRGBDCalib *calibration) :
        calib(new ITMRGBDCalib(*calibration)),
        rgbData(new ITMUChar4Image()),
        depthData(new ITMFloatImage()), 
        rawDepthImageGPU(new ITMShortImage()),
        
        depthImage(new DepthImage(depthData, CoordinateSystem::global(), calib->intrinsics_d.projectionParamsSimple.all)),
        colorImage(new CameraImage<Vector4u>(rgbData, CoordinateSystem::global(), calib->intrinsics_rgb.projectionParamsSimple.all)) {
        assert(colorImage->eyeCoordinates == CoordinateSystem::global());
        assert(depthImage->eyeCoordinates == CoordinateSystem::global());

        Matrix4f M; M.setIdentity();
        ChangePose(M);
        assert(!(colorImage->eyeCoordinates == CoordinateSystem::global()));
        assert(!(depthImage->eyeCoordinates == CoordinateSystem::global()));
        assert(!(colorImage->eyeCoordinates == depthImage->eyeCoordinates));
	}

    static std::string depthConversionType;

    /// prepare image and turn it into a world-scale depth image (if it is a disparity image)
    void ITMView::ChangeImages(
        ITMUChar4Image *rgbImage,
        ITMShortImage *rawDepthImage);
};

