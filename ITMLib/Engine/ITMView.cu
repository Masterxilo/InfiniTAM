#include "ITMView.h"
#include "CUDADefines.h"
#include "MemoryBlock.h"
#include "ITMPixelUtils.h"
#include "itmlibdefines.h"

static __managed__ float *d_out;
static __managed__ const short *d_in;
static __managed__ Vector2f disparityCalibParams;
static __managed__ float fx_depth;

/// current depth & color image
__managed__ ITMView * currentView;


/// case ITMDisparityCalib::TRAFO_KINECT:
/// Raw values are transformed according to \f$\frac{8 c_2 f_x}{c_1 - d}\f$
/// Where f_x is the x focal length of the depth camera.
//!< \returns >0 on success, outputs INVALID_DEPTH on failure
struct ConvertDisparityToDepth {
    forEachPixelNoImage_process() {
        const short disparity = d_in[locId];

        // for kinect, raw (e.g. Teddy)
        float disparity_tmp = disparityCalibParams.x - (float)(disparity);
        float depth;

        if (disparity_tmp == 0)
            depth = 0.0;
        else depth = 8.0f * disparityCalibParams.y * fx_depth / disparity_tmp;

        d_out[locId] = (depth > 0) ? depth : INVALID_DEPTH;
    }
};

struct ScaleAndValidateDepth {
    forEachPixelNoImage_process() {
        const short depth = d_in[locId];
        // http://qianyi.info/scenedata.html
        // for fountain
        d_out[locId] = depth <= 0 ? INVALID_DEPTH : (depth) / 1000.f;
    }
};

std::string ITMView::depthConversionType;

void ITMView::Update(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage)
{
    // copy rgb as-is
	rgb->SetFrom(rgbImage, MemoryBlock<Vector4u>::CPU_TO_CUDA);

    // copy rawDepthImage to gpu, then store ConvertDisparityToDepth result in view->depth
    rawDepthImageGPU->SetFrom(rawDepthImage, MemoryBlock<short>::CPU_TO_CUDA);

    ::d_in = rawDepthImageGPU->GetData(MEMORYDEVICE_CUDA);


    depth->ChangeDims(rawDepthImageGPU->noDims);

    ::d_out = depth->GetData(MEMORYDEVICE_CUDA);
    ::fx_depth = calib->intrinsics_d.projectionParamsSimple.fx;
    ::disparityCalibParams = calib->disparityCalib.params;

#define dcp(name) if (ITMView::depthConversionType == #name) {forEachPixelNoImage<name>(depth->noDims); return;}
    dcp(ConvertDisparityToDepth);
    dcp(ScaleAndValidateDepth);

    assert(false); // unkown depth conversion type
}
