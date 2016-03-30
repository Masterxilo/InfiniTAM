#include "ITMViewBuilder.h"
#include "CUDADefines.h"
#include "MemoryBlock.h"
#include "ITMPixelUtils.h"
using namespace ITMLib::Engine;
using namespace ORUtils;

/// case ITMDisparityCalib::TRAFO_KINECT:
/// Raw values are transformed according to \f$\frac{8 c_2 f_x}{c_1 - d}\f$
/// Where f_x is the x focal length of the depth camera.
//!< \returns >0 on success, outputs INVALID_DEPTH on failure
CPU_AND_GPU inline float convertDisparityToDepth(
    const short disparity, 
    const Vector2f disparityCalibParams, 
    const float fx_depth)
{ 
    // http://qianyi.info/scenedata.html
    // for fountain // TODO make configurable. Note that the tests expect kinect.
    return disparity <= 0 ? INVALID_DEPTH : (disparity) / 1000.f;
    
    // for kinect, raw (e.g. Teddy)
    float disparity_tmp = disparityCalibParams.x - (float)(disparity);
    float depth;

    if (disparity_tmp == 0)
        depth = 0.0;
    else depth = 8.0f * disparityCalibParams.y * fx_depth / disparity_tmp;

    return (depth > 0) ? depth : INVALID_DEPTH;

   
}

KERNEL convertDisparityToDepth_device(float *d_out, const short *d_in, Vector2f disparityCalibParams, float fx_depth, Vector2i imgSize)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x >= imgSize.x) || (y >= imgSize.y)) return;

    const int locId = x + y * imgSize.x;
    d_out[locId] = convertDisparityToDepth(d_in[locId], disparityCalibParams, fx_depth);
}

static void ConvertDisparityToDepth(
    ITMFloatImage* depth_out,
    const ITMShortImage* depth_in,
    const ITMIntrinsics *depthIntrinsics,
    Vector2f disparityCalibParams)
{
    Vector2i imgSize = depth_in->noDims;

    const short *d_in = depth_in->GetData(MEMORYDEVICE_CUDA);
    float *d_out = depth_out->GetData(MEMORYDEVICE_CUDA);

    float fx_depth = depthIntrinsics->projectionParamsSimple.fx;

    dim3 blockSize(16, 16);
    dim3 gridSize((int)ceil((float)imgSize.x / (float)blockSize.x), (int)ceil((float)imgSize.y / (float)blockSize.y));

    convertDisparityToDepth_device << <gridSize, blockSize >> >(d_out, d_in, disparityCalibParams, fx_depth, imgSize);
}

void ITMViewBuilder::UpdateView(ITMView **view_ptr, ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage)
{
	if (*view_ptr == NULL)
	{
		*view_ptr = new ITMView(calib, rgbImage->noDims, rawDepthImage->noDims, true);
        if (this->rawDepthImage != NULL) delete this->rawDepthImage;
        this->rawDepthImage = new ITMShortImage(rawDepthImage->noDims, true, true);
	}

	ITMView *view = *view_ptr;

    // copy rgb as-is
	view->rgb->SetFrom(rgbImage, MemoryBlock<Vector4u>::CPU_TO_CUDA);

    // copy rawDepthImage to gpu, then store ConvertDisparityToDepth result in view->depth
    this->rawDepthImage->SetFrom(rawDepthImage, MemoryBlock<short>::CPU_TO_CUDA);
    ConvertDisparityToDepth(
        view->depth,
        this->rawDepthImage,
        &(view->calib->intrinsics_d),
        view->calib->disparityCalib.params);
}
