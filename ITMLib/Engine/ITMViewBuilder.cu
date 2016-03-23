// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMViewBuilder.h"
#include "CUDADefines.h"
#include "MemoryBlock.h"

#include "ITMPixelUtils.h"

using namespace ITMLib::Engine;
using namespace ORUtils;

//---------------------------------------------------------------------------
//
// kernel function implementation
//
//---------------------------------------------------------------------------


/// case ITMDisparityCalib::TRAFO_KINECT:
/// Raw values are transformed according to \f$\frac{8 c_2 f_x}{c_1 - d}\f$
CPU_AND_GPU inline void convertDisparityToDepth(DEVICEPTR(float) *d_out, int x, int y, const CONSTPTR(short) *d_in,
    Vector2f disparityCalibParams, float fx_depth, Vector2i imgSize)
{
    int locId = x + y * imgSize.x;

    short disparity = d_in[locId];
    float disparity_tmp = disparityCalibParams.x - (float)(disparity);
    float depth;

    if (disparity_tmp == 0) depth = 0.0;
    else depth =
        8.0f * disparityCalibParams.y * fx_depth / disparity_tmp;

    d_out[locId] = (depth > 0) ? depth : -1.0f;
}

__global__ void convertDisparityToDepth_device(float *d_out, const short *d_in, Vector2f disparityCalibParams, float fx_depth, Vector2i imgSize)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x >= imgSize.x) || (y >= imgSize.y)) return;

    convertDisparityToDepth(d_out, x, y, d_in, disparityCalibParams, fx_depth, imgSize);
}

//---------------------------------------------------------------------------
//
// host methods
//
//---------------------------------------------------------------------------

void ITMViewBuilder::UpdateView(ITMView **view_ptr, ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage)
{
	if (*view_ptr == NULL)
	{
		*view_ptr = new ITMView(calib, rgbImage->noDims, rawDepthImage->noDims, true);
		if (this->shortImage != NULL) delete this->shortImage;
		this->shortImage = new ITMShortImage(rawDepthImage->noDims, true, true);
		if (this->floatImage != NULL) delete this->floatImage;
		this->floatImage = new ITMFloatImage(rawDepthImage->noDims, true, true);
	}

	ITMView *view = *view_ptr;

	view->rgb->SetFrom(rgbImage, MemoryBlock<Vector4u>::CPU_TO_CUDA);
	this->shortImage->SetFrom(rawDepthImage, MemoryBlock<short>::CPU_TO_CUDA);

    this->ConvertDisparityToDepth(view->depth, this->shortImage, &(view->calib->intrinsics_d), view->calib->disparityCalib.params);
}

void ITMViewBuilder::ConvertDisparityToDepth(ITMFloatImage *depth_out, const ITMShortImage *depth_in, const ITMIntrinsics *depthIntrinsics,
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