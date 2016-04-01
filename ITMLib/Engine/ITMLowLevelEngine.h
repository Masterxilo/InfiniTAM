#pragma once

#include "CUDADefines.h"
#include "Image.h"
#include "ITMLibDefines.h"
#include "ITMPixelUtils.h"


template<bool withHoles = false, typename T>
CPU_AND_GPU inline void filterSubsample(
    DEVICEPTR(T) *imageData_out, int x, int y, Vector2i newDims,
    const CONSTPTR(T) *imageData_in, Vector2i oldDims)
{
    int src_pos_x = x * 2, src_pos_y = y * 2;
    T pixel_out = 0.0f, pixel_in; 
    float no_good_pixels = 0.0f;

#define sample(dx,dy) \
    pixel_in = imageData_in[(src_pos_x + dx) + (src_pos_y + dy) * oldDims.x]; \
	if (!withHoles || isLegalColor(pixel_in)) { pixel_out += pixel_in; no_good_pixels++; }

    sample(0, 0);
    sample(1, 0);
    sample(0, 1);
    sample(1, 1);
#undef sample

    if (no_good_pixels > 0) pixel_out /= no_good_pixels;
    else if (withHoles) pixel_out = IllegalColor<T>::make();

    imageData_out[pixelLocId(x, y, newDims)] = pixel_out;
}

// device functions
#define FILTER(FILTERNAME)\
template<bool withHoles, typename T>\
static KERNEL FILTERNAME ## _device(T *imageData_out, Vector2i newDims, const T *imageData_in, Vector2i oldDims) {\
    int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;\
    if (x > newDims.x - 1 || y > newDims.y - 1) return;\
    FILTERNAME<withHoles>(imageData_out, x, y, newDims, imageData_in, oldDims);\
}

FILTER(filterSubsample)



            // host methods
#define FILTERMETHOD(METHODNAME, WITHHOLES)\
            template<typename T>\
                void METHODNAME (ORUtils::Image<T> *image_out, const ORUtils::Image<T> *image_in) {\
                    Vector2i oldDims = image_in->noDims; \
                    Vector2i newDims; newDims.x = image_in->noDims.x / 2; newDims.y = image_in->noDims.y / 2; \
                    \
                    image_out->ChangeDims(newDims); \
                    \
                    const T *imageData_in = image_in->GetData(MEMORYDEVICE_CUDA); \
                    T *imageData_out = image_out->GetData(MEMORYDEVICE_CUDA); \
                    \
                    dim3 blockSize(16, 16); \
                    dim3 gridSize((int)ceil((float)newDims.x / (float)blockSize.x), (int)ceil((float)newDims.y / (float)blockSize.y)); \
                    \
                    filterSubsample_device<WITHHOLES> << <gridSize, blockSize >> >(imageData_out, newDims, imageData_in, oldDims); \
            }

FILTERMETHOD(FilterSubsample, false)
FILTERMETHOD(FilterSubsampleWithHoles, WITH_HOLES)
