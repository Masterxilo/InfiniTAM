// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

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

// TODO support any convolution kernel (more general)
/**
Convolution of image around x,y with
\f[
\frac{1}{8}
\begin{bmatrix}
-1 &0& 1\\
-2 &0& 2\\
-1 &0& 1\\
\end{bmatrix}
\f]
Gradient and color image are of size imgSize.
*/
CPU_AND_GPU inline void gradientX(
    DEVICEPTR(Vector4s) *grad, int x, int y,
    const CONSTPTR(Vector4u) *image,
    Vector2i imgSize)
{
    Vector4s d1, d2, d3, d_out;

    d1.x = image[(x + 1) + (y - 1) * imgSize.x].x - image[(x - 1) + (y - 1) * imgSize.x].x;
    d1.y = image[(x + 1) + (y - 1) * imgSize.x].y - image[(x - 1) + (y - 1) * imgSize.x].y;
    d1.z = image[(x + 1) + (y - 1) * imgSize.x].z - image[(x - 1) + (y - 1) * imgSize.x].z;

    d2.x = image[(x + 1) + (y)* imgSize.x].x - image[(x - 1) + (y)* imgSize.x].x;
    d2.y = image[(x + 1) + (y)* imgSize.x].y - image[(x - 1) + (y)* imgSize.x].y;
    d2.z = image[(x + 1) + (y)* imgSize.x].z - image[(x - 1) + (y)* imgSize.x].z;

    d3.x = image[(x + 1) + (y + 1) * imgSize.x].x - image[(x - 1) + (y + 1) * imgSize.x].x;
    d3.y = image[(x + 1) + (y + 1) * imgSize.x].y - image[(x - 1) + (y + 1) * imgSize.x].y;
    d3.z = image[(x + 1) + (y + 1) * imgSize.x].z - image[(x - 1) + (y + 1) * imgSize.x].z;

    d1.w = d2.w = d3.w = 2 * 255;

    d_out = (d1 + 2 * d2 + d3) / 8;

    grad[x + y * imgSize.x] = d_out;
}

/**
Convolution of image around x,y with
\f[
\frac{1}{8}
\begin{bmatrix}
-1 &-2& -1\\
0 &0& 0\\
1 &2& 1\\
\end{bmatrix}
\f]
Gradient and color image are of size imgSize.
*/
CPU_AND_GPU inline void gradientY(DEVICEPTR(Vector4s) *grad, int x, int y, const CONSTPTR(Vector4u) *image, Vector2i imgSize)
{
    Vector4s d1, d2, d3, d_out;

    d1.x = image[(x - 1) + (y + 1) * imgSize.x].x - image[(x - 1) + (y - 1) * imgSize.x].x;
    d1.y = image[(x - 1) + (y + 1) * imgSize.x].y - image[(x - 1) + (y - 1) * imgSize.x].y;
    d1.z = image[(x - 1) + (y + 1) * imgSize.x].z - image[(x - 1) + (y - 1) * imgSize.x].z;

    d2.x = image[(x)+(y + 1) * imgSize.x].x - image[(x)+(y - 1) * imgSize.x].x;
    d2.y = image[(x)+(y + 1) * imgSize.x].y - image[(x)+(y - 1) * imgSize.x].y;
    d2.z = image[(x)+(y + 1) * imgSize.x].z - image[(x)+(y - 1) * imgSize.x].z;

    d3.x = image[(x + 1) + (y + 1) * imgSize.x].x - image[(x + 1) + (y - 1) * imgSize.x].x;
    d3.y = image[(x + 1) + (y + 1) * imgSize.x].y - image[(x + 1) + (y - 1) * imgSize.x].y;
    d3.z = image[(x + 1) + (y + 1) * imgSize.x].z - image[(x + 1) + (y - 1) * imgSize.x].z;

    d1.w = d2.w = d3.w = 2 * 255;

    d_out.x = (d1.x + 2 * d2.x + d3.x) / 8;
    d_out.y = (d1.y + 2 * d2.y + d3.y) / 8;
    d_out.z = (d1.z + 2 * d2.z + d3.z) / 8;
    d_out.w = (d1.w + 2 * d2.w + d3.w) / 8;

    grad[x + y * imgSize.x] = d_out;
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

#define gradient__device(X_or_Y) \
static KERNEL gradient## X_or_Y ##_device(Vector4s *grad, const Vector4u *image, Vector2i imgSize) { \
    int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;\
    if (x < 2 || x > imgSize.x - 2 || y < 2 || y > imgSize.y - 2) return;\
    gradient ## X_or_Y (grad, x, y, image, imgSize);\
}

gradient__device(X);
gradient__device(Y);

namespace ITMLib
{
	namespace Engine
	{
		class ITMLowLevelEngine
		{
        public:
            template<typename T>
            void CopyImage(ORUtils::Image<T> *image_out, const ORUtils::Image<T> *image_in) const
            {
                cudaSafeCall(cudaMemcpy(
                    image_out->GetData(MEMORYDEVICE_CUDA),
                    image_in->GetData(MEMORYDEVICE_CUDA),
                    image_in->dataSize * sizeof(T), cudaMemcpyDeviceToDevice));
            }

            // host methods
#define FILTERMETHOD(METHODNAME, WITHHOLES)\
            template<typename T>\
                void METHODNAME (ORUtils::Image<T> *image_out, const ORUtils::Image<T> *image_in) const {\
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

#define GRADIENTMETHOD(X_OR_Y)\
    void Gradient ## X_OR_Y(ITMShort4Image *grad_out, const ITMUChar4Image *image_in) const {\
	grad_out->ChangeDims(image_in->noDims);\
	Vector2i imgSize = image_in->noDims;\
    \
	Vector4s *grad = grad_out->GetData(MEMORYDEVICE_CUDA);\
	const Vector4u *image = image_in->GetData(MEMORYDEVICE_CUDA);\
    \
	dim3 blockSize(16, 16);\
	dim3 gridSize((int)ceil((float)imgSize.x / (float)blockSize.x), (int)ceil((float)imgSize.y / (float)blockSize.y));\
    \
	cudaSafeCall(cudaMemset(grad, 0, imgSize.x * imgSize.y * sizeof(Vector4s)));\
    \
    gradient ## X_OR_Y ## _device << <gridSize, blockSize >> >(grad, image, imgSize);\
}

GRADIENTMETHOD(X)
GRADIENTMETHOD(Y)

		};
	}
}
