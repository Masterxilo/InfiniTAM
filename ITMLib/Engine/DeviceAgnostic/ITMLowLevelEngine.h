/// \file this
// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../Utils/ITMLibDefines.h"

_CPU_AND_GPU_CODE_ inline void filterSubsample(DEVICEPTR(Vector4u) *imageData_out, int x, int y, Vector2i newDims, 
	const CONSTPTR(Vector4u) *imageData_in, Vector2i oldDims)
{
	int src_pos_x = x * 2, src_pos_y = y * 2;
	Vector4u pixel_out, pixels_in[4];

	pixels_in[0] = imageData_in[(src_pos_x + 0) + (src_pos_y + 0) * oldDims.x];
	pixels_in[1] = imageData_in[(src_pos_x + 1) + (src_pos_y + 0) * oldDims.x];
	pixels_in[2] = imageData_in[(src_pos_x + 0) + (src_pos_y + 1) * oldDims.x];
	pixels_in[3] = imageData_in[(src_pos_x + 1) + (src_pos_y + 1) * oldDims.x];

	pixel_out = (pixels_in[0] + pixels_in[1] + pixels_in[2] + pixels_in[3]) / 4;

	imageData_out[x + y * newDims.x] = pixel_out;
}

_CPU_AND_GPU_CODE_ inline void filterSubsampleWithHoles(DEVICEPTR(float) *imageData_out, int x, int y, Vector2i newDims, 
	const CONSTPTR(float) *imageData_in, Vector2i oldDims)
{
	int src_pos_x = x * 2, src_pos_y = y * 2;
	float pixel_out = 0.0f, pixel_in, no_good_pixels = 0.0f;

#define sample(dx,dy) \
    pixel_in = imageData_in[(src_pos_x + dx) + (src_pos_y + dy) * oldDims.x]; \
	if (pixel_in > 0.0f) { pixel_out += pixel_in; no_good_pixels++; }

    sample(0, 0);
    sample(1, 0);
    sample(0, 1);
    sample(1, 1);
#undef sample

	if (no_good_pixels > 0) pixel_out /= no_good_pixels;

	imageData_out[x + y * newDims.x] = pixel_out;
}

_CPU_AND_GPU_CODE_ inline void filterSubsampleWithHoles(DEVICEPTR(Vector4f) *imageData_out, int x, int y, Vector2i newDims, 
	const CONSTPTR(Vector4f) *imageData_in, Vector2i oldDims)
{
	int src_pos_x = x * 2, src_pos_y = y * 2;
	Vector4f pixel_out = 0.0f, pixel_in; float no_good_pixels = 0.0f;

	pixel_in = imageData_in[(src_pos_x + 0) + (src_pos_y + 0) * oldDims.x];
	if (pixel_in.w >= 0) { pixel_out += pixel_in; no_good_pixels++; }

	pixel_in = imageData_in[(src_pos_x + 1) + (src_pos_y + 0) * oldDims.x];
	if (pixel_in.w >= 0) { pixel_out += pixel_in; no_good_pixels++; }

	pixel_in = imageData_in[(src_pos_x + 0) + (src_pos_y + 1) * oldDims.x];
	if (pixel_in.w >= 0) { pixel_out += pixel_in; no_good_pixels++; }

	pixel_in = imageData_in[(src_pos_x + 1) + (src_pos_y + 1) * oldDims.x];
	if (pixel_in.w >= 0) { pixel_out += pixel_in; no_good_pixels++; }

	if (no_good_pixels > 0) pixel_out /= no_good_pixels;
	else { pixel_out.w = -1.0f; }

	imageData_out[x + y * newDims.x] = pixel_out;
}

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
_CPU_AND_GPU_CODE_ inline void gradientX(
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
_CPU_AND_GPU_CODE_ inline void gradientY(DEVICEPTR(Vector4s) *grad, int x, int y, const CONSTPTR(Vector4u) *image, Vector2i imgSize)
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
