/// \file Apply disparityCalibParams
// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../Utils/ITMLibDefines.h"
#include "ITMPixelUtils.h"

/// case ITMDisparityCalib::TRAFO_KINECT:
/// Raw values are transformed according to \f$\frac{8 c_2 f_x}{c_1 - d}\f$
_CPU_AND_GPU_CODE_ inline void convertDisparityToDepth(DEVICEPTR(float) *d_out, int x, int y, const CONSTPTR(short) *d_in,
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

/// case ITMDisparityCalib::TRAFO_AFFINE:
/// Raw values are transformed according to \f$c_1 d + c_2\f$
_CPU_AND_GPU_CODE_ inline void convertDepthAffineToFloat(DEVICEPTR(float) *d_out, int x, int y, const CONSTPTR(short) *d_in, Vector2i imgSize, Vector2f depthCalibParams)
{
	int locId = x + y * imgSize.x;

	short depth_in = d_in[locId];
	d_out[locId] = (
        (depth_in <= 0)||(depth_in > 32000)) ? -1.0f : 
        (float)depth_in * depthCalibParams.x + depthCalibParams.y;
}

#define FILTER_MAX_RADIUS 2
#define MEAN_SIGMA_L 1.2232f
/// We apply a bilateral filter  to the raw depth map to
/// obtain a discontinuity preserved depth map with reduced noise
_CPU_AND_GPU_CODE_ inline void filterDepth(
    DEVICEPTR(float) *imageData_out,
    const CONSTPTR(float) *imageData_in,
    int x, int y, Vector2i imgDims)
{
	float z, tmpz, dz, final_depth = 0.0f, w, w_sum = 0.0f;

    // u is given by (x,y)

    // R_k(u)
	z = imageData_in[x + y * imgDims.x]; 
    // Illegal input
	if (z < 0.0f) { imageData_out[x + y * imgDims.x] = -1.0f; return; }

	float sigma_z = 1.0f / (0.0012f + 0.0019f*(z - 0.4f)*(z - 0.4f) + 0.0001f / sqrt(z) * 0.25f);

    for (int i = -FILTER_MAX_RADIUS, count = 0; i <= FILTER_MAX_RADIUS; i++) for (int j = -FILTER_MAX_RADIUS; j <= FILTER_MAX_RADIUS; j++, count++)
    {
        // R_k(q)
		tmpz = imageData_in[(x + j) + (y + i) * imgDims.x];
		if (tmpz < 0.0f) continue;

		dz = (tmpz - z);
        dz *= dz;

        // Filter kernel
        // abs(i) + abs(j) is approximation to (||u - q||_2)
		w = expf(-0.5f * ((abs(i) + abs(j))*MEAN_SIGMA_L*MEAN_SIGMA_L + dz * sigma_z * sigma_z));
		
        w_sum += w;
		final_depth += w*tmpz;
	}

	final_depth /= w_sum; // 1/W_p (normalizing constant)
	imageData_out[x + y*imgDims.x] = final_depth;
}


/**
Computing the surface normal in image space.

In image space, since the normals are computed on a regular grid,
there are only 4 uninterpolated read operations followed by a cross-product.

Well we also read the center value.

\returns normal_out[idx].w = sigmaZ_out[idx] = -1 on error where idx = x + y * imgDims.x
*/
_CPU_AND_GPU_CODE_ inline void computeNormalAndWeight(
    const float *depth_in,
    Vector4f *normal_out, //!< .w == -1 on error
    float *sigmaZ_out, //!< == -1 on error
    int x, int y,
    Vector2i imgDims, Vector4f projParams)
{
    ///
    // TODO this should be done in the code calling this
    Vector4f invProjParams = projParams;
    invProjParams.x = 1.0f / invProjParams.x;
    invProjParams.y = 1.0f / invProjParams.y;
    ///

	Vector3f outNormal;

	int idx = x + y * imgDims.x;

	float z = depth_in[x + y * imgDims.x];
#define returnError() \
    {\
        normal_out[idx].w = -1.0f;\
        sigmaZ_out[idx] = -1;\
        return;\
    }

    if (z < 0.0f) returnError();

	// first compute the normal
	Vector3f xp1_y, xm1_y, x_yp1, x_ym1;
	Vector3f diff_x(0.0f, 0.0f, 0.0f), diff_y(0.0f, 0.0f, 0.0f);
    
#define unproject(dx, dy) depthTo3DInvProjParams(invProjParams, x + dx, y + dy, depth_in[(x + dx) + (y + dy) * imgDims.x]).toVector3()

    xp1_y = unproject(1, 0);
    x_yp1 = unproject(0, 1);

    xm1_y = unproject(-1, 0);
    x_ym1 = unproject(0, -1);
#undef unproject

    if (xp1_y.z <= 0 || x_yp1.z <= 0 || xm1_y.z <= 0 || x_ym1.z <= 0) returnError();

	// gradients x and y
	diff_x = xp1_y - xm1_y, diff_y = x_yp1 - x_ym1;

	// cross product
    outNormal = cross(diff_x, diff_y);

    if (outNormal.x == 0.0f && outNormal.y == 0 && outNormal.z == 0) returnError();
	outNormal = outNormal.normalised();	
	
    normal_out[idx] = Vector4f(outNormal, 1.0f);

	// now compute weight
	float theta = acosf(outNormal.z);
	float theta_diff = theta / (PI*0.5f - theta);

	sigmaZ_out[idx] = (0.0012f + 0.0019f * (z - 0.4f) * (z - 0.4f) + 0.0001f / sqrt(z) * theta_diff * theta_diff);
}
#undef returnError