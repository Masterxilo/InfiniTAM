// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include <math.h>
#include <ostream>

#include "../../Utils/ITMLibDefines.h"

#define weightedCombine(oldX, oldW, newX, newW) \
    newX = oldW * oldX + newW * newX; \
    newW = oldW + newW;\
    newX /= newW;\
    newW = MIN(newW, maxW);

/// Linearized pixel index
_CPU_AND_GPU_CODE_ inline int pixelLocId(int x, int y, const THREADPTR(Vector2i) &imgSize) {
    return x + y * imgSize.x;
}

template<class TVoxel>
_CPU_AND_GPU_CODE_ inline void updateVoxelColorInformation(
    DEVICEPTR(TVoxel) & voxel,
    Vector3f oldC, int oldW, Vector3f newC, int newW,
    int maxW)
{
    weightedCombine(oldC, oldW, newC, newW);

    // write back
    /// C(X) <-  
    voxel.clr = TO_UCHAR3(newC);
    voxel.w_color = (uchar)newW;
}

template<class TVoxel>
_CPU_AND_GPU_CODE_ inline void updateVoxelDepthInformation(
    DEVICEPTR(TVoxel) & voxel,
    float oldF, int oldW, float newF, int newW,
    int maxW)
{
    weightedCombine(oldF, oldW, newF, newW);

    // write back
    /// D(X) <-  (4)
    voxel.sdf = TVoxel::SDF_floatToValue(newF);
    voxel.w_depth = newW;
}
#undef weightedCombine

/// Computes a position in camera space given a 2d image coordinate and a depth.
/// \f$ z K^{-1}u\f$
/// \param x,y \f$ u\f$
_CPU_AND_GPU_CODE_ inline Vector4f depthTo3D(
    const CONSTPTR(Vector4f) & viewIntrinsics, //!< K
    const THREADPTR(int) & x, const THREADPTR(int) & y,
    const CONSTPTR(float) &depth //!< z
    ) {
    /// Note: division by projection parameters .x, .y, i.e. fx and fy.
    /// The function below takes <inverse projection parameters> which have 1/fx, 1/fy, cx, cy
    Vector4f o;
    o.x = depth * ((float(x) - viewIntrinsics.z) / viewIntrinsics.x);
    o.y = depth * ((float(y) - viewIntrinsics.w) / viewIntrinsics.y);
    o.z = depth;
    o.w = 1.0f;
    return o;
}


_CPU_AND_GPU_CODE_ inline Vector4f depthTo3DInvProjParams(
    const CONSTPTR(Vector4f) & invProjParams, //!< <inverse projection parameters> which contain (1/fx, 1/fy, cx, cy)
    const THREADPTR(int) & x, const THREADPTR(int) & y, const CONSTPTR(float) &depth) {
    Vector4f o;
    o.x = depth * ((float(x) - invProjParams.z) * invProjParams.x);
    o.y = depth * ((float(y) - invProjParams.w) * invProjParams.y);
    o.z = depth;
    o.w = 1.0f;
    return o;
}

_CPU_AND_GPU_CODE_ inline bool projectNoBounds(
    Vector4f projParams, Vector4f pt_camera, Vector2f& pt_image) {
    if (pt_camera.z <= 0) return false;

    pt_image.x = projParams.x * pt_camera.x / pt_camera.z + projParams.z;
    pt_image.y = projParams.y * pt_camera.y / pt_camera.z + projParams.w;

    return true;
}

/// $$\\pi(K p)$$
/// Projects pt_model, given in camera coordinates to 2d image coordinates (dropping depth).
/// \returns false when point projects outside of image
_CPU_AND_GPU_CODE_ inline bool project(
    Vector4f projParams, //!< K 
    const CONSTPTR(Vector2i) & imgSize,
    Vector4f pt_camera, //!< p
    Vector2f& pt_image) {
    if (!projectNoBounds(projParams, pt_camera, pt_image)) return false;

    if (pt_image.x < 0 || pt_image.x > imgSize.x - 1 || pt_image.y < 0 || pt_image.y > imgSize.y - 1) return false;
    // for inner points, when we compute gradients
    // was used like that in computeUpdatedVoxelDepthInfo
    //if ((pt_image.x < 1) || (pt_image.x > imgSize.x - 2) || (pt_image.y < 1) || (pt_image.y > imgSize.y - 2)) return -1;

    return true;
}

/// Reject pixels on the right lower boundary of the image 
// (which have an incomplete forward-neighborhood)
_CPU_AND_GPU_CODE_ inline bool projectExtraBounds(
    Vector4f projParams, const CONSTPTR(Vector2i) & imgSize,
    Vector4f pt_camera, Vector2f& pt_image) {
    if (!projectNoBounds(projParams, pt_camera, pt_image)) return false;

    if (pt_image.x < 0 || pt_image.x > imgSize.x - 2 || pt_image.y < 0 || pt_image.y > imgSize.y - 2) return false;

    return true;
}

/// $$\\pi(K M p)$$
/// Projects pt_model, given in model coordinates to 2d image coordinates (dropping depth).
/// \returns false when point projects outside of image
_CPU_AND_GPU_CODE_ inline bool projectModel(
    Vector4f projParams, Matrix4f M, const CONSTPTR(Vector2i) & imgSize,
    Vector4f pt_model,
    Vector4f& pt_camera, Vector2f& pt_image) {
    pt_camera = M * pt_model;
    return project(projParams, imgSize, pt_camera, pt_image);
}

/// Sample image without interpolation at integer location
template<typename T> _CPU_AND_GPU_CODE_
inline T sampleNearest(
const CONSTPTR(T) *source,
int x, int y,
const CONSTPTR(Vector2i) & imgSize)
{
    return source[pixelLocId(x, y, imgSize)];
}

/// Sample image without interpolation at rounded location
template<typename T> _CPU_AND_GPU_CODE_
inline T sampleNearest(
    const CONSTPTR(T) *source,
    const THREADPTR(Vector2f) & pt_image,
    const CONSTPTR(Vector2i) & imgSize) {
    return source[
        pixelLocId(
            (int)(pt_image.x + 0.5f),
            (int)(pt_image.y + 0.5f),
            imgSize)];
}

/// Sample 4 channel image with bilinear interpolation.
template<typename T> _CPU_AND_GPU_CODE_ inline Vector4f interpolateBilinear(const CONSTPTR(T) *source,
	const THREADPTR(Vector2f) & position, const CONSTPTR(Vector2i) & imgSize)
{
    T a, b, c, d; Vector4f result;
    Vector2i p; Vector2f delta;

    p.x = (int)floor(position.x); p.y = (int)floor(position.y);
    delta.x = position.x - (float)p.x; delta.y = position.y - (float)p.y;

    b.x = 0; b.y = 0; b.z = 0; b.w = 0;
    c.x = 0; c.y = 0; c.z = 0; c.w = 0;
    d.x = 0; d.y = 0; d.z = 0; d.w = 0;

    a = source[p.x + p.y * imgSize.x];
    if (delta.x != 0) b = source[(p.x + 1) + p.y * imgSize.x];
    if (delta.y != 0) c = source[p.x + (p.y + 1) * imgSize.x];
    if (delta.x != 0 && delta.y != 0) d = source[(p.x + 1) + (p.y + 1) * imgSize.x];

    /**
    ------> dx
    | a b
    | c d
    dy
    \/
    */
	result =
        a.toFloat() * (1.0f - delta.x) * (1.0f - delta.y) + 
        b.toFloat() * delta.x * (1.0f - delta.y) +
		c.toFloat() * (1.0f - delta.x) * delta.y + 
        d.toFloat() * delta.x * delta.y;

	return result;
}

/// Same as interpolateBilinear, but return (0,0,0,-1) when any of the four surrounding pixels is illegal (has negative w).
template<typename T> _CPU_AND_GPU_CODE_ inline Vector4f interpolateBilinear_withHoles(const CONSTPTR(T) *source,
	const THREADPTR(Vector2f) & position, const CONSTPTR(Vector2i) & imgSize)
{
	T a, b, c, d; Vector4f result;
	Vector2s p; Vector2f delta;

	p.x = (short)floor(position.x); p.y = (short)floor(position.y);
	delta.x = position.x - (float)p.x; delta.y = position.y - (float)p.y;

	a = source[p.x + p.y * imgSize.x];
	b = source[(p.x + 1) + p.y * imgSize.x];
	c = source[p.x + (p.y + 1) * imgSize.x];
	d = source[(p.x + 1) + (p.y + 1) * imgSize.x];

	if (a.w < 0 || b.w < 0 || c.w < 0 || d.w < 0)
	{
		result.x = 0; result.y = 0; result.z = 0; result.w = -1.0f;
		return result;
	}

	result.x = ((float)a.x * (1.0f - delta.x) * (1.0f - delta.y) + (float)b.x * delta.x * (1.0f - delta.y) +
		(float)c.x * (1.0f - delta.x) * delta.y + (float)d.x * delta.x * delta.y);
	result.y = ((float)a.y * (1.0f - delta.x) * (1.0f - delta.y) + (float)b.y * delta.x * (1.0f - delta.y) +
		(float)c.y * (1.0f - delta.x) * delta.y + (float)d.y * delta.x * delta.y);
	result.z = ((float)a.z * (1.0f - delta.x) * (1.0f - delta.y) + (float)b.z * delta.x * (1.0f - delta.y) +
		(float)c.z * (1.0f - delta.x) * delta.y + (float)d.z * delta.x * delta.y);
	result.w = ((float)a.w * (1.0f - delta.x) * (1.0f - delta.y) + (float)b.w * delta.x * (1.0f - delta.y) +
		(float)c.w * (1.0f - delta.x) * delta.y + (float)d.w * delta.x * delta.y);

	return result;
}

/// For single channel images.
template<typename T> _CPU_AND_GPU_CODE_ inline float interpolateBilinear_withHoles_single(const CONSTPTR(T) *source,
	const THREADPTR(Vector2f) & position, const CONSTPTR(Vector2i) & imgSize)
{
	T a = 0, b = 0, c = 0, d = 0; float result;
	Vector2i p; Vector2f delta;

	p.x = (int)floor(position.x); p.y = (int)floor(position.y);
	delta.x = position.x - (float)p.x; delta.y = position.y - (float)p.y;

	a = source[p.x + p.y * imgSize.x];
	if (delta.x != 0) b = source[(p.x + 1) + p.y * imgSize.x];
	if (delta.y != 0) c = source[p.x + (p.y + 1) * imgSize.x];
	if (delta.x != 0 && delta.y != 0) d = source[(p.x + 1) + (p.y + 1) * imgSize.x];

	if (a < 0 || b < 0 || c < 0 || d < 0) return -1;

	result = ((float)a * (1.0f - delta.x) * (1.0f - delta.y) + (float)b * delta.x * (1.0f - delta.y) +
		(float)c * (1.0f - delta.x) * delta.y + (float)d * delta.x * delta.y);

	return result;
}
