// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include <math.h>
#include <ostream>

#include "../../Utils/ITMLibDefines.h"

#define weightedCombine(oldX, oldW, newX, newW) \
    newX = (float)oldW * oldX + (float)newW * newX; \
    newW = oldW + newW;\
    newX /= (float)newW;\
    newW = MIN(newW, maxW);

/// Linearized pixel index
_CPU_AND_GPU_CODE_ inline int pixelLocId(int x, int y, const THREADPTR(Vector2i) &imgSize) {
    return x + y * imgSize.x;
}

_CPU_AND_GPU_CODE_ inline void updateVoxelColorInformation(
    DEVICEPTR(ITMVoxel) & voxel,
    Vector3f oldC, int oldW, Vector3f newC, int newW,
    int maxW)
{
    weightedCombine(oldC, oldW, newC, newW);

    // write back
    /// C(X) <-  
    voxel.clr = TO_UCHAR3(newC);
    voxel.w_color = (uchar)newW;
}

_CPU_AND_GPU_CODE_ inline void updateVoxelDepthInformation(
    DEVICEPTR(ITMVoxel) & voxel,
    float oldF, int oldW, float newF, int newW,
    int maxW)
{
    weightedCombine(oldF, oldW, newF, newW);

    // write back
    /// D(X) <-  (4)
    voxel.sdf = ITMVoxel::SDF_floatToValue(newF);
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

template<typename T>
struct IllegalColor {
    static _CPU_AND_GPU_CODE_ T make();
};
inline _CPU_AND_GPU_CODE_ float IllegalColor<float>::make() {
    return -1;
}
inline _CPU_AND_GPU_CODE_ Vector4f IllegalColor<Vector4f>::make() {
    return Vector4f(0, 0, 0, -1);
}
inline _CPU_AND_GPU_CODE_ bool isLegalColor(float c) {
    return c >= 0;
}
inline _CPU_AND_GPU_CODE_ bool isLegalColor(Vector4f c) {
    return c.w >= 0;
}
inline _CPU_AND_GPU_CODE_ bool isLegalColor(Vector4u c) {
    // NOTE this should never be called -- withHoles should be false for a Vector4u
    // implementing this just calms the compiler
    assert(false);
    return false;
}
inline _CPU_AND_GPU_CODE_ Vector4f toFloat(Vector4u c) {
    return c.toFloat();
}

inline _CPU_AND_GPU_CODE_ Vector4f toFloat(Vector4f c) {
    return c;
}
inline _CPU_AND_GPU_CODE_ float toFloat(float c) {
    return c;
}

#define WITH_HOLES true
/// Sample 4 channel image with bilinear interpolation (T::toFloat must return Vector4f)
/// withHoles: returns makeIllegalColor<OUT>() when any of the four surrounding pixels is illegal (has negative w).
template<typename T_OUT, //!< Vector4f or float
    bool withHoles = false, typename T_IN> _CPU_AND_GPU_CODE_ inline Vector4f interpolateBilinear(
    const CONSTPTR(T_IN) *source,
	const THREADPTR(Vector2f) & position, const CONSTPTR(Vector2i) & imgSize)
{
    T_OUT result;
    Vector2i p; Vector2f delta;

    p.x = (int)floor(position.x); p.y = (int)floor(position.y);
    delta.x = position.x - p.x; delta.y = position.y - p.y;

#define sample(dx, dy) sampleNearest(source, p.x + dx, p.y + dy, imgSize);
    T_IN a = sample(0, 0);
    T_IN b = sample(1, 0);
    T_IN c = sample(0, 1);
    T_IN d = sample(1, 1);
#undef sample

    if (withHoles && (!isLegalColor(a) || !isLegalColor(b) || !isLegalColor(c) || !isLegalColor(d))) return IllegalColor<T_OUT>::make();

    /**
    ------> dx
    | a b
    | c d
    dy
    \/
    */
	result =
        toFloat(a) * (1.0f - delta.x) * (1.0f - delta.y) +
        toFloat(b) * delta.x * (1.0f - delta.y) +
        toFloat(c) * (1.0f - delta.x) * delta.y +
        toFloat(d) * delta.x * delta.y;

	return result;
}
