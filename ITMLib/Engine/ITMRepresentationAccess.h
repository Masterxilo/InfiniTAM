/// \file Implements sparse voxel data structure
// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMLibDefines.h"
#include "ITMPixelUtils.h"
#include "Scene.h"


/// === ITMBlockhash methods (readVoxel) ===
GPU_ONLY inline ITMVoxel readVoxel(
    const CONSTPTR(ITMVoxelBlock) * const voxelData,
    const CONSTPTR(ITMHashEntry) * const voxelIndex,
	const THREADPTR(Vector3i) & point,
    THREADPTR(bool) &isFound,
    THREADPTR(ITMLib::Objects::ITMVoxelBlockHash::IndexCache) & cache = ITMVoxelBlockHash::IndexCache())
{
    ITMVoxel* v = Scene::getCurrentSceneVoxel(point);
    if (!v) {
        isFound = false;
        return;
    }
    isFound = true;
    return *v;
}


/// === Generic methods (readSDF) ===
GPU_ONLY inline float readFromSDF_float_uninterpolated(
    const CONSTPTR(ITMVoxelBlock) * const voxelData,
    const CONSTPTR(ITMLib::Objects::ITMVoxelBlockHash::IndexData) *voxelIndex,
    Vector3f point, //!< in voxel-fractional-world-coordinates (such that one voxel has size 1)
    THREADPTR(bool) &isFound,
    THREADPTR(ITMLib::Objects::ITMVoxelBlockHash::IndexCache) & cache = ITMVoxelBlockHash::IndexCache())
{
    ITMVoxel res = readVoxel(voxelData, voxelIndex, Vector3i((int)ROUND(point.x), (int)ROUND(point.y), (int)ROUND(point.z)), isFound, cache);
    return res.getSDF();
}

#define COMPUTE_COEFF_POS_FROM_POINT() \
    /* Coeff are the sub-block coordinates, used for interpolation*/\
    Vector3f coeff; Vector3i pos; TO_INT_FLOOR3(pos, coeff, point);

GPU_ONLY inline float readFromSDF_float_interpolated(
    const CONSTPTR(ITMVoxelBlock) * const voxelData,
    const CONSTPTR(ITMLib::Objects::ITMVoxelBlockHash::IndexData) *voxelIndex,
    Vector3f point, //!< in voxel-fractional-world-coordinates (such that one voxel has size 1)
    THREADPTR(bool) &isFound, 
    THREADPTR(ITMLib::Objects::ITMVoxelBlockHash::IndexCache) & cache = ITMVoxelBlockHash::IndexCache())
{
	float res1, res2, v1, v2;
    COMPUTE_COEFF_POS_FROM_POINT();

    // z = 0 layer -> res1
	v1 = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 0, 0), isFound, cache).getSDF();
	v2 = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 0, 0), isFound, cache).getSDF();
	res1 = (1.0f - coeff.x) * v1 + coeff.x * v2;

	v1 = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 1, 0), isFound, cache).getSDF();
	v2 = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 1, 0), isFound, cache).getSDF();
	res1 = (1.0f - coeff.y) * res1 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

    // z = 1 layer -> res2
	v1 = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 0, 1), isFound, cache).getSDF();
	v2 = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 0, 1), isFound, cache).getSDF();
	res2 = (1.0f - coeff.x) * v1 + coeff.x * v2;

	v1 = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 1, 1), isFound, cache).getSDF();
	v2 = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 1, 1), isFound, cache).getSDF();
	res2 = (1.0f - coeff.y) * res2 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

	isFound = true;
    return (1.0f - coeff.z) * res1 + coeff.z * res2;
}

/// Assumes voxels store color in some type convertible to Vector3f (e.g. Vector3u)
GPU_ONLY inline Vector3f readFromSDF_color4u_interpolated(
    const CONSTPTR(ITMVoxelBlock) * const voxelData,
    const CONSTPTR(typename ITMVoxelBlockHash::IndexData) *voxelIndex, 
    const THREADPTR(Vector3f) & point, //!< in voxel-fractional world coordinates, comes e.g. from raycastResult
    THREADPTR(typename ITMVoxelBlockHash::IndexCache) & cache = ITMVoxelBlockHash::IndexCache())
{
    ITMVoxel resn; 
    Vector3f ret = 0.0f; 
    bool isFound;
    
    COMPUTE_COEFF_POS_FROM_POINT()

#define access(dx,dy,dz) \
    resn = readVoxel(voxelData, voxelIndex, pos + Vector3i(dx, dy, dz), isFound, cache);\
    ret += \
    (dx ? coeff.x : 1.0f - coeff.x) *\
    (dy ? coeff.y : 1.0f - coeff.y) *\
    (dz ? coeff.z : 1.0f - coeff.z) *\
    resn.clr.toFloat();
    
    access(0, 0, 0);
    access(0, 0, 1);
    access(0, 1, 0);
    access(0, 1, 1);
    access(1, 0, 0);
    access(1, 0, 1);
    access(1, 1, 0);
    access(1, 1, 1);

    return ret;
}

#define lookup(dx,dy,dz) readVoxel(voxelData, voxelIndex, pos + Vector3i(dx,dy,dz), isFound).getSDF()

GPU_ONLY inline Vector3f computeSingleNormalFromSDFByForwardDifference(
    const CONSTPTR(ITMVoxelBlock) * const voxelData,
    const CONSTPTR(typename ITMVoxelBlockHash::IndexData) *voxelIndex,

    const THREADPTR(Vector3i) &pos, //!< [in] global voxel position
    bool& isFound //!< [out] whether all values needed existed;
    ) {
    float sdf0 = lookup(0,0,0);
    if (!isFound) return Vector3f();

    // TODO handle !isFound
    Vector3f n(
        lookup(1, 0, 0) - sdf0,
        lookup(0, 1, 0) - sdf0,
        lookup(0, 0, 1) - sdf0
        );
    return n.normalised(); // TODO in a distance field, normalization should not be necesary. But this is not a true distance field.
}

/// Compute SDF normal 
/// Used in processPixelGrey
// Note: this gets the localVBA list, not just a *single* voxel block.
GPU_ONLY inline Vector3f computeSingleNormalFromSDF(
    const CONSTPTR(ITMVoxelBlock) * const voxelData,
    const CONSTPTR(typename ITMVoxelBlockHash::IndexData) *voxelIndex,
    const THREADPTR(Vector3f) &point)
{

	Vector3f ret;
    COMPUTE_COEFF_POS_FROM_POINT();
    Vector3f ncoeff = Vector3f(1,1,1) - coeff;

    bool isFound; // swallow

    /*
    x direction gradient at point is evaluated by computing interpolated sdf value in next (1 -- 2, v2) and previous (-1 -- 0, v1) cell:

    -1  0   1   2
    *---*---*---*
    |v1 |   | v2|
    *---*---*---*

    0 z is called front, 1 z is called back

    gradient is then
    v2 - v1
    */

    /* using xyzw components of vector4f to store 4 sdf values as follows:
    
    *---0--1-> x
    |
    0   x--y
    |   |  |
    1   z--w
    \/
    y
    */

	// all 8 values are going to be reused several times
	Vector4f front, back;
    front.x = lookup(0, 0, 0);
	front.y = lookup(1, 0, 0);
	front.z = lookup(0, 1, 0);
	front.w = lookup(1, 1, 0);
	back.x  = lookup(0, 0, 1);
	back.y  = lookup(1, 0, 1);
	back.z  = lookup(0, 1, 1);
	back.w  = lookup(1, 1, 1);

	Vector4f tmp;
	float p1, p2, v1;
	// gradient x
    // v1
    // 0-layer
	p1 = front.x * ncoeff.y * ncoeff.z +
	     front.z *  coeff.y * ncoeff.z +
	     back.x  * ncoeff.y *  coeff.z +
         back.z  *  coeff.y *  coeff.z;
    // (-1)-layer
	tmp.x = lookup(-1, 0, 0);
	tmp.y = lookup(-1, 1, 0);
	tmp.z = lookup(-1, 0, 1);
    tmp.w = lookup(-1, 1, 1);
	p2 = tmp.x * ncoeff.y * ncoeff.z +
	     tmp.y *  coeff.y * ncoeff.z +
	     tmp.z * ncoeff.y *  coeff.z +
	     tmp.w *  coeff.y *  coeff.z;

	v1 = p1 * coeff.x + p2 * ncoeff.x;

    // v2
    // 1-layer
	p1 = front.y * ncoeff.y * ncoeff.z +
	     front.w *  coeff.y * ncoeff.z +
	     back.y  * ncoeff.y *  coeff.z +
         back.w  *  coeff.y *  coeff.z;
    // 2-layer
	tmp.x = lookup(2, 0, 0);
	tmp.y = lookup(2, 1, 0);
	tmp.z = lookup(2, 0, 1);
	tmp.w = lookup(2, 1, 1);
	p2 = tmp.x * ncoeff.y * ncoeff.z +
	     tmp.y *  coeff.y * ncoeff.z +
	     tmp.z * ncoeff.y *  coeff.z +
	     tmp.w *  coeff.y *  coeff.z;

    ret.x = (
        p1 * ncoeff.x + p2 * coeff.x // v2
        - 
        v1);

	// gradient y
	p1 = front.x * ncoeff.x * ncoeff.z +
	     front.y *  coeff.x * ncoeff.z +
	     back.x  * ncoeff.x *  coeff.z +
	     back.y  *  coeff.x *  coeff.z;
	tmp.x = lookup(0, -1, 0);
	tmp.y = lookup(1, -1, 0);
	tmp.z = lookup(0, -1, 1);
	tmp.w = lookup(1, -1, 1);
	p2 = tmp.x * ncoeff.x * ncoeff.z +
	     tmp.y *  coeff.x * ncoeff.z +
	     tmp.z * ncoeff.x *  coeff.z +
	     tmp.w *  coeff.x *  coeff.z;
	v1 = p1 * coeff.y + p2 * ncoeff.y;

	p1 = front.z * ncoeff.x * ncoeff.z +
	     front.w *  coeff.x * ncoeff.z +
	     back.z  * ncoeff.x *  coeff.z +
	     back.w  *  coeff.x *  coeff.z;
	tmp.x = lookup(0, 2, 0);
	tmp.y = lookup(1, 2, 0);
	tmp.z = lookup(0, 2, 1);
	tmp.w = lookup(1, 2, 1);
	p2 = tmp.x * ncoeff.x * ncoeff.z +
	     tmp.y *  coeff.x * ncoeff.z +
	     tmp.z * ncoeff.x *  coeff.z +
	     tmp.w *  coeff.x *  coeff.z;

    ret.y = (p1 * ncoeff.y + p2 * coeff.y - v1);

	// gradient z
	p1 = front.x * ncoeff.x * ncoeff.y +
	     front.y *  coeff.x * ncoeff.y +
	     front.z * ncoeff.x *  coeff.y +
	     front.w *  coeff.x *  coeff.y;
	tmp.x = lookup(0, 0, -1);
	tmp.y = lookup(1, 0, -1);
	tmp.z = lookup(0, 1, -1);
	tmp.w = lookup(1, 1, -1);
	p2 = tmp.x * ncoeff.x * ncoeff.y +
	     tmp.y *  coeff.x * ncoeff.y +
	     tmp.z * ncoeff.x *  coeff.y +
	     tmp.w *  coeff.x *  coeff.y;
	v1 = p1 * coeff.z + p2 * ncoeff.z;

	p1 = back.x * ncoeff.x * ncoeff.y +
	     back.y *  coeff.x * ncoeff.y +
	     back.z * ncoeff.x *  coeff.y +
	     back.w *  coeff.x *  coeff.y;
	tmp.x = lookup(0, 0, 2);
	tmp.y = lookup(1, 0, 2);
	tmp.z = lookup(0, 1, 2);
	tmp.w = lookup(1, 1, 2);
	p2 = tmp.x * ncoeff.x * ncoeff.y +
	     tmp.y *  coeff.x * ncoeff.y +
	     tmp.z * ncoeff.x *  coeff.y +
	     tmp.w *  coeff.x *  coeff.y;

    ret.z = (p1 * ncoeff.z + p2 * coeff.z - v1);
#undef lookup
	return ret;
}

#undef COMPUTE_COEFF_POS_FROM_POINT
