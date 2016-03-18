/// \file Implements sparse voxel data structure
// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../Utils/ITMLibDefines.h"
#include "ITMPixelUtils.h"

#define COMPUTE_COEFF_POS_FROM_POINT() \
    /* Coeff are the sub-block coordinates, used for interpolation*/\
    Vector3f coeff; Vector3i pos; TO_INT_FLOOR3(pos, coeff, point);

// -- (i) retrieval of voxel (blocks) --
// === ITMVoxelBlockHash methods (findVoxel) ===

/// Voxel block hash index computation
/// "we employ a hash function to associate 3D block coordinates with entries in a hash table"
template<typename T> _CPU_AND_GPU_CODE_ inline int hashIndex(const THREADPTR(T) & blockPos) {
	return (((uint)blockPos.x * 73856093u) ^ ((uint)blockPos.y * 19349669u) ^ ((uint)blockPos.z * 83492791u)) & (uint)SDF_HASH_MASK;
}

///

/// \returns linearIdx to be used to access individual voxels within a voxel block.
_CPU_AND_GPU_CODE_ inline int pointToVoxelBlockPos(
    const THREADPTR(Vector3i) & point, //!< in voxel coordinates
    THREADPTR(Vector3i) &blockPos //!< [out] In voxel-block coordinates, floor(voxel coordinate / SDF_BLOCK_SIZE)
    ) {
    // "The 3D voxel block location is obtained by dividing the voxel coordinates with the block size along each axis."

    // if SDF_BLOCK_SIZE == 8, then -3 should go to block -1, so we need to adjust negative values 
    // (C's quotient-remainder division gives -3/8 == 0)
	blockPos.x = ((point.x < 0) ? point.x - SDF_BLOCK_SIZE + 1 : point.x) / SDF_BLOCK_SIZE;
	blockPos.y = ((point.y < 0) ? point.y - SDF_BLOCK_SIZE + 1 : point.y) / SDF_BLOCK_SIZE;
	blockPos.z = ((point.z < 0) ? point.z - SDF_BLOCK_SIZE + 1 : point.z) / SDF_BLOCK_SIZE;

    // This works too: [[
	Vector3i locPos = point - blockPos * SDF_BLOCK_SIZE; // localized coordinate
	return locPos.x + locPos.y * SDF_BLOCK_SIZE + locPos.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
    // ]]

    // TODO what does this do? Avoid bank conflicts in cache by reindexing?
/*	return 
        point.x + 
        (point.y - blockPos.x) * SDF_BLOCK_SIZE +
        (point.z - blockPos.y) * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE - blockPos.z * SDF_BLOCK_SIZE3;
        */
}

/// \returns linearIdx to be used in TVoxel* voxelData, voxelData[voxelAddress]
_CPU_AND_GPU_CODE_ inline int findVoxel(
    const CONSTPTR(ITMLib::Objects::ITMVoxelBlockHash::IndexData) *voxelIndex, //<! hash table buckets, indexed by values computed from hashIndex
    const THREADPTR(Vector3i) & point,
	THREADPTR(bool) &isFound, 
    THREADPTR(ITMLib::Objects::ITMVoxelBlockHash::IndexCache) & cache)
{
	Vector3i blockPos;
	short linearIdx = pointToVoxelBlockPos(point, blockPos);

    // Can we find it in the cache?
	if IS_EQUAL3(blockPos, cache.blockPos)
	{
		isFound = true;
		return cache.blockPtr + linearIdx;
	}

    // No, search
	int hashIdx = hashIndex(blockPos);

	while (true) 
	{
		ITMHashEntry hashEntry = voxelIndex[hashIdx];

		if (IS_EQUAL3(hashEntry.pos, blockPos) && hashEntry.isAllocatedAndActive())
		{
			isFound = true;
            // Cache it for later reuse
			cache.blockPos = blockPos;
            cache.blockPtr = hashEntry.ptr * SDF_BLOCK_SIZE3;
			return cache.blockPtr + linearIdx;
		}

        // Excess list ends here?
        if (!hashEntry.hasExcessListOffset()) break;
        // Access next excess list entry
        hashIdx = hashEntry.getHashIndexOfNextExcessEntry();
	}

    // Not found
	isFound = false;
	return -1;
}

_CPU_AND_GPU_CODE_ inline int findVoxel(const CONSTPTR(ITMLib::Objects::ITMVoxelBlockHash::IndexData) *voxelIndex, Vector3i point, THREADPTR(bool) &isFound)
{
	ITMLib::Objects::ITMVoxelBlockHash::IndexCache cache;
	return findVoxel(voxelIndex, point, isFound, cache);
}



/// === ITMBlockhash methods (readVoxel) ===
template<class TVoxel>
_CPU_AND_GPU_CODE_ inline TVoxel readVoxel(const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(ITMLib::Objects::ITMVoxelBlockHash::IndexData) *voxelIndex,
	const THREADPTR(Vector3i) & point, THREADPTR(bool) &isFound, THREADPTR(ITMLib::Objects::ITMVoxelBlockHash::IndexCache) & cache)
{
    int voxelAddress = findVoxel(voxelIndex, point, isFound, cache);
    return isFound ? voxelData[voxelAddress] : TVoxel();
}

template<class TVoxel>
_CPU_AND_GPU_CODE_ inline TVoxel readVoxel(const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(ITMLib::Objects::ITMVoxelBlockHash::IndexData) *voxelIndex,
	Vector3i point, THREADPTR(bool) &isFound)
{
	ITMLib::Objects::ITMVoxelBlockHash::IndexCache cache;
	return readVoxel(voxelData, voxelIndex, point, isFound, cache);
}


/// === Generic methods (readSDF) ===
template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline float readFromSDF_float_uninterpolated(
    const CONSTPTR(TVoxel) *voxelData,
	const CONSTPTR(TIndex) *voxelIndex, 
    Vector3f point,  //!< in voxel-fractional-world-coordinates (such that one voxel has size 1)
    THREADPTR(bool) &isFound)
{
	TVoxel res = readVoxel(voxelData, voxelIndex, Vector3i((int)ROUND(point.x), (int)ROUND(point.y), (int)ROUND(point.z)), isFound);
	return TVoxel::SDF_valueToFloat(res.sdf);
}

template<class TVoxel, class TIndex, class TCache>
_CPU_AND_GPU_CODE_ inline float readFromSDF_float_uninterpolated(
    const CONSTPTR(TVoxel) *voxelData,
	const CONSTPTR(TIndex) *voxelIndex, 
    Vector3f point, //!< in voxel-fractional-world-coordinates (such that one voxel has size 1)
    THREADPTR(bool) &isFound, THREADPTR(TCache) & cache)
{
	TVoxel res = readVoxel(voxelData, voxelIndex, Vector3i((int)ROUND(point.x), (int)ROUND(point.y), (int)ROUND(point.z)), isFound, cache);
	return TVoxel::SDF_valueToFloat(res.sdf);
}

template<class TVoxel, class TIndex, class TCache>
_CPU_AND_GPU_CODE_ inline float readFromSDF_float_interpolated(
    const CONSTPTR(TVoxel) *voxelData,
	const CONSTPTR(TIndex) *voxelIndex,
    Vector3f point, //!< in voxel-fractional-world-coordinates (such that one voxel has size 1)
    THREADPTR(bool) &isFound, 
    THREADPTR(TCache) & cache)
{
	float res1, res2, v1, v2;
    COMPUTE_COEFF_POS_FROM_POINT();

    // z = 0 layer -> res1
	v1 = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 0, 0), isFound, cache).sdf;
	v2 = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 0, 0), isFound, cache).sdf;
	res1 = (1.0f - coeff.x) * v1 + coeff.x * v2;

	v1 = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 1, 0), isFound, cache).sdf;
	v2 = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 1, 0), isFound, cache).sdf;
	res1 = (1.0f - coeff.y) * res1 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

    // z = 1 layer -> res2
	v1 = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 0, 1), isFound, cache).sdf;
	v2 = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 0, 1), isFound, cache).sdf;
	res2 = (1.0f - coeff.x) * v1 + coeff.x * v2;

	v1 = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 1, 1), isFound, cache).sdf;
	v2 = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 1, 1), isFound, cache).sdf;
	res2 = (1.0f - coeff.y) * res2 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

	isFound = true;
	return TVoxel::SDF_valueToFloat((1.0f - coeff.z) * res1 + coeff.z * res2);
}

/// Assumes voxels store color in some type convertible to Vector3f (e.g. Vector3u) and that color values are in the range 0-255.
/// \returns [0,1]^4 color (with w = 1)
template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline Vector4f readFromSDF_color4u_interpolated(const CONSTPTR(TVoxel) *voxelData,
	const CONSTPTR(typename TIndex::IndexData) *voxelIndex, const THREADPTR(Vector3f) & point, 
	THREADPTR(typename TIndex::IndexCache) & cache)
{
    TVoxel resn;
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

    return Vector4f(ret, 255.0f) / 255.0f;
}

/// Compute SDF normal 
/// Used in processPixelGrey
template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline Vector3f computeSingleNormalFromSDF(const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(TIndex) *voxelIndex, const THREADPTR(Vector3f) &point)
{

	Vector3f ret;
    COMPUTE_COEFF_POS_FROM_POINT();
    Vector3f ncoeff = Vector3f(1,1,1) - coeff;

    bool isFound;
#define lookup(dx,dy,dz) readVoxel(voxelData, voxelIndex, pos + Vector3i(dx,dy,dz), isFound).sdf

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

	ret.x = TVoxel::SDF_valueToFloat(
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

	ret.y = TVoxel::SDF_valueToFloat(p1 * ncoeff.y + p2 * coeff.y - v1);

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

	ret.z = TVoxel::SDF_valueToFloat(p1 * ncoeff.z + p2 * coeff.z - v1);
#undef lookup
	return ret;
}

template<bool hasColor,class TVoxel,class TIndex> struct VoxelColorReader;

template<class TVoxel, class TIndex>
struct VoxelColorReader<false,TVoxel,TIndex> {
	_CPU_AND_GPU_CODE_ static Vector4f interpolate(const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(typename TIndex::IndexData) *voxelIndex,
		const THREADPTR(Vector3f) & point)
	{ return Vector4f(0.0f,0.0f,0.0f,0.0f); }
};

template<class TVoxel, class TIndex>
struct VoxelColorReader<true,TVoxel,TIndex> {
	_CPU_AND_GPU_CODE_ static Vector4f interpolate(const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(typename TIndex::IndexData) *voxelIndex,
		const THREADPTR(Vector3f) & point)
	{
		typename TIndex::IndexCache cache;
		return readFromSDF_color4u_interpolated<TVoxel,TIndex>(voxelData, voxelIndex, point, cache);
	}
};

#undef COMPUTE_COEFF_POS_FROM_POINT