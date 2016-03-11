/**
\file we run a secondary integration after the
swapping and fuse the transferred voxel block data from the long term
storage with the active online data. Effectively we treat the long term
storage simply as a secondary source of information.
*/

// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../Utils/ITMLibDefines.h"

#include "ITMPixelUtils.h"

template<class TVoxel>
_CPU_AND_GPU_CODE_ inline void combineVoxelDepthInformation(const CONSTPTR(TVoxel) & src, DEVICEPTR(TVoxel) & dst, int maxW)
{
	int oldW = src.w_depth;
    if (oldW == 0) return; // Return if there is no color depth sample

	float oldF = TVoxel::SDF_valueToFloat(src.sdf);

	int newW = dst.w_depth;
	float newF = TVoxel::SDF_valueToFloat(dst.sdf);

    updateVoxelDepthInformation(
        dst,
        oldF, oldW, newF, newW, maxW);
}

template<class TVoxel>
_CPU_AND_GPU_CODE_ inline void combineVoxelColorInformation(const CONSTPTR(TVoxel) & src, DEVICEPTR(TVoxel) & dst, int maxW)
{
	int oldW = src.w_color;
	if (oldW == 0) return; // Return if there is no color sample

	Vector3f oldC = src.clr.toFloat();

	int newW = dst.w_color;
	Vector3f newC = dst.clr.toFloat();

    updateVoxelColorInformation(
        dst,
        oldC, oldW, newC, newW, maxW);
}


template<bool hasColor,class TVoxel> struct CombineVoxelInformation;

template<class TVoxel>
struct CombineVoxelInformation<false,TVoxel> {
	_CPU_AND_GPU_CODE_ static void compute(const CONSTPTR(TVoxel) & src, DEVICEPTR(TVoxel) & dst, int maxW)
	{
		combineVoxelDepthInformation(src, dst, maxW);
	}
};

template<class TVoxel>
struct CombineVoxelInformation<true,TVoxel> {
	_CPU_AND_GPU_CODE_ static void compute(const CONSTPTR(TVoxel) & src, DEVICEPTR(TVoxel) & dst, int maxW)
	{
		combineVoxelDepthInformation(src, dst, maxW);
		combineVoxelColorInformation(src, dst, maxW);
	}
};

