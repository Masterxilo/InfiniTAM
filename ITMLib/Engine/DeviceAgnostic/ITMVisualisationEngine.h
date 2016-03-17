///\file RENDERING STAGE, Raycasting
// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMPixelUtils.h"

#include "../DeviceAgnostic/ITMRepresentationAccess.h"
#include "../../Utils/ITMLibDefines.h"

struct RenderingBlock {
	Vector2s upperLeft;
	Vector2s lowerRight;
	Vector2f zRange;
};

#ifndef FAR_AWAY
#define FAR_AWAY 999999.9f
#endif

#ifndef VERY_CLOSE
#define VERY_CLOSE 0.05f
#endif

static const CONSTPTR(int) renderingBlockSizeX = 16;
static const CONSTPTR(int) renderingBlockSizeY = 16;

static const CONSTPTR(int) MAX_RENDERING_BLOCKS = 65536*4;
//static const int MAX_RENDERING_BLOCKS = 16384;
static const CONSTPTR(int) minmaximg_subsample = 8;

/**
Project visible blocks into the desired image.

Compute the bounding box (upperLeft, lowerRight, zRange) of the
projection of all eight corners in image space and store the minimum
and maximum Z coordinates of the block in the camera coordinate
system
*/
_CPU_AND_GPU_CODE_ inline bool ProjectSingleBlock(
    const THREADPTR(Vector3s) & blockPos, 
    const THREADPTR(Matrix4f) & pose, 
    const THREADPTR(Vector4f) & intrinsics, 
	const THREADPTR(Vector2i) & imgSize,
    float voxelSize, 
    THREADPTR(Vector2i) & upperLeft, //!<
    THREADPTR(Vector2i) & lowerRight,  //!<
    THREADPTR(Vector2f) & zRange //!<
    )
{
    Vector2i minmaxImgSize = imgSize / minmaximg_subsample;
    upperLeft = minmaxImgSize;
	lowerRight = Vector2i(-1, -1);
    // zMin, zmax
	zRange = Vector2f(FAR_AWAY, VERY_CLOSE);

    // project all 8 corners down to 2D image
	for (int corner = 0; corner < 8; ++corner)
	{
		Vector3s tmp = blockPos;
		tmp.x += (corner & 1) ? 1 : 0;
		tmp.y += (corner & 2) ? 1 : 0;
		tmp.z += (corner & 4) ? 1 : 0;
		Vector4f pt3d(TO_FLOAT3(tmp) * (float)SDF_BLOCK_SIZE * voxelSize, 1.0f);
		pt3d = pose * pt3d;

		Vector2f pt2d;
        if (!projectNoBounds(intrinsics, pt3d, pt2d)) continue;
        pt2d /= minmaximg_subsample;

		// remember bounding box, zmin and zmax
		if (upperLeft.x > floor(pt2d.x)) upperLeft.x = (int)floor(pt2d.x);
		if (lowerRight.x < ceil(pt2d.x)) lowerRight.x = (int)ceil(pt2d.x);
		if (upperLeft.y > floor(pt2d.y)) upperLeft.y = (int)floor(pt2d.y);
		if (lowerRight.y < ceil(pt2d.y)) lowerRight.y = (int)ceil(pt2d.y);
		if (zRange.x > pt3d.z) zRange.x = pt3d.z;
		if (zRange.y < pt3d.z) zRange.y = pt3d.z;
	}

	// do some sanity checks and respect image bounds
	if (upperLeft.x < 0) upperLeft.x = 0;
	if (upperLeft.y < 0) upperLeft.y = 0;
    if (lowerRight.x >= minmaxImgSize.x) lowerRight.x = minmaxImgSize.x - 1;
    if (lowerRight.y >= minmaxImgSize.y) lowerRight.y = minmaxImgSize.y - 1;
	if (upperLeft.x > lowerRight.x) return false;
	if (upperLeft.y > lowerRight.y) return false;
	//if (zRange.y <= VERY_CLOSE) return false; never seems to happen
	if (zRange.x < VERY_CLOSE) zRange.x = VERY_CLOSE;
	if (zRange.y < VERY_CLOSE) return false;

	return true;
}

/**
Split image-depth space bounding box described by (upperLeft, lowerRight, zRange)
into (renderingBlockSizeX by renderingBlockSizeY) pixel (or less) RenderingBlocks of same zRange.

Store the resulting blocks into renderingBlockList, incrementing the current position 'offset' in this list.
*/
_CPU_AND_GPU_CODE_ inline void CreateRenderingBlocks(
    DEVICEPTR(RenderingBlock) *renderingBlockList, int offset,

	const THREADPTR(Vector2i) & upperLeft,
    const THREADPTR(Vector2i) & lowerRight,
    const THREADPTR(Vector2f) & zRange)
{
	for (int by = 0; by < ceil((float)(1 + lowerRight.y - upperLeft.y) / renderingBlockSizeY); ++by) {
		for (int bx = 0; bx < ceil((float)(1 + lowerRight.x - upperLeft.x) / renderingBlockSizeX); ++bx) {
            // End if list is full.
			if (offset >= MAX_RENDERING_BLOCKS) return;
			//for each rendering block: add it to the list
			DEVICEPTR(RenderingBlock) & b(renderingBlockList[offset++]);

			b.upperLeft.x = upperLeft.x + bx*renderingBlockSizeX;
			b.upperLeft.y = upperLeft.y + by*renderingBlockSizeY;

            // lowerRight corner
			b.lowerRight.x = upperLeft.x + (bx + 1)*renderingBlockSizeX - 1;
			b.lowerRight.y = upperLeft.y + (by + 1)*renderingBlockSizeY - 1;

            // Stay within image bounds (renderingBlockSizeX, renderingBlockSizeY) might not fit
			if (b.lowerRight.x>lowerRight.x) b.lowerRight.x = lowerRight.x;
			if (b.lowerRight.y>lowerRight.y) b.lowerRight.y = lowerRight.y;

			b.zRange = zRange;
		}
	}
}

/// \returns whether any intersection was found
template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline bool castRay(
    DEVICEPTR(Vector4f) &pt_out, //<! the intersection point. w is 1 for a valid point, 0 for no intersection; in voxel-fractional-world-coordinates
    int x, int y,
    const CONSTPTR(TVoxel) *voxelData,
	const CONSTPTR(typename TIndex::IndexData) *voxelIndex,
    Matrix4f invM, //!< camera-to-world transform
    Vector4f invProjParams, //!< camera-to-world transform
    float oneOverVoxelSize, 
	float mu, const CONSTPTR(Vector2f) & viewFrustum_minmax)
{
	Vector4f pt_camera_f;
    Vector3f pt_block_s, pt_block_e;
	
	float totalLength, stepLength, totalLengthMax;


    // Starting point
    pt_camera_f = depthTo3DInvProjParams(invProjParams, x, y, viewFrustum_minmax.x);
    // Lengths given in voxel-fractional-coordinates (such that one voxel has size 1)
	totalLength = length(TO_VECTOR3(pt_camera_f)) * oneOverVoxelSize;
    // in voxel-fractional-world-coordinates (such that one voxel has size 1)
    pt_block_s = TO_VECTOR3(invM * pt_camera_f) * oneOverVoxelSize;

    // End point
    pt_camera_f = depthTo3DInvProjParams(invProjParams, x, y, viewFrustum_minmax.y);
	totalLengthMax = length(TO_VECTOR3(pt_camera_f)) * oneOverVoxelSize;
	pt_block_e = TO_VECTOR3(invM * pt_camera_f) * oneOverVoxelSize;


    // Raymarching
    const Vector3f rayDirection = normalize(pt_block_e - pt_block_s);
    Vector3f pt_result = pt_block_s; // Current position in voxel-fractional-world-coordinates
    const float stepScale = mu * oneOverVoxelSize;
	typename TIndex::IndexCache cache;
    float sdfValue = 1.0f;
    bool hash_found;
    while (totalLength < totalLengthMax) {
        // D(X)
		sdfValue = readFromSDF_float_uninterpolated(voxelData, voxelIndex, pt_result, hash_found, cache);

        if (!hash_found) {
            //  First we try to find an allocated voxel block, and the length of the steps we take is determined by the block size
			stepLength = SDF_BLOCK_SIZE;
		} else {
            // If we found an allocated block, 
            // [Once we are inside the truncation band], the values from the SDF give us conservative step lengths.

            // using trilinear interpolation only if we have read values in the range −0.5 ≤ D(X) ≤ 0.1
			if ((sdfValue <= 0.1f) && (sdfValue >= -0.5f)) {
				sdfValue = readFromSDF_float_interpolated(voxelData, voxelIndex, pt_result, hash_found, cache);
			}
            // once we read a negative value from the SDF, we found the intersection with the surface.
			if (sdfValue <= 0.0f) break;

			stepLength = MAX(
                sdfValue * stepScale, 
                1.0f // if we are outside the truncation band µ, our step size is determined by the truncation band 
                // (note that the distance is normalized to lie in [-1,1] within the truncation band)
                );
		}

		pt_result += stepLength * rayDirection;
        totalLength += stepLength;
	}

    bool pt_found;
    //  If the T - SDF value is negative after such a trilinear interpolation, the surface
    //  has indeed been found and we terminate the ray, performing one last
    //  trilinear interpolation step for a smoother appearance.
	if (sdfValue <= 0.0f)
	{
        // Refine position
		stepLength = sdfValue * stepScale;
		pt_result += stepLength * rayDirection;

        // Read again
		sdfValue = readFromSDF_float_interpolated(voxelData, voxelIndex, pt_result, hash_found, cache);
        // Refine position
        stepLength = sdfValue * stepScale;
		pt_result += stepLength * rayDirection;

		pt_found = true;
	} else pt_found = false;

    pt_out = Vector4f(pt_result, (pt_found) ? 1.0f : 0.0f);

	return pt_found;
}

/// \returns -1 on failure, otherwise the linear index of the pixel the given world 
/// coordinate 'pixel' projects to
_CPU_AND_GPU_CODE_ inline int forwardProjectPixel(
    Vector4f pixel, const CONSTPTR(Matrix4f) &M,
    const CONSTPTR(Vector4f) &projParams,
	const THREADPTR(Vector2i) &imgSize)
{
	pixel.w = 1;

	Vector2f pt_image;
    if (!projectModel(projParams, M, imgSize, pixel, pixel, pt_image)) return -1;

	return (int)(pt_image.x + 0.5f) + (int)(pt_image.y + 0.5f) * imgSize.x;
}

/// Compute normal in the distance field via the gradient.
/// c.f. computeSingleNormalFromSDF
template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void computeNormalAndAngle(
    THREADPTR(bool) & foundPoint, //!< in,out
    const THREADPTR(Vector3f) & point,
    const CONSTPTR(TVoxel) *voxelBlockData,
    const CONSTPTR(typename TIndex::IndexData) *indexData,
    const THREADPTR(Vector3f) & lightSource,
    THREADPTR(Vector3f) & outNormal,
    THREADPTR(float) & angle //!< outNormal . lightSource
    )
{
	if (!foundPoint) return;

	outNormal = normalize(computeSingleNormalFromSDF(voxelBlockData, indexData, point));

    angle = dot(outNormal, lightSource);
	if (!(angle > 0.0)) foundPoint = false;
}

/**
Computing the surface normal in image space given raycasted image.

In image space, since the normals are computed on a regular grid,
there are only 4 uninterpolated read operations followed by a cross-product.

\returns normal_out[idx].w = sigmaZ_out[idx] = -1 on error where idx = x + y * imgDims.x
*/
template <bool useSmoothing>
_CPU_AND_GPU_CODE_ inline void computeNormalAndAngle(
    THREADPTR(bool) & foundPoint, //!< in,out. Set to false when the normal cannot be computed
    const THREADPTR(int) &x, const THREADPTR(int) &y,
	const CONSTPTR(Vector4f) *pointsRay,
    const THREADPTR(Vector3f) & lightSource,
    const THREADPTR(float) &voxelSize,
	const THREADPTR(Vector2i) &imgSize,
    THREADPTR(Vector3f) & outNormal,
    THREADPTR(float) & angle //!< outNormal . lightSource
    )
{
	if (!foundPoint) return;

    // Lookup world coordinates of points surrounding (x,y)
    // and compute forward difference vectors
    Vector4f xp1_y, xm1_y, x_yp1, x_ym1;
    Vector4f diff_x(0.0f, 0.0f, 0.0f, 0.0f), diff_y(0.0f, 0.0f, 0.0f, 0.0f);

    // If useSmoothing, use positions 2 away
    int extraDelta = useSmoothing ? 1 : 0;

#define d(x) (x + extraDelta)

    if (y <= d(1) || y >= imgSize.y - d(2) || x <= d(1) || x >= imgSize.x - d(2)) { foundPoint = false; return; }

#define lookupNeighbors() \
    xp1_y = sampleNearest(pointsRay, x + d(1), y, imgSize);\
    x_yp1 = sampleNearest(pointsRay, x, y + d(1), imgSize);\
    xm1_y = sampleNearest(pointsRay, x - d(1), y, imgSize);\
    x_ym1 = sampleNearest(pointsRay, x, y - d(1), imgSize);\
    diff_x = xp1_y - xm1_y;\
    diff_y = x_yp1 - x_ym1;

    lookupNeighbors();

#define isAnyPointIllegal() (xp1_y.w <= 0 || x_yp1.w <= 0 || xm1_y.w <= 0 || x_ym1.w <= 0)

    float length_diff = MAX(length2(diff_x.toVector3()), length2(diff_y.toVector3()));
    bool lengthDiffTooLarge = (length_diff * voxelSize * voxelSize > (0.15f * 0.15f));

    if (isAnyPointIllegal() || (lengthDiffTooLarge && useSmoothing)) {
        if (!useSmoothing) { foundPoint = false; return; }
       
        // In case we used smoothing, try again without extra delta 
        extraDelta = 0;
        lookupNeighbors();

        if (isAnyPointIllegal()){ foundPoint = false; return; }
    }

#undef d
#undef isAnyPointIllegal
#undef lookupNeighbors

    // TODO why the extra minus?
    outNormal = normalize(-cross(diff_x.toVector3(), diff_y.toVector3()));

    angle = dot(outNormal, lightSource);
	if (!(angle > 0.0)) foundPoint = false;
}

_CPU_AND_GPU_CODE_ inline void drawPixelGrey(DEVICEPTR(Vector4u) & dest, const THREADPTR(float) & angle)
{
	float outRes = (0.8f * angle + 0.2f) * 255.0f;
	dest = Vector4u((uchar)outRes);
}

_CPU_AND_GPU_CODE_ inline void drawPixelNormal(DEVICEPTR(Vector4u) & dest, const THREADPTR(Vector3f) & normal_obj)
{
	dest.r = (uchar)((0.3f + (-normal_obj.r + 1.0f)*0.35f)*255.0f);
	dest.g = (uchar)((0.3f + (-normal_obj.g + 1.0f)*0.35f)*255.0f);
	dest.b = (uchar)((0.3f + (-normal_obj.b + 1.0f)*0.35f)*255.0f);
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void drawPixelColour(DEVICEPTR(Vector4u) & dest, const CONSTPTR(Vector3f) & point, 
	const CONSTPTR(TVoxel) *voxelBlockData, const CONSTPTR(typename TIndex::IndexData) *indexData)
{
	Vector4f clr = VoxelColorReader<TVoxel::hasColorInformation, TVoxel, TIndex>::interpolate(voxelBlockData, indexData, point);

	dest.x = (uchar)(clr.x * 255.0f);
	dest.y = (uchar)(clr.y * 255.0f);
	dest.z = (uchar)(clr.z * 255.0f);
	dest.w = 255;
}


_CPU_AND_GPU_CODE_ inline void processPixelICPPost(
    float angle, Vector3f outNormal,
    DEVICEPTR(Vector4u) &outRendering,
    DEVICEPTR(Vector4f) &pointsMap,
    DEVICEPTR(Vector4f) &normalsMap,
    const THREADPTR(Vector3f) & point,
    bool foundPoint,
    float voxelSize)
{

    if (foundPoint)
    {
        drawPixelGrey(outRendering, angle);

        Vector4f outPoint4;
        outPoint4.x = point.x * voxelSize; outPoint4.y = point.y * voxelSize;
        outPoint4.z = point.z * voxelSize; outPoint4.w = 1.0f;
        pointsMap = outPoint4;

        Vector4f outNormal4;
        outNormal4.x = outNormal.x; outNormal4.y = outNormal.y; outNormal4.z = outNormal.z; outNormal4.w = 0.0f;
        normalsMap = outNormal4;
    }
    else
    {
        Vector4f out4;
        out4.x = 0.0f; out4.y = 0.0f; out4.z = 0.0f; out4.w = -1.0f;

        pointsMap = out4; normalsMap = out4; outRendering = Vector4u((uchar)0);
    }
}
template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void processPixelICP(DEVICEPTR(Vector4u) &outRendering, DEVICEPTR(Vector4f) &pointsMap, DEVICEPTR(Vector4f) &normalsMap,
	const THREADPTR(Vector3f) & point, bool foundPoint, const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(typename TIndex::IndexData) *voxelIndex,
	float voxelSize, const THREADPTR(Vector3f) &lightSource)
{
	Vector3f outNormal;
	float angle;

	computeNormalAndAngle<TVoxel, TIndex>(foundPoint, point, voxelData, voxelIndex, lightSource, outNormal, angle);

    processPixelICPPost(
        angle, outNormal,
       outRendering,
       pointsMap,
       normalsMap,
        point,
        foundPoint,
        voxelSize);

}

/// \param useSmoothing whether to compute normals by forward differences two pixels away (true) or just one pixel away (false)
template<bool useSmoothing>
_CPU_AND_GPU_CODE_ inline void processPixelICP(
    DEVICEPTR(Vector4u) *outRendering,
    DEVICEPTR(Vector4f) *pointsMap, //!< receives output points in world coordinates
    DEVICEPTR(Vector4f) *normalsMap,
	const CONSTPTR(Vector4f) *pointsRay, //!< input points in voxel-fractional-world-coordinates
    const THREADPTR(Vector2i) &imgSize,
    const THREADPTR(int) &x, const THREADPTR(int) &y,
    float voxelSize,
	const THREADPTR(Vector3f) &lightSource)
{
	Vector3f outNormal;
	float angle;

    int locId = pixelLocId(x, y, imgSize);
	Vector4f point = pointsRay[locId];

	bool foundPoint = point.w > 0.0f;

	computeNormalAndAngle<useSmoothing>(foundPoint, x, y, pointsRay, lightSource, voxelSize, imgSize, outNormal, angle);

    processPixelICPPost(
        angle, outNormal,
        outRendering[locId],
        pointsMap[locId],
        normalsMap[locId],
        point.toVector3(),
        foundPoint,
        voxelSize);
}

template<bool useSmoothing>
_CPU_AND_GPU_CODE_ inline void processPixelForwardRender(DEVICEPTR(Vector4u) *outRendering, const CONSTPTR(Vector4f) *pointsRay, 
	const THREADPTR(Vector2i) &imgSize, const THREADPTR(int) &x, const THREADPTR(int) &y, float voxelSize, const THREADPTR(Vector3f) &lightSource)
{
	Vector3f outNormal;
	float angle;

    int locId = pixelLocId(x, y, imgSize);
	Vector4f point = pointsRay[locId];

	bool foundPoint = point.w > 0.0f;
	computeNormalAndAngle<useSmoothing>(foundPoint, x, y, pointsRay, lightSource, voxelSize, imgSize, outNormal, angle);

	if (foundPoint) drawPixelGrey(outRendering[locId], angle);
	else outRendering[locId] = Vector4u((uchar)0);
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void processPixelGrey(DEVICEPTR(Vector4u) &outRendering, const CONSTPTR(Vector3f) & point, 
	bool foundPoint, const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(typename TIndex::IndexData) *voxelIndex, 
	Vector3f lightSource)
{
	Vector3f outNormal;
	float angle;

	computeNormalAndAngle<TVoxel, TIndex>(foundPoint, point, voxelData, voxelIndex, lightSource, outNormal, angle);

	if (foundPoint) drawPixelGrey(outRendering, angle);
	else outRendering = Vector4u((uchar)0);
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void processPixelColour(DEVICEPTR(Vector4u) &outRendering, const CONSTPTR(Vector3f) & point,
	bool foundPoint, const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(typename TIndex::IndexData) *voxelIndex, 
	Vector3f lightSource)
{
	Vector3f outNormal;
	float angle;

	computeNormalAndAngle<TVoxel, TIndex>(foundPoint, point, voxelData, voxelIndex, lightSource, outNormal, angle);

	if (foundPoint) drawPixelColour<TVoxel, TIndex>(outRendering, point, voxelData, voxelIndex);
	else outRendering = Vector4u((uchar)0);
}


template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void processPixelNormal(DEVICEPTR(Vector4u) &outRendering, const CONSTPTR(Vector3f) & point,
	bool foundPoint, const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(typename TIndex::IndexData) *voxelIndex,
	Vector3f lightSource)
{
	Vector3f outNormal;
	float angle;

	computeNormalAndAngle<TVoxel, TIndex>(foundPoint, point, voxelData, voxelIndex, lightSource, outNormal, angle);

	if (foundPoint) drawPixelNormal(outRendering, outNormal);
	else outRendering = Vector4u((uchar)0);
}
