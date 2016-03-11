// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../Utils/ITMLibDefines.h"
#include "ITMPixelUtils.h"
#include "ITMRepresentationAccess.h"

/// Camera Data Integration
/// \returns \f$\eta\f$, -1 on failure
// Note that the stored T-SDF values are normalized to lie
// in [-1,1] within the truncation band.
template<class TVoxel>
_CPU_AND_GPU_CODE_ inline float computeUpdatedVoxelDepthInfo(
    DEVICEPTR(TVoxel) &voxel, //!< X
    const THREADPTR(Vector4f) & pt_model, //!< voxel location
    const CONSTPTR(Matrix4f) & M_d, //!< depth camera pose
    const CONSTPTR(Vector4f) & projParams_d, //!< intrinsic camera parameters \f$K_d\f$
    float mu, int maxW, const CONSTPTR(float) *depth, const CONSTPTR(Vector2i) & imgSize)
{
    /// X_d
    Vector4f pt_camera;
    /// \pi(K_dX_d)
    Vector2f pt_image;

    float depth_measure, eta, oldF, newF;
    int oldW, newW;

    // project point into depth image
    if (!projectModel(projParams_d, M_d,
        imgSize, pt_model, pt_camera, pt_image)) return -1;

    // get measured depth from image, no interpolation
    /// I_d(\pi(K_dX_d))
    depth_measure = sampleNearest(depth, pt_image, imgSize);
    if (depth_measure <= 0.0) return -1;

    /// I_d(\pi(K_dX_d)) - X_d^(z)          (3)
    eta = depth_measure - pt_camera.z;
    // check whether voxel needs updating
    if (eta < -mu) return eta;

    // compute updated SDF value and reliability (number of observations)
    /// D(X), w(X)
    oldF = TVoxel::SDF_valueToFloat(voxel.sdf);
    oldW = voxel.w_depth;

    // newF, normalized for -1 to 1
    newF = MIN(1.0f, eta / mu);
    newW = 1;

    updateVoxelDepthInformation(
        voxel,
        oldF, oldW, newF, newW, maxW);

    return eta;
}


/// \returns -1 on failure
template<class TVoxel>
_CPU_AND_GPU_CODE_ inline void computeUpdatedVoxelColorInfo(DEVICEPTR(TVoxel) &voxel, const THREADPTR(Vector4f) & pt_model, const CONSTPTR(Matrix4f) & M_rgb,
    const CONSTPTR(Vector4f) & projParams_rgb, float mu, uchar maxW, float eta, const CONSTPTR(Vector4u) *rgb, const CONSTPTR(Vector2i) & imgSize)
{
    Vector4f pt_camera; Vector2f pt_image;
    Vector3f oldC, newC;
    float newW, oldW;

    if (!projectModel(projParams_rgb, M_rgb,
        imgSize, pt_model, pt_camera, pt_image)) return;

    oldW = (float)voxel.w_color;
    oldC = TO_FLOAT3(voxel.clr);

    /// Like formula (4) for depth
    newC = TO_VECTOR3(interpolateBilinear(rgb, pt_image, imgSize));
    newW = 1;

    updateVoxelColorInformation(
        voxel,
        oldC, oldW, newC, newW, maxW);
}

template<bool hasColor, class TVoxel> struct ComputeUpdatedVoxelInfo;

template<class TVoxel>
struct ComputeUpdatedVoxelInfo<false, TVoxel> {
    _CPU_AND_GPU_CODE_ static void compute(DEVICEPTR(TVoxel) & voxel, const THREADPTR(Vector4f) & pt_model,
        const CONSTPTR(Matrix4f) & M_d, const CONSTPTR(Vector4f) & projParams_d,
        const CONSTPTR(Matrix4f) & M_rgb, const CONSTPTR(Vector4f) & projParams_rgb,
        float mu, int maxW,
        const CONSTPTR(float) *depth, const CONSTPTR(Vector2i) & imgSize_d,
        const CONSTPTR(Vector4u) *rgb, const CONSTPTR(Vector2i) & imgSize_rgb)
    {
        computeUpdatedVoxelDepthInfo(voxel, pt_model, M_d, projParams_d, mu, maxW, depth, imgSize_d);
    }
};

template<class TVoxel>
struct ComputeUpdatedVoxelInfo<true, TVoxel> {
    _CPU_AND_GPU_CODE_ static void compute(DEVICEPTR(TVoxel) & voxel, const THREADPTR(Vector4f) & pt_model,
        const THREADPTR(Matrix4f) & M_d, const THREADPTR(Vector4f) & projParams_d,
        const THREADPTR(Matrix4f) & M_rgb, const THREADPTR(Vector4f) & projParams_rgb,
        float mu, int maxW,
        const CONSTPTR(float) *depth, const CONSTPTR(Vector2i) & imgSize_d,
        const CONSTPTR(Vector4u) *rgb, const THREADPTR(Vector2i) & imgSize_rgb)
    {
        float eta = computeUpdatedVoxelDepthInfo(voxel, pt_model, M_d, projParams_d, mu, maxW, depth, imgSize_d);

        // Only the voxels withing 25% mu affect the color
        if ((eta > mu) || (fabs(eta / mu) > 0.25f)) return;
        computeUpdatedVoxelColorInfo(voxel, pt_model, M_rgb, projParams_rgb, mu, maxW, eta, rgb, imgSize_rgb);
    }
};

// alloc types
#define AT_NEEDS_ALLOC_FITS 1 //needs allocation, fits in the ordered list
#define AT_NEEDS_ALLOC_EXCESS 2 //needs allocation in the excess list

// visible type
#define VT_VISIBLE_AND_IN_MEMORY 1 //make child visible and in memory
#define VT_VISIBLE_AND_STREAMED_OUT 2//entry has been streamed out but is visible
#define VT_VISIBLE_PREVIOUS_AND_UNSTREAMED 3 // visible at previous frame and unstreamed

/// Determine the blocks around a given depth sample that are currently visible
/// and need to be allocated.
/// Builds hashVisibility and entriesAllocType.
/// \param x,y [in] loop over depth image.
_CPU_AND_GPU_CODE_ inline void buildHashAllocAndVisibleTypePP(
    DEVICEPTR(uchar) *entriesAllocType, //!< [out] allocation type AT_ for each hash table bucket, indexed by values computed from hashIndex
    DEVICEPTR(uchar) *entriesVisibleType,//!< [out] visibility type for each hash table bucket, indexed by values computed from hashIndex
    int x, int y,
    DEVICEPTR(Vector4s) *blockCoords, //!< [out] blockPos coordinate of each voxel block, indexed by values computed from hashIndex 
    const CONSTPTR(float) *depth,
    Matrix4f invM_d, //!< depth to world transformation
    Vector4f invProjParams_d, //!< Note: Inverse projection parameters to avoid division by fx, fy.
    float mu, 
    Vector2i imgSize,
    float oneOverVoxelSize, //!< 1 / (voxelSize * SDF_BLOCK_SIZE)
    const CONSTPTR(ITMHashEntry) *hashTable, //<! [in] hash table buckets, indexed by values computed from hashIndex
    float viewFrustum_min, //!< znear
    float viewFrustum_max  //!< zfar
    )
{
    float depth_measure; unsigned int hashIdx; int noSteps;
    Vector4f pt_camera_f; Vector3f point_e, point, direction; Vector3s blockPos;

    // Find 3d position of depth pixel xy
    depth_measure = depth[x + y * imgSize.x];
    if (depth_measure <= 0 || (depth_measure - mu) < 0 || (depth_measure - mu) < viewFrustum_min || (depth_measure + mu) > viewFrustum_max) return;

    pt_camera_f = depthTo3DInvProjParams(invProjParams_d, x, y, depth_measure);

    // distance from camera
    float norm = length(pt_camera_f.toVector3());

    // Transform into block coordinates the found point +- mu
    point = TO_VECTOR3(invM_d * (pt_camera_f * (1.0f - mu / norm))) * oneOverVoxelSize;
    point_e = TO_VECTOR3(invM_d * (pt_camera_f * (1.0f + mu / norm))) * oneOverVoxelSize;

    // We will step along point -> point_e and add all voxel blocks we encounter to the visible list
    direction = point_e - point;
    norm = length(direction);
    noSteps = (int)ceil(2.0f*norm);

    direction /= (float)(noSteps - 1);

    //add neighbouring blocks
    for (int i = 0; i < noSteps; i++)
    {
        blockPos = TO_SHORT_FLOOR3(point);

        //compute index in hash table
        hashIdx = hashIndex(blockPos);

        //check if hash table contains entry
        bool isFound = false;

        ITMHashEntry hashEntry = hashTable[hashIdx];

        if (IS_EQUAL3(hashEntry.pos, blockPos) && hashEntry.ptr >= -1)
        {
            //entry has been streamed out but is visible or in memory and visible
            entriesVisibleType[hashIdx] = (hashEntry.ptr == -1) ? VT_VISIBLE_AND_STREAMED_OUT : VT_VISIBLE_AND_IN_MEMORY;

            isFound = true;
        }

        if (!isFound)
        {
            bool isExcess = false;
            if (hashEntry.ptr >= -1) //seach excess list only if there is no room in ordered part
            {
                while (hashEntry.offset >= 1)
                {
                    hashIdx = SDF_BUCKET_NUM + hashEntry.offset - 1;
                    hashEntry = hashTable[hashIdx];

                    if (IS_EQUAL3(hashEntry.pos, blockPos) && hashEntry.ptr >= -1)
                    {
                        //entry has been streamed out but is visible or in memory and visible
                        entriesVisibleType[hashIdx] = (hashEntry.ptr == -1) ? 2 : 1;

                        isFound = true;
                        break;
                    }
                }

                isExcess = true;
            }

            if (!isFound) //still not found
            {
                entriesAllocType[hashIdx] = isExcess ? AT_NEEDS_ALLOC_EXCESS : AT_NEEDS_ALLOC_FITS; //needs allocation 
                if (!isExcess) entriesVisibleType[hashIdx] = 1; //new entry is visible

                blockCoords[hashIdx] = Vector4s(blockPos.x, blockPos.y, blockPos.z, 1);
            }
        }

        point += direction;
    }
}

template<bool useSwapping>
_CPU_AND_GPU_CODE_ inline void checkPointVisibility(THREADPTR(bool) &isVisible, THREADPTR(bool) &isVisibleEnlarged,
    const THREADPTR(Vector4f) &pt_model, const CONSTPTR(Matrix4f) & M_d, const CONSTPTR(Vector4f) &projParams_d,
    const CONSTPTR(Vector2i) &imgSize)
{
    Vector4f pt_camera; Vector2f pt_image;
    if (projectModel(projParams_d, M_d, imgSize, pt_model, pt_camera, pt_image)) {
        isVisible = true; isVisibleEnlarged = true;
    }
    else if (useSwapping)
    {
        // If swapping is used, consider the block visible when in the following enlarged region (enlarge by 1/8).
        Vector4i lims;
        lims.x = -imgSize.x / 8; lims.y = imgSize.x + imgSize.x / 8;
        lims.z = -imgSize.y / 8; lims.w = imgSize.y + imgSize.y / 8;

        if (pt_image.x >= lims.x && pt_image.x < lims.y && pt_image.y >= lims.z && pt_image.y < lims.w) isVisibleEnlarged = true;
    }
}

/// project the eight corners of the given voxel block
/// into the camera viewpoint and check their visibility
template<bool useSwapping>
_CPU_AND_GPU_CODE_ inline void checkBlockVisibility(THREADPTR(bool) &isVisible, THREADPTR(bool) &isVisibleEnlarged,
    const THREADPTR(Vector3s) &hashPos, const CONSTPTR(Matrix4f) & M_d, const CONSTPTR(Vector4f) &projParams_d,
    const CONSTPTR(float) &voxelSize, const CONSTPTR(Vector2i) &imgSize)
{
    Vector4f pt_model;
    const float factor = (float)SDF_BLOCK_SIZE * voxelSize;

    isVisible = false; isVisibleEnlarged = false;

    // 0 0 0
    pt_model.x = (float)hashPos.x * factor; pt_model.y = (float)hashPos.y * factor;
    pt_model.z = (float)hashPos.z * factor; pt_model.w = 1.0f;
    checkPointVisibility<useSwapping>(isVisible, isVisibleEnlarged, pt_model, M_d, projParams_d, imgSize);
    if (isVisible) return;

    // 0 0 1
    pt_model.z += factor;
    checkPointVisibility<useSwapping>(isVisible, isVisibleEnlarged, pt_model, M_d, projParams_d, imgSize);
    if (isVisible) return;

    // 0 1 1
    pt_model.y += factor;
    checkPointVisibility<useSwapping>(isVisible, isVisibleEnlarged, pt_model, M_d, projParams_d, imgSize);
    if (isVisible) return;

    // 1 1 1
    pt_model.x += factor;
    checkPointVisibility<useSwapping>(isVisible, isVisibleEnlarged, pt_model, M_d, projParams_d, imgSize);
    if (isVisible) return;

    // 1 1 0 
    pt_model.z -= factor;
    checkPointVisibility<useSwapping>(isVisible, isVisibleEnlarged, pt_model, M_d, projParams_d, imgSize);
    if (isVisible) return;

    // 1 0 0 
    pt_model.y -= factor;
    checkPointVisibility<useSwapping>(isVisible, isVisibleEnlarged, pt_model, M_d, projParams_d, imgSize);
    if (isVisible) return;

    // 0 1 0
    pt_model.x -= factor; pt_model.y += factor;
    checkPointVisibility<useSwapping>(isVisible, isVisibleEnlarged, pt_model, M_d, projParams_d, imgSize);
    if (isVisible) return;

    // 1 0 1
    pt_model.x += factor; pt_model.y -= factor; pt_model.z += factor;
    checkPointVisibility<useSwapping>(isVisible, isVisibleEnlarged, pt_model, M_d, projParams_d, imgSize);
    if (isVisible) return;
}
