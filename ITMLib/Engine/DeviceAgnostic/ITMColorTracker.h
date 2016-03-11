/// \file ITMColorTracker
/// TYPE& are output parameters
/// const CONSTPTR(Vector2i) & are inputs
// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../Utils/ITMLibDefines.h"
#include "ITMPixelUtils.h"



/// $$I_C(\\pi(M p))$$
/// \returns false on error
_CPU_AND_GPU_CODE_ inline bool grabObservedColor(
    Vector4f projParams, Matrix4f M, const CONSTPTR(Vector2i) & imgSize,
    Vector4f pt_model,
    Vector4f& pt_camera, Vector2f& pt_image,
    DEVICEPTR(Vector4u) *rgb, DEVICEPTR(Vector4f) *locations, int locId_global, Vector4f& colour_obs
    ) {
    if (!projectModel(projParams, M, imgSize, locations[locId_global], pt_camera, pt_image)) return false;
    colour_obs = interpolateBilinear(rgb, pt_image, imgSize);
    if (colour_obs.w < 254.0f) return false;

    return true;
}

/// $$I_C(\\pi(M p)) - C(p)$$
/// difference between known color in rgb image and on model (colours)
/// \returns false on failure
_CPU_AND_GPU_CODE_ inline float getColorDifference(DEVICEPTR(Vector4f) *locations, DEVICEPTR(Vector4f) *colours, DEVICEPTR(Vector4u) *rgb,
    const CONSTPTR(Vector2i) & imgSize, int locId_global, Vector4f projParams, Matrix4f M,
    Vector3f& colour_diff, Vector4f& pt_camera, Vector2f& pt_image)
{
    Vector4f colour_known, colour_obs;

    if (!grabObservedColor(projParams, M, imgSize, locations[locId_global], pt_camera, pt_image,
        rgb, locations, locId_global, colour_obs)) return false;

    // Grab known color
    colour_known = colours[locId_global];

    // Compute difference (colour_obs is in R8G8B8A8, but colour_known in UNORM_RGBA)
    colour_diff = colour_obs.toVector3() - 255.0f * colour_known.toVector3();
    return true;
}

/// Compute squared difference between color in rgb image and on model (colours)
/// \f[||I_C(\pi(M p)) - C(p)||_2^2\f]
/// where p and C(p) are the 3D points and their colours, as extracted in
/// the raycasting stage, and \f$I_C\f$ is the current colour image.
///
/// * p = locations[locId_global]
/// * C(p) = colours[locId_global]
/// * I_C = rgb
/// * projParams controls \f$\pi\f$
///
/// \returns -1.f on failure
_CPU_AND_GPU_CODE_ inline float getColorDifferenceSq(DEVICEPTR(Vector4f) *locations, DEVICEPTR(Vector4f) *colours, DEVICEPTR(Vector4u) *rgb,
    const CONSTPTR(Vector2i) & imgSize, int locId_global, Vector4f projParams, Matrix4f M)
{
    Vector3f colour_diff;
    Vector2f pt_image;
    Vector4f pt_camera;
    if (!getColorDifference(
        locations, colours, rgb, imgSize, locId_global, projParams, M,
        colour_diff, pt_camera, pt_image)) return -1;

    return dot(colour_diff, colour_diff);
}

/// Returns false on failure
_CPU_AND_GPU_CODE_ inline bool computePerPointGH_rt_Color(THREADPTR(float) *localGradient, THREADPTR(float) *localHessian,
    DEVICEPTR(Vector4f) *locations, DEVICEPTR(Vector4f) *colours, DEVICEPTR(Vector4u) *rgb, const CONSTPTR(Vector2i) & imgSize,
    int locId_global, Vector4f projParams, Matrix4f M, DEVICEPTR(Vector4s) *gx, DEVICEPTR(Vector4s) *gy, int numPara, int startPara)
{
    Vector4f pt_camera, gx_obs, gy_obs;
    Vector3f colour_diff_d, d_pt_cam_dpi, d[6];
    Vector2f pt_image, d_proj_dpi;


    if (!getColorDifference(
        locations, colours, rgb, imgSize, locId_global, projParams, M,
        colour_diff_d, pt_camera, pt_image)) return false;
    colour_diff_d *= 2.0f;

    gx_obs = interpolateBilinear(gx, pt_image, imgSize);
    gy_obs = interpolateBilinear(gy, pt_image, imgSize);

    for (int para = 0, counter = 0; para < numPara; para++)
    {
        switch (para + startPara)
        {
        case 0: d_pt_cam_dpi.x = pt_camera.w;  d_pt_cam_dpi.y = 0.0f;         d_pt_cam_dpi.z = 0.0f;         break;
        case 1: d_pt_cam_dpi.x = 0.0f;         d_pt_cam_dpi.y = pt_camera.w;  d_pt_cam_dpi.z = 0.0f;         break;
        case 2: d_pt_cam_dpi.x = 0.0f;         d_pt_cam_dpi.y = 0.0f;         d_pt_cam_dpi.z = pt_camera.w;  break;
        case 3: d_pt_cam_dpi.x = 0.0f;         d_pt_cam_dpi.y = -pt_camera.z;  d_pt_cam_dpi.z = pt_camera.y;  break;
        case 4: d_pt_cam_dpi.x = pt_camera.z;  d_pt_cam_dpi.y = 0.0f;         d_pt_cam_dpi.z = -pt_camera.x;  break;
        default:
        case 5: d_pt_cam_dpi.x = -pt_camera.y;  d_pt_cam_dpi.y = pt_camera.x;  d_pt_cam_dpi.z = 0.0f;         break;
        };

        d_proj_dpi.x = projParams.x * ((pt_camera.z * d_pt_cam_dpi.x - d_pt_cam_dpi.z * pt_camera.x) / (pt_camera.z * pt_camera.z));
        d_proj_dpi.y = projParams.y * ((pt_camera.z * d_pt_cam_dpi.y - d_pt_cam_dpi.z * pt_camera.y) / (pt_camera.z * pt_camera.z));

        d[para] = d_proj_dpi.x * gx_obs.toVector3() + d_proj_dpi.y * gy_obs.toVector3();

        localGradient[para] = dot(d[para], colour_diff_d);

        for (int col = 0; col <= para; col++)
            localHessian[counter++] = 2.0f * dot(d[para], d[col]);
    }

    return true;
}
