/** \brief
Interface to engines helping with the visualisation of
the results from the rest of the library.

This is also used internally to get depth estimates for the
raycasting done for the trackers. The basic idea there is
to project down a scene of 8x8x8 voxel
blocks and look at the bounding boxes. The projection
provides an idea of the possible depth range for each pixel
in an image, which can be used to speed up raycasting
operations.
*/
#pragma once

#include "ITMLibDefines.h"
#include "ITMView.h"
#include "ITMTrackingState.h"
#include "ITMRenderState.h"
#include "cameraimage.h"

/** This will render an image using raycasting.
TODO could render into a view*/
CameraImage<Vector4u>* RenderImage(
    const ITMPose *pose,
    const ITMIntrinsics *intrinsics,
    const Vector2i imgSize,
    std::string shader);
