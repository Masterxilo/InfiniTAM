#pragma once

#include "ITMLibDefines.h"
#include "itmcudautils.h"
#include "ITMLowLevelEngine.h"
#include "ITMView.h"
#include "itmtrackingstate.h"

/** Performing ICP based depth tracking. 
Implements the original KinectFusion tracking algorithm.

c.f. newcombe_etal_ismar2011.pdf section "Sensor Pose Estimation"

6-d parameter vector "x" is (beta, gamma, alpha, tx, ty, tz)
*/
void TrackCamera(
    ITMTrackingState *trackingState, //!< [in,out] in: the current and out: the computed best fit adjusted camera pose for the new view
    const ITMView *view //<! latest camera data, for which the camera pose shall be adjusted
);
