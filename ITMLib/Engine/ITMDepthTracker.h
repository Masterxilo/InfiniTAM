#pragma once

#include "ITMLibDefines.h"
#include "itmcudautils.h"
#include "ITMLowLevelEngine.h"
#include "ITMView.h"

/** Performing ICP based depth tracking. 
Implements the original KinectFusion tracking algorithm.

c.f. newcombe_etal_ismar2011.pdf section "Sensor Pose Estimation"

6-d parameter vector "x" is (beta, gamma, alpha, tx, ty, tz)
*/
void ImprovePose();
