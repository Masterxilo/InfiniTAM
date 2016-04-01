#pragma once

#include <math.h>

#include "ITMLibDefines.h"

#include "ITMView.h"
#include "ITMTrackingState.h"
#include "ITMRenderState.h"


/** \brief
    main KinectFusion depth integration process
*/
void FuseView(
    const ITMView * const view,
    Matrix4f M_d);
