// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMLibDefines.h"
#include "itmtrackingstate.h"
#include "ITMLowLevelEngine.h"
#include "ITMPixelUtils.h"
#include "itmview.h"

using namespace ITMLib::Objects;

#include <vector>

namespace ITMLib
{
	namespace Engine
	{
		/** Performing ICP based depth tracking. 
            Implements the original KinectFusion tracking algorithm.

            c.f. newcombe_etal_ismar2011.pdf section "Sensor Pose Estimation"

            6-d parameter vector "x" is (beta, gamma, alpha, tx, ty, tz)
		*/
        class ITMDepthTrackerOld
        {
        public:

            void TrackCamera(
                ITMTrackingState *trackingState, //!< [in,out] in: the current and out: the computed best fit adjusted camera pose for the new view
                const ITMView *view //<! latest camera data, for which the camera pose shall be adjusted
                );


            ITMDepthTrackerOld(
                Vector2i depthImgSize) {}

		};
	}
}
