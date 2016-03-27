// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include <math.h>

#include "ITMLibDefines.h"

#include "ITMView.h"
#include "ITMTrackingState.h"
#include "ITMRenderState.h"

using namespace ITMLib::Objects;

		/** \brief
            main KinectFusion depth integration process
		*/
        void ITMSceneReconstructionEngine_ProcessFrame(
                const ITMView * const view,
                const ITMTrackingState * const trackingState
                );
