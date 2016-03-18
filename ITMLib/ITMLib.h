// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#ifndef __InfiniTAM_LIB__
#define __InfiniTAM_LIB__

#include "Utils/ITMLibDefines.h"

#include "Objects/ITMScene.h"
#include "Objects/ITMView.h"

#include "Engine/ITMLowLevelEngine.h"

#include "Engine/ITMDepthTracker.h"
#include "Engine/DeviceSpecific/CUDA/ITMDepthTracker_CUDA.h"

#include "Engine/ITMSceneReconstructionEngine.h"
#include "Engine/DeviceSpecific/CUDA/ITMSceneReconstructionEngine_CUDA.h"

#include "Engine/ITMVisualisationEngine.h"
#include "Engine/DeviceSpecific/CUDA/ITMVisualisationEngine_CUDA.h"

#include "Engine/ITMTrackingController.h"

#include "Engine/ITMViewBuilder.h"
#include "Engine/DeviceSpecific/CUDA/ITMViewBuilder_CUDA.h"

#include "Engine/ITMMainEngine.h"

using namespace ITMLib::Objects;
using namespace ITMLib::Engine;

#endif
