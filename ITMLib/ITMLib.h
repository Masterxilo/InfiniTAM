// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#ifndef __InfiniTAM_LIB__
#define __InfiniTAM_LIB__

#include "Utils/ITMLibDefines.h"

#include "Objects/ITMScene.h"
#include "Objects/ITMView.h"

#include "Engine/ITMLowLevelEngine.h"
#ifndef COMPILE_WITHOUT_CUDA
#include "Engine/DeviceSpecific/CUDA/ITMLowLevelEngine_CUDA.h"
#else
#include "Engine/DeviceSpecific/CPU/ITMLowLevelEngine_CPU.h"
#endif

#include "Engine/ITMDepthTracker.h"
#ifndef COMPILE_WITHOUT_CUDA
#include "Engine/DeviceSpecific/CUDA/ITMDepthTracker_CUDA.h"
#else
#include "Engine/DeviceSpecific/CPU/ITMDepthTracker_CPU.h"
#endif

#include "Engine/ITMWeightedICPTracker.h"
#ifndef COMPILE_WITHOUT_CUDA
#include "Engine/DeviceSpecific/CUDA/ITMWeightedICPTracker_CUDA.h"
#else
#include "Engine/DeviceSpecific/CPU/ITMWeightedICPTracker_CPU.h"
#endif



#include "Engine/ITMSceneReconstructionEngine.h"
#ifndef COMPILE_WITHOUT_CUDA
#include "Engine/DeviceSpecific/CUDA/ITMSceneReconstructionEngine_CUDA.h"
#else
#include "Engine/DeviceSpecific/CPU/ITMSceneReconstructionEngine_CPU.h"
#endif

#include "Engine/ITMVisualisationEngine.h"
#ifndef COMPILE_WITHOUT_CUDA
#include "Engine/DeviceSpecific/CUDA/ITMVisualisationEngine_CUDA.h"
#else
#include "Engine/DeviceSpecific/CPU/ITMVisualisationEngine_CPU.h"
#endif

#include "Engine/ITMColorTracker.h"
#ifndef COMPILE_WITHOUT_CUDA
#include "Engine/DeviceSpecific/CUDA/ITMColorTracker_CUDA.h"
#else
#include "Engine/DeviceSpecific/CPU/ITMColorTracker_CPU.h"
#endif

#include "Engine/ITMRenTracker.h"
#ifndef COMPILE_WITHOUT_CUDA
#include "Engine/DeviceSpecific/CUDA/ITMRenTracker_CUDA.h"
#else
#include "Engine/DeviceSpecific/CPU/ITMRenTracker_CPU.h"
#endif

#include "Engine/ITMIMUTracker.h"
#include "Engine/ITMCompositeTracker.h"
#include "Engine/ITMTrackingController.h"

#include "Engine/ITMViewBuilder.h"
#ifndef COMPILE_WITHOUT_CUDA
#include "Engine/DeviceSpecific/CUDA/ITMViewBuilder_CUDA.h"
#else
#include "Engine/DeviceSpecific/CPU/ITMViewBuilder_CPU.h"
#endif

#include "Engine/ITMMeshingEngine.h"
#ifndef COMPILE_WITHOUT_CUDA
#include "Engine/DeviceSpecific/CUDA/ITMMeshingEngine_CUDA.h"
#else
#include "Engine/DeviceSpecific/CPU/ITMMeshingEngine_CPU.h"
#endif

#include "Engine/ITMDenseMapper.h"
#include "Engine/ITMMainEngine.h"

using namespace ITMLib::Objects;
using namespace ITMLib::Engine;

#endif
