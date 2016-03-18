// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Utils/ITMLibDefines.h"

#include "../Objects/ITMTrackingState.h"
#include "ITMTracker.h"
#include "../Objects/ITMRenderState.h"

#include "../Engine/ITMVisualisationEngine.h"
#include "../Engine/ITMLowLevelEngine.h"


namespace ITMLib
{
	namespace Engine
	{
		/** \brief ITMTracker driver
		*/
		class ITMTrackingController
		{
		private:
			const ITMLibSettings *settings;
            const ITMVisualisationEngine *visualisationEngine;
			const ITMLowLevelEngine *lowLevelEngine;

			ITMTracker *tracker;

			MemoryDeviceType memoryType;

		public:
			void Track(ITMTrackingState *trackingState, const ITMView *view);
			void Prepare(ITMTrackingState *trackingState, const ITMView *view, ITMRenderState *renderState);

            ITMTrackingController(ITMTracker *tracker, const ITMVisualisationEngine *visualisationEngine, const ITMLowLevelEngine *lowLevelEngine,
				const ITMLibSettings *settings)
			{
				this->tracker = tracker;
				this->settings = settings;
				this->visualisationEngine = visualisationEngine;
				this->lowLevelEngine = lowLevelEngine;

				memoryType = MEMORYDEVICE_CUDA;
			}

			ITMTrackingState *BuildTrackingState(const Vector2i & trackedImageSize) const
			{
				return new ITMTrackingState(trackedImageSize, memoryType);
			}

			static Vector2i GetTrackedImageSize(const ITMLibSettings *settings, const Vector2i& imgSize_rgb, const Vector2i& imgSize_d)
			{
				return imgSize_d;
			}
		};
	}
}
