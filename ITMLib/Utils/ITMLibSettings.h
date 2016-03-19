// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Objects/ITMSceneParams.h"

namespace ITMLib
{
	namespace Objects
	{
		class ITMLibSettings
		{
		public:

			/// For ITMDepthTracker: ICP distance threshold
			float depthTrackerICPThreshold;

			/// For ITMDepthTracker: ICP iteration termination threshold
			float depthTrackerTerminationThreshold;

			/// Further, scene specific parameters such as voxel size
			ITMLib::Objects::ITMSceneParams sceneParams;

            ITMLibSettings::ITMLibSettings(void)
                : sceneParams(0.02f, 100, 0.005f, 0.2f, 3.0f)
            {
                /// depth threashold for the ICP tracker
                depthTrackerICPThreshold = 0.1f * 0.1f;

                /// For ITMDepthTracker: ICP iteration termination threshold
                depthTrackerTerminationThreshold = 1e-3f;
            }

		};
	}
}
