// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../ITMDepthTracker.h"

namespace ITMLib
{
	namespace Engine
	{
		class ITMDepthTracker_CUDA : public ITMDepthTracker
		{

		private:
            /// CUDA managed host/device memory (call cudaDeviceSynchronize before using on host)
			AccuCell *accu;

		protected:
            AccuCell ComputeGandH(Matrix4f T_g_k_estimate);

		public:
			ITMDepthTracker_CUDA(Vector2i imgSize, TrackerIterationType *trackingRegime, int noHierarchyLevels, int noICPRunTillLevel, float distThresh,
				float terminationThreshold, const ITMLowLevelEngine *lowLevelEngine);
			~ITMDepthTracker_CUDA(void);
		};
	}
}
