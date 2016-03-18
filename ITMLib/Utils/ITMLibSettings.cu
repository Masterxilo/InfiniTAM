// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMLibSettings.h"

#include <stdio.h>

using namespace ITMLib::Objects;

ITMLibSettings::ITMLibSettings(void)
	: sceneParams(0.02f, 100, 0.005f, 0.2f, 3.0f)
{
	/// depth threashold for the ICP tracker
	depthTrackerICPThreshold = 0.1f * 0.1f;

	/// For ITMDepthTracker: ICP iteration termination threshold
	depthTrackerTerminationThreshold = 1e-3f;

	/// skips every other point when using the colour tracker
	skipPoints = true;

	//deviceType = DEVICE_CPU;
}

ITMLibSettings::~ITMLibSettings()
{
	delete[] trackingRegime;
}
