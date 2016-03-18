// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMTrackingController.h"


#include "../ITMLib.h"

using namespace ITMLib::Engine;

void ITMTrackingController::Track(ITMTrackingState *trackingState, const ITMView *view)
{
    tracker->TrackCamera(trackingState, view);
}

void ITMTrackingController::Prepare(ITMTrackingState *trackingState, const ITMView *view, ITMRenderState *renderState)
{
	visualisationEngine->CreateICPMaps(view, trackingState, renderState);
	trackingState->pose_pointCloud->SetFrom(trackingState->pose_d);
}
