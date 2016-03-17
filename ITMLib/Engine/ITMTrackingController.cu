// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMTrackingController.h"

#include "../Objects/ITMRenderState_VH.h"

#include "../ITMLib.h"

using namespace ITMLib::Engine;

void ITMTrackingController::Track(ITMTrackingState *trackingState, const ITMView *view)
{
	if (trackingState->age_pointCloud!=-1) tracker->TrackCamera(trackingState, view);

	trackingState->requiresFullRendering = trackingState->TrackerFarFromPointCloud() || !settings->useApproximateRaycast;
}

void ITMTrackingController::Prepare(ITMTrackingState *trackingState, const ITMView *view, ITMRenderState *renderState)
{
	visualisationEngine->CreateExpectedDepths(trackingState->pose_d, &(view->calib->intrinsics_d), renderState);

	if (trackingState->requiresFullRendering)
	{
		visualisationEngine->CreateICPMaps(view, trackingState, renderState);
		trackingState->pose_pointCloud->SetFrom(trackingState->pose_d);
		if (trackingState->age_pointCloud==-1) trackingState->age_pointCloud=-2;
		else trackingState->age_pointCloud = 0;
	}
	else
	{
		visualisationEngine->ForwardRender(view, trackingState, renderState);
		trackingState->age_pointCloud++;
	}
}
