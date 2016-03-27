// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMMainEngine.h"

using namespace ITMLib::Engine;

ITMMainEngine::ITMMainEngine(const ITMRGBDCalib *calib, Vector2i imgSize_rgb, Vector2i imgSize_d)
{
    scene = new Scene();

	lowLevelEngine = new ITMLowLevelEngine();
	viewBuilder = new ITMViewBuilder(calib);
	visualisationEngine = new ITMVisualisationEngine();

    Vector2i trackedImageSize = imgSize_d;

	renderState_live = visualisationEngine->CreateRenderState(trackedImageSize);
	renderState_freeview = NULL; //will be created by the visualisation engine on demand

    ResetScene();

    tracker = new ITMDepthTracker(
        trackedImageSize,
        lowLevelEngine
        );
    trackingState = tracker->BuildTrackingState();

	view = NULL; // will be allocated by the view builder

	fusionActive = true;
	mainProcessingActive = true;
}

ITMMainEngine::~ITMMainEngine()
{
	delete renderState_live;
    delete renderState_freeview;

	delete scene;

	delete tracker;

	delete lowLevelEngine;
	delete viewBuilder;

	delete trackingState;
	if (view != NULL) delete view;

	delete visualisationEngine;
}

void ITMMainEngine::ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage)
{
    CURRENT_SCENE_SCOPE(scene);

	// prepare image and turn it into a depth image
	viewBuilder->UpdateView(&view, rgbImage, rawDepthImage);

	// tracking
    tracker->TrackCamera(trackingState, view);

	// fusion
    ITMSceneReconstructionEngine_ProcessFrame(view, trackingState);

    // raycast scene from current viewpoint 
    // to create point cloud for tracking
    visualisationEngine->CreateICPMaps(trackingState, &view->calib->intrinsics_d, renderState_live);
}

void ITMMainEngine::GetImage(
    ITMUChar4Image * const out,
    const GetImageType getImageType, 
    const ITMPose * const pose, 
    const ITMIntrinsics * const intrinsics 
    )
{
    assert(out->isAllocated_CPU() && out->isAllocated_CUDA());
	if (view == NULL) return;

	out->Clear();

	switch (getImageType)
	{
	case ITMMainEngine::InfiniTAM_IMAGE_ORIGINAL_RGB:
		out->ChangeDims(view->rgb->noDims);
        out->SetFrom(view->rgb, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
        break;

	case ITMMainEngine::InfiniTAM_IMAGE_ORIGINAL_DEPTH:
		out->ChangeDims(view->depth->noDims);
        view->depth->UpdateHostFromDevice();
        ITMVisualisationEngine::DepthToUchar4(out, view->depth);
		break;

	case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_SHADED:
	case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME:
	case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL:
	{
		ITMVisualisationEngine::RenderImageType type = ITMVisualisationEngine::RENDER_SHADED_GREYSCALE;
		if (getImageType == ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME) 
            type = ITMVisualisationEngine::RENDER_COLOUR_FROM_VOLUME;
		else if (getImageType == ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL) 
            type = ITMVisualisationEngine::RENDER_COLOUR_FROM_NORMAL;

		if (renderState_freeview == NULL)
            renderState_freeview = visualisationEngine->CreateRenderState(out->noDims);

        assert(renderState_freeview->raycastResult->noDims == out->noDims);

        CURRENT_SCENE_SCOPE(scene);
		visualisationEngine->RenderImage(pose, intrinsics, renderState_freeview, out, type);
        out->UpdateHostFromDevice();
		break;
	}
	case ITMMainEngine::InfiniTAM_IMAGE_UNKNOWN:
		break;
	};
}