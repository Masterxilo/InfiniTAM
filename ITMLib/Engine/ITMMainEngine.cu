// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMMainEngine.h"

using namespace ITMLib::Engine;

ITMMainEngine::ITMMainEngine(const ITMLibSettings *settings, const ITMRGBDCalib *calib, Vector2i imgSize_rgb, Vector2i imgSize_d)
{
	this->settings = settings;

	scene = new ITMScene(&(settings->sceneParams));

	lowLevelEngine = new ITMLowLevelEngine();
	viewBuilder = new ITMViewBuilder_CUDA(calib);
	visualisationEngine = new ITMVisualisationEngine_CUDA(scene);

    Vector2i trackedImageSize = imgSize_d;

	renderState_live = visualisationEngine->CreateRenderState(trackedImageSize);
	renderState_freeview = NULL; //will be created by the visualisation engine on demand

    sceneRecoEngine = new ITMSceneReconstructionEngine_CUDA();
    ResetScene();

    tracker = new ITMDepthTracker_CUDA(
        trackedImageSize,
        settings->depthTrackerICPThreshold,
        settings->depthTrackerTerminationThreshold,
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
	if (renderState_freeview!=NULL) delete renderState_freeview;

	delete scene;

    delete sceneRecoEngine;

	delete tracker;

	delete lowLevelEngine;
	delete viewBuilder;

	delete trackingState;
	if (view != NULL) delete view;

	delete visualisationEngine;
}

void ITMMainEngine::ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage)
{
	// prepare image and turn it into a depth image
	viewBuilder->UpdateView(&view, rgbImage, rawDepthImage);

	// tracking
    tracker->TrackCamera(trackingState, view);

	// fusion
    sceneRecoEngine->ProcessFrame(view, trackingState, scene, renderState_live);

	// raycast to create point cloud for tracking in next iteration
    visualisationEngine->CreateICPMaps(&view->calib->intrinsics_d, trackingState, renderState_live);
}


Vector2i ITMMainEngine::GetImageSize(void) const
{
	return renderState_live->raycastImage->noDims;
}

void ITMMainEngine::GetImage(ITMUChar4Image *out, GetImageType getImageType, 
    const ITMPose * const pose, //!< used for InfiniTAM_IMAGE_FREECAMERA_... image type
    const ITMIntrinsics * const intrinsics  //!< used for InfiniTAM_IMAGE_FREECAMERA_... image type
    )
{
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

	case ITMMainEngine::InfiniTAM_IMAGE_SCENERAYCAST:
	{
		ORUtils::Image<Vector4u> *srcImage = renderState_live->raycastImage;
		out->ChangeDims(srcImage->noDims);
        
        out->SetFrom(srcImage, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);	
		break;
	}
	case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_SHADED:
	case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME:
	case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL:
	{
		ITMVisualisationEngine::RenderImageType type = ITMVisualisationEngine::RENDER_SHADED_GREYSCALE;
		if (getImageType == ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME) 
            type = ITMVisualisationEngine::RENDER_COLOUR_FROM_VOLUME;
		else if (getImageType == ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL) 
            type = ITMVisualisationEngine::RENDER_COLOUR_FROM_NORMAL;

		if (renderState_freeview == NULL) renderState_freeview = visualisationEngine->CreateRenderState(out->noDims);

		visualisationEngine->FindVisibleBlocks(pose, intrinsics, renderState_freeview);
		visualisationEngine->RenderImage(pose, intrinsics, renderState_freeview, renderState_freeview->raycastImage, type);

        out->SetFrom(renderState_freeview->raycastImage, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
		break;
	}
	case ITMMainEngine::InfiniTAM_IMAGE_UNKNOWN:
		break;
	};
}