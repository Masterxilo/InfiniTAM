#include "ITMMainEngine.h"
ITMMainEngine::ITMMainEngine(const ITMRGBDCalib *calib)
{
    scene = new Scene();

    renderState_live = NULL;
	renderState_freeview = NULL; // will be created by the visualisation engine on demand
    trackingState = 0;

    view = new ITMView(calib); // will be allocated by the view builder
}

ITMMainEngine::~ITMMainEngine()
{
	delete renderState_live;
    delete renderState_freeview;

	delete scene;

	delete trackingState;
    delete view;
}

void ITMMainEngine::ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage)
{
    assert(rgbImage->noDims.area() > 1);
    assert(rawDepthImage->noDims.area() > 1);

    CURRENT_SCENE_SCOPE(scene);

	// prepare image and turn it into a depth image
    view->Update(rgbImage, rawDepthImage);

	// tracking

    // 1. raycast scene from current viewpoint 
    // to create point cloud for tracking
    cudaDeviceSynchronize();
    Vector2i imgSize_d = rawDepthImage->noDims;
    assert(imgSize_d.area() > 1);
    if (!trackingState) trackingState = new ITMTrackingState(imgSize_d);
    if (!renderState_live) renderState_live = new ITMRenderState(imgSize_d);
    assert(trackingState->pointCloud->locations->noDims == imgSize_d);
    assert(renderState_live->raycastResult->noDims == imgSize_d);

    CreateICPMaps(trackingState, &view->calib->intrinsics_d, renderState_live);
    
    // 2. align
    TrackCamera(trackingState, view); 

	// fusion
    FuseView(view, trackingState->pose_d->GetM());
}
void ITMMainEngine::GetImage(
    ITMUChar4Image * const out,
    const ITMPose * const pose, 
    const ITMIntrinsics * const intrinsics,
    std::string shader
    )
{
	if (!renderState_freeview) renderState_freeview = new ITMRenderState(out->noDims);

    assert(renderState_freeview->raycastResult->noDims == out->noDims);

    CURRENT_SCENE_SCOPE(scene);
	RenderImage(pose, intrinsics, renderState_freeview, out, shader);
    assert(out->dirtyGPU);
}