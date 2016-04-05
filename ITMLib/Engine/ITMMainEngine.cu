#include "ITMMainEngine.h"


void buildSphereScene(const float radiusInWorldCoordinates);
ITMMainEngine::ITMMainEngine(const ITMRGBDCalib *calib)
{
    scene = new Scene();
    CURRENT_SCENE_SCOPE(scene);
   // buildSphereScene(10 * voxelBlockSize);

    renderState_live = NULL;
	renderState_freeview = NULL; // will be created by the visualisation engine on demand

    view = new ITMView(calib); // will be allocated by the view builder
}

ITMMainEngine::~ITMMainEngine()
{
	delete renderState_live;
    delete renderState_freeview;

	delete scene;

    delete view;
}

void ITMMainEngine::ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage)
{
    assert(rgbImage->noDims.area() > 1);
    assert(rawDepthImage->noDims.area() > 1);

    CURRENT_SCENE_SCOPE(scene);
    currentView = view;

    currentView->ChangeImages(rgbImage, rawDepthImage);
    ImprovePose();
    Fuse();

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
    cudaDeviceSynchronize();
    assert(out->dirtyGPU);
}