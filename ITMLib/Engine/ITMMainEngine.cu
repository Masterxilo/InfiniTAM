#include "ITMMainEngine.h"
using namespace ITMLib::Engine;

ITMMainEngine::ITMMainEngine(const ITMRGBDCalib *calib, Vector2i imgSize_rgb, Vector2i imgSize_d)
{
    scene = new Scene();
    renderState_live = new ITMRenderState(imgSize_d);
	renderState_freeview = NULL; // will be created by the visualisation engine on demand

    trackingState = new ITMTrackingState(imgSize_d);

    view = new ITMView(calib, imgSize_rgb, imgSize_d); // will be allocated by the view builder
}

ITMMainEngine::~ITMMainEngine()
{
	delete renderState_live;
    delete renderState_freeview;

	delete scene;

	delete trackingState;
    delete view;
}

#include "fileutils.h"

using namespace ORUtils;
template<typename T>
static bool checkImageSame(Image<T>* a_, Image<T>* b_) {
    T* a = a_->GetData(MEMORYDEVICE_CPU);
    T* b = b_->GetData(MEMORYDEVICE_CPU);
#define failifnot(x) if (!(x)) return false;
    failifnot(a_->dataSize == b_->dataSize);
    failifnot(a_->noDims == b_->noDims);
    int s = a_->dataSize;
    while (s--) {
        if (*a != *b) {
            failifnot(false);
        }
        a++;
        b++;
    }
    return true;
}

/// Must exist on cpu
template<typename T>
static void assertImageSame(Image<T>* a_, Image<T>* b_) {
    assert(checkImageSame(a_, b_));
}

int jj = 0;

#include "ITMDepthTrackerOld.h"
void ITMMainEngine::ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage)
{
    CURRENT_SCENE_SCOPE(scene);

	// prepare image and turn it into a depth image
    view->Update(rgbImage, rawDepthImage);

	// tracking
    //ITMDepthTracker::

    // raycast scene from current viewpoint 
    // to create point cloud for tracking
    cudaDeviceSynchronize();
    ITMVisualisationEngine::CreateICPMaps(trackingState, &view->calib->intrinsics_d, renderState_live);

    // store input
    dump::store(trackingState, "Tests/Tracker/trackingState");
    dump::store(view, "Tests/Tracker/view");

    if (jj++) {
        auto trackingState_ = dump::load<ITMTrackingState>("Tests/Tracker/trackingState");
        assertImageSame(
            trackingState->pointCloud->locations,
            trackingState_->pointCloud->locations);

        auto view_ = dump::load<ITMView>("Tests/Tracker/view");


    }

    auto dt = new ITMDepthTracker(trackingState->pointCloud->locations->noDims);
    dt->TrackCamera(trackingState, view); // affects relative orientation of blocks to camera because the pose is not left unchanged even on the first try (why? on what basis is the camera moved?
    // store output  
    //dump::store(trackingState->pose_d, "Tests/Tracker/out.pose_d");

    auto expectedPose = dump::load<ITMPose>("Tests/Tracker/out.pose_d");

	// fusion
    ITMSceneReconstructionEngine_ProcessFrame(view, trackingState->pose_d->GetM());

}
#include "fileutils.h"
void ITMMainEngine::GetImage(
    ITMUChar4Image * const out,
    const GetImageType getImageType, 
    const ITMPose * const pose, 
    const ITMIntrinsics * const intrinsics 
    )
{
	out->Clear();

	switch (getImageType)
	{
	case ITMMainEngine::InfiniTAM_IMAGE_ORIGINAL_RGB:
        out->ChangeDims(view->rgb->noDims);
        out->SetFrom(view->rgb, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
        break;

	case ITMMainEngine::InfiniTAM_IMAGE_ORIGINAL_DEPTH:
        out->ChangeDims(view->depth->noDims);
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
            renderState_freeview = new ITMRenderState(out->noDims);

        assert(renderState_freeview->raycastResult->noDims == out->noDims);

        CURRENT_SCENE_SCOPE(scene);
		ITMVisualisationEngine::RenderImage(pose, intrinsics, renderState_freeview, out, type);

        png::SaveImageToFile(out, "out.png");

		break;
	}
	case ITMMainEngine::InfiniTAM_IMAGE_UNKNOWN:
		break;
	};
}