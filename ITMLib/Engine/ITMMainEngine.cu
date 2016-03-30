// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMMainEngine.h"

using namespace ITMLib::Engine;

static KERNEL buildASceneA() {
    Scene::requestCurrentSceneVoxelBlockAllocation(
        VoxelBlockPos(blockIdx.x,
        blockIdx.y,
        blockIdx.z));
}

struct EachV {
    static GPU_ONLY void process(const ITMVoxelBlock* vb, ITMVoxel* v, const Vector3i localPos) {

        float z = (threadIdx.z) * voxelSize;
        assert(v);
        float eta = (SDF_BLOCK_SIZE / 2)*voxelSize - z;
        v->setSDF(MAX(MIN(1.0f, eta / mu), -1.f));

    }
};

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

void ITMMainEngine::ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage)
{
    CURRENT_SCENE_SCOPE(scene);

	// prepare image and turn it into a depth image
    view->Update(rgbImage, rawDepthImage);

	// tracking
    ITMDepthTracker::TrackCamera(trackingState, view); // affects relative orientation of blocks to camera because the pose is not left unchanged even on the first try (why? on what basis is the camera moved?

	// fusion
    ITMSceneReconstructionEngine_ProcessFrame(view, trackingState->pose_d->GetM());

    // raycast scene from current viewpoint 
    // to create point cloud for tracking
    ITMVisualisationEngine::CreateICPMaps(trackingState, &view->calib->intrinsics_d, renderState_live);
}
#include "fileutils.h"
void ITMMainEngine::GetImage(
    ITMUChar4Image * const out,
    const GetImageType getImageType, 
    const ITMPose * const pose, 
    const ITMIntrinsics * const intrinsics 
    )
{
    assert(out->isAllocated_CPU() && out->isAllocated_CUDA());
	

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
            renderState_freeview = new ITMRenderState(out->noDims);

        assert(renderState_freeview->raycastResult->noDims == out->noDims);

        CURRENT_SCENE_SCOPE(scene);
		ITMVisualisationEngine::RenderImage(pose, intrinsics, renderState_freeview, out, type);
        out->UpdateHostFromDevice();

        png::SaveImageToFile(out, "out.png");

		break;
	}
	case ITMMainEngine::InfiniTAM_IMAGE_UNKNOWN:
		break;
	};
}