// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMMainEngine.h"

using namespace ITMLib::Engine;

KERNEL buildASceneA() {
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

KERNEL buildASceneB() {
    ITMVoxel* v = Scene::getCurrentSceneVoxel(
        Vector3i(blockIdx.x,
        blockIdx.y,
        blockIdx.z) * SDF_BLOCK_SIZE
        +
        Vector3i(threadIdx.x,
        threadIdx.y,
        threadIdx.z));
    float z = (blockIdx.z *  SDF_BLOCK_SIZE + threadIdx.z) * voxelSize;
    assert(v);
    float eta = (SDF_BLOCK_SIZE / 2)*voxelSize - z;
    v->setSDF(MAX(MIN(1.0f, eta / mu), -1.f));
    // solid wall at z == (SDF_BLOCK_SIZE / 2), negative at greater z values
    // notice that we normalize with mu and compute position in world space
}

ITMMainEngine::ITMMainEngine(const ITMRGBDCalib *calib, Vector2i imgSize_rgb, Vector2i imgSize_d)
{
    scene = new Scene();
    CURRENT_SCENE_SCOPE(scene);
    //buildASceneA << <dim3(10, 10, 1), 1 >> >();
    cudaDeviceSynchronize();
    Scene::performCurrentSceneAllocations();
    //buildASceneB << <dim3(10, 10, 1), dim3(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE) >> >();
    //Scene::getCurrentScene()->doForEachAllocatedVoxel<EachV>();

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
    tracker->TrackCamera(trackingState, view); // affects relative orientation of blocks to camera because the pose is not left unchanged even on the first try (why? on what basis is the camera moved?

	// fusion
    ITMSceneReconstructionEngine_ProcessFrame(view, trackingState->pose_d->GetM());

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