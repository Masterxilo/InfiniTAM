// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMMainEngine.h"

using namespace ITMLib::Engine;

ITMMainEngine::ITMMainEngine(const ITMLibSettings *settings, const ITMRGBDCalib *calib, Vector2i imgSize_rgb, Vector2i imgSize_d)
{
	this->settings = settings;

	scene = new ITMScene(&(settings->sceneParams));

	lowLevelEngine = new ITMLowLevelEngine();
	viewBuilder = new ITMViewBuilder(calib);
	visualisationEngine = new ITMVisualisationEngine(scene);

    Vector2i trackedImageSize = imgSize_d;

	renderState_live = visualisationEngine->CreateRenderState(trackedImageSize);
	renderState_freeview = NULL; //will be created by the visualisation engine on demand

    sceneRecoEngine = new ITMSceneReconstructionEngine();
    ResetScene();

    tracker = new ITMDepthTracker(
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
    delete renderState_freeview;

	delete scene;

    delete sceneRecoEngine;

	delete tracker;

	delete lowLevelEngine;
	delete viewBuilder;

	delete trackingState;
	if (view != NULL) delete view;

	delete visualisationEngine;
}

#include <fstream>
#include <map>
void ITMMainEngine::ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage)
{
	// prepare image and turn it into a depth image
	viewBuilder->UpdateView(&view, rgbImage, rawDepthImage);

	// tracking
    tracker->TrackCamera(trackingState, view);

	// fusion
    sceneRecoEngine->ProcessFrame(view, trackingState, scene);
    if (0)
    {
        // create scene from dump [[
         std::map<VoxelBlockPos, ITMVoxelBlock> vbs; 
        {
            std::ifstream in("scenedump1.txt");
            while (1) {
                ITMVoxelBlock b;
                in >> b.pos.x >> b.pos.y >> b.pos.z;
                if (in.fail()) break;

                int i = 0;
                for (auto& v : b.blockVoxels) {
                    float s; in >> s;
                    v.setSDF(s);
                }

                vbs[b.pos] = b;
            }

            in.close();
        }

    {
        std::ifstream in("scenehashdump1.txt");
        while (1) {
            int j;
            ITMHashEntry he;
            in >> j >> he.pos.x >> he.pos.y >> he.pos.z >> he.offset >> he.ptr;
            if (in.fail()) break;

            cudaMemcpy(scene->index.GetEntries() + j, &he,
                sizeof(ITMHashEntry), cudaMemcpyHostToDevice);


            cudaMemcpy(scene->localVBA.GetVoxelBlocks() + he.ptr, &vbs[he.pos],
                sizeof(ITMVoxelBlock), cudaMemcpyHostToDevice);
        }
        in.close();
    }

    }
    // ]]


    // raycast scene from current viewpoint 
    // to create point cloud for tracking
    visualisationEngine->CreateICPMaps(&view->calib->intrinsics_d, trackingState, renderState_live);
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

        if (0) {
            // dump hash index:
            {
                std::ofstream of("scenehashdump1.txt");

                ITMHashEntry *hes = (ITMHashEntry *)malloc(SDF_GLOBAL_BLOCK_NUM * sizeof(ITMHashEntry));

                cudaMemcpy(hes,
                    scene->index.GetEntries(),
                    SDF_GLOBAL_BLOCK_NUM * sizeof(ITMHashEntry),
                    cudaMemcpyDeviceToHost);


                for (int j = SDF_GLOBAL_BLOCK_NUM - 1; j >= 0; j--) {
                    ITMHashEntry& he = hes[j];
                    if (!he.isAllocated()) continue;

                    of << j << " " << he.pos.x << " " << he.pos.y << " " << he.pos.z << " " << he.offset << " " << he.ptr << std::endl;
                }
                of.close();
                exit(0);

                exit(0);
            }
            // dump scene [[
            std::ofstream of("scenedump1.txt");
            for (int j = SDF_LOCAL_BLOCK_NUM - 1; j >= 0; j--) {
                ITMVoxelBlock b;
                cudaMemcpy(&b, scene->localVBA.GetVoxelBlocks() + j,
                    sizeof(ITMVoxelBlock), cudaMemcpyDeviceToHost);
                if (b.pos == INVALID_VOXEL_BLOCK_POS) continue;

                of << b.pos.x << " " << b.pos.y << " " << b.pos.z << " " << std::endl;
                int i = 0;
                for (auto& v : b.blockVoxels) {
                    of << v.getSDF() << " ";
                }
                of << std::endl;
            }
            of.close();
            exit(0);
            // ]]
        }

		visualisationEngine->RenderImage(pose, intrinsics, renderState_freeview, out, type);
        out->UpdateHostFromDevice();
		break;
	}
	case ITMMainEngine::InfiniTAM_IMAGE_UNKNOWN:
		break;
	};
}