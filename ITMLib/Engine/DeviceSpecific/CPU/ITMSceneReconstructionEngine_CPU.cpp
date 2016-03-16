// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMSceneReconstructionEngine_CPU.h"
#include "../../DeviceAgnostic/ITMSceneReconstructionEngine.h"
#include "../../../Objects/ITMRenderState_VH.h"

using namespace ITMLib::Engine;

template<class TVoxel>
ITMSceneReconstructionEngine_CPU<TVoxel,ITMVoxelBlockHash>::ITMSceneReconstructionEngine_CPU(void) 
{
	int noTotalEntries = ITMVoxelBlockHash::noTotalEntries;
	entriesAllocType = new ORUtils::MemoryBlock<unsigned char>(noTotalEntries, MEMORYDEVICE_CPU);
	blockCoords = new ORUtils::MemoryBlock<Vector4s>(noTotalEntries, MEMORYDEVICE_CPU);
}

template<class TVoxel>
ITMSceneReconstructionEngine_CPU<TVoxel,ITMVoxelBlockHash>::~ITMSceneReconstructionEngine_CPU(void) 
{
	delete entriesAllocType;
	delete blockCoords;
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel,ITMVoxelBlockHash>::ResetScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene)
{
	int numBlocks = scene->index.getNumAllocatedVoxelBlocks();
	int blockSize = scene->index.getVoxelBlockSize();

    // Reset all voxels in all voxel blocks
	TVoxel *voxelBlocks_ptr = scene->localVBA.GetVoxelBlocks();
	for (int i = 0; i < numBlocks * blockSize; ++i) voxelBlocks_ptr[i] = TVoxel();

    // Reset voxel allocation list
    scene->localVBA.voxelAllocationList->Reset();

    // Reset hash entries
    ITMHashEntry tmpEntry = ITMHashEntry::createIllegalEntry();
	ITMHashEntry *hashEntry_ptr = scene->index.GetEntries();
	for (int i = 0; i < scene->index.noTotalEntries; ++i) hashEntry_ptr[i] = tmpEntry;

    // Reset excess allocation list
    scene->index.excessAllocationList->Reset();
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel, ITMVoxelBlockHash>::IntegrateIntoScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState)
{
    // [[ rename parameters
	Vector2i rgbImgSize = view->rgb->noDims;
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;
	ITMRenderState_VH *renderState_vh = (ITMRenderState_VH*)renderState;

    // Prepare camera calibration
	Matrix4f M_d, M_rgb;
	M_d = trackingState->pose_d->GetM();
	if (TVoxel::hasColorInformation) M_rgb = view->calib->trafo_rgb_to_depth.calib_inv * M_d;

	Vector4f projParams_d, projParams_rgb;
	projParams_d = view->calib->intrinsics_d.projectionParamsSimple.all;
	projParams_rgb = view->calib->intrinsics_rgb.projectionParamsSimple.all;

    // TSDF update parameters
	float mu = scene->sceneParams->mu;
    int maxW = scene->sceneParams->maxW;

    // Sensor data
	float *depth = view->depth->GetData(MEMORYDEVICE_CPU);
	Vector4u *rgb = view->rgb->GetData(MEMORYDEVICE_CPU);
	// TSDF data
    TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
	ITMHashEntry *hashTable = scene->index.GetEntries();

    // Visibility information
	int *visibleEntryIds = renderState_vh->GetVisibleEntryIDs();
	int noVisibleEntries = renderState_vh->noVisibleEntries;

	bool stopIntegratingAtMaxW = scene->sceneParams->stopIntegratingAtMaxW;
	//bool approximateIntegration = !trackingState->requiresFullRendering;
    // ]]

#ifdef WITH_OPENMP
	#pragma omp parallel for
#endif
    // for each voxel block
	for (int entryId = 0; entryId < noVisibleEntries; entryId++)
	{
		Vector3i globalPos;
		const ITMHashEntry &currentHashEntry = hashTable[visibleEntryIds[entryId]];

		if (currentHashEntry.ptr < 0) continue;

        // voxel block base position, in voxel coordinates
        globalPos = currentHashEntry.pos.toInt();
		globalPos *= SDF_BLOCK_SIZE;

		TVoxel *localVoxelBlock = &(localVBA[currentHashEntry.ptr * (SDF_BLOCK_SIZE3)]);

        // for each voxel, in local voxel coordinates
		for (int z = 0; z < SDF_BLOCK_SIZE; z++) for (int y = 0; y < SDF_BLOCK_SIZE; y++) for (int x = 0; x < SDF_BLOCK_SIZE; x++)
		{
            integrateVoxel(x, y, z,
                stopIntegratingAtMaxW, globalPos, localVoxelBlock, voxelSize,
                M_d, projParams_d, M_rgb, projParams_rgb, mu, maxW, depth, depthImgSize, rgb, rgbImgSize);
		}
	}
}


/// allocation & visible list update
template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel, ITMVoxelBlockHash>::AllocateSceneFromDepth(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view,
	const ITMTrackingState *trackingState,
    const ITMRenderState *renderState,
    bool onlyUpdateVisibleList)
{
	const float voxelSize = scene->sceneParams->voxelSize;

    // Cache camera calibration
	Matrix4f M_d, invM_d;
	Vector4f projParams_d, invProjParams_d;
	M_d = trackingState->pose_d->GetM(); M_d.inv(invM_d);
	projParams_d = view->calib->intrinsics_d.projectionParamsSimple.all;
    invProjParams_d = view->calib->intrinsics_d.getInverseProjParams();

	float mu = scene->sceneParams->mu;

	ITMRenderState_VH *renderState_vh = (ITMRenderState_VH*)renderState;

    // Depth image for visibility detection
    Vector2i depthImgSize = view->depth->noDims;
	float *depth = view->depth->GetData(MEMORYDEVICE_CPU);

	ITMHashEntry *hashTable = scene->index.GetEntries();

    uchar *entriesVisibleType = renderState_vh->GetEntriesVisibleType();

    ITMLocalVBA<TVoxel>::VoxelAllocationList *voxelAllocationList = scene->localVBA.voxelAllocationList;
    ITMVoxelBlockHash::ExcessAllocationList *excessAllocationList = scene->index.excessAllocationList;
	uchar *entriesAllocType = this->entriesAllocType->GetData(MEMORYDEVICE_CPU);

    // Of blocks that need allocation
	Vector4s *blockCoords = this->blockCoords->GetData(MEMORYDEVICE_CPU);
	int noTotalEntries = scene->index.noTotalEntries;


	float oneOverVoxelSize = 1.0f / (voxelSize * SDF_BLOCK_SIZE);

    AllocationTempData allocData;
    allocData.noVisibleEntries = 0;

	memset(entriesAllocType, 0, noTotalEntries);

    // Collect visible entries
    int *visibleEntryIDs = renderState_vh->GetVisibleEntryIDs();
    
    // Mark previously visible entries as such.
    for (int i = 0; i < renderState_vh->noVisibleEntries; i++)
        entriesVisibleType[visibleEntryIDs[i]] = VT_VISIBLE_PREVIOUS_AND_UNSTREAMED;

	//build hashVisibility
#ifdef WITH_OPENMP
	#pragma omp parallel for
#endif
	for (int locId = 0; locId < depthImgSize.x*depthImgSize.y; locId++)
	{
		int y = locId / depthImgSize.x;
		int x = locId - y * depthImgSize.x;
		buildHashAllocAndVisibleTypePP(entriesAllocType, entriesVisibleType, x, y, blockCoords, depth, invM_d,
			invProjParams_d, mu, depthImgSize, oneOverVoxelSize, hashTable, scene->sceneParams->viewFrustum_min,
			scene->sceneParams->viewFrustum_max);
	}

    // Use results
	if (!onlyUpdateVisibleList)
	{
		// allocate
		for (int targetIdx = 0; targetIdx < noTotalEntries; targetIdx++)
		{
            allocateVoxelBlock<TVoxel>(targetIdx,
                voxelAllocationList,
                excessAllocationList,
                hashTable,
                &allocData,

                entriesAllocType,
                entriesVisibleType,
                blockCoords);
		}
	}

	//build visible list
	for (int targetIdx = 0; targetIdx < noTotalEntries; targetIdx++)
	{
        if (visibilityTestIfNeeded(
            targetIdx, entriesVisibleType, hashTable,
            M_d, projParams_d, depthImgSize, voxelSize))
		{	
            visibleEntryIDs[allocData.noVisibleEntries++] = targetIdx;
		}
	}

    // Copy back AllocationTempData
    renderState_vh->noVisibleEntries = allocData.noVisibleEntries;
}

template<class TVoxel>
ITMSceneReconstructionEngine_CPU<TVoxel,ITMPlainVoxelArray>::ITMSceneReconstructionEngine_CPU(void) 
{}

template<class TVoxel>
ITMSceneReconstructionEngine_CPU<TVoxel,ITMPlainVoxelArray>::~ITMSceneReconstructionEngine_CPU(void) 
{}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel,ITMPlainVoxelArray>::ResetScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene)
{
	int numBlocks = scene->index.getNumAllocatedVoxelBlocks();
	int blockSize = scene->index.getVoxelBlockSize();

	TVoxel *voxelBlocks_ptr = scene->localVBA.GetVoxelBlocks();
	for (int i = 0; i < numBlocks * blockSize; ++i) voxelBlocks_ptr[i] = TVoxel();
	int *vbaAllocationList_ptr = scene->localVBA.GetAllocationList();
	for (int i = 0; i < numBlocks; ++i) vbaAllocationList_ptr[i] = i;
	scene->localVBA.lastFreeBlockId = numBlocks - 1;
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel, ITMPlainVoxelArray>::AllocateSceneFromDepth(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState, bool onlyUpdateVisibleList)
{}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel, ITMPlainVoxelArray>::IntegrateIntoScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState)
{
	Vector2i rgbImgSize = view->rgb->noDims;
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, M_rgb;
	Vector4f projParams_d, projParams_rgb;

	M_d = trackingState->pose_d->GetM();
	if (TVoxel::hasColorInformation) M_rgb = view->calib->trafo_rgb_to_depth.calib_inv * M_d;

	projParams_d = view->calib->intrinsics_d.projectionParamsSimple.all;
	projParams_rgb = view->calib->intrinsics_rgb.projectionParamsSimple.all;

	float mu = scene->sceneParams->mu; int maxW = scene->sceneParams->maxW;

	float *depth = view->depth->GetData(MEMORYDEVICE_CPU);
	Vector4u *rgb = view->rgb->GetData(MEMORYDEVICE_CPU);
	TVoxel *voxelArray = scene->localVBA.GetVoxelBlocks();

	const ITMPlainVoxelArray::IndexData *arrayInfo = scene->index.getIndexData();

	bool stopIntegratingAtMaxW = scene->sceneParams->stopIntegratingAtMaxW;
	//bool approximateIntegration = !trackingState->requiresFullRendering;

#ifdef WITH_OPENMP
	#pragma omp parallel for
#endif
	for (int locId = 0; locId < scene->index.getVolumeSize().x*scene->index.getVolumeSize().y*scene->index.getVolumeSize().z; ++locId)
	{
		int z = locId / (scene->index.getVolumeSize().x*scene->index.getVolumeSize().y);
		int tmp = locId - z * scene->index.getVolumeSize().x*scene->index.getVolumeSize().y;
		int y = tmp / scene->index.getVolumeSize().x;
		int x = tmp - y * scene->index.getVolumeSize().x;
		Vector4f pt_model;

		if (stopIntegratingAtMaxW) if (voxelArray[locId].w_depth == maxW) continue;
		//if (approximateIntegration) if (voxelArray[locId].w_depth != 0) continue;

		pt_model.x = (float)(x + arrayInfo->offset.x) * voxelSize;
		pt_model.y = (float)(y + arrayInfo->offset.y) * voxelSize;
		pt_model.z = (float)(z + arrayInfo->offset.z) * voxelSize;
		pt_model.w = 1.0f;

		ComputeUpdatedVoxelInfo<TVoxel::hasColorInformation,TVoxel>::compute(voxelArray[locId], pt_model, M_d, projParams_d, M_rgb, projParams_rgb, mu, maxW, 
			depth, depthImgSize, rgb, rgbImgSize);
	}
}

template class ITMLib::Engine::ITMSceneReconstructionEngine_CPU<ITMVoxel, ITMVoxelIndex>;
