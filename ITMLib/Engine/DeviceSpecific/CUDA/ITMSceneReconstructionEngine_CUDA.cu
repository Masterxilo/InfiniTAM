// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMSceneReconstructionEngine_CUDA.h"
#include "ITMCUDAUtils.h"
#include "../../DeviceAgnostic/ITMSceneReconstructionEngine.h"


using namespace ITMLib::Engine;

// device functions
template<class TVoxel, bool stopIntegratingAtMaxW, bool approximateIntegration>
__global__ void integrateIntoScene_device(TVoxel *localVBA, const ITMHashEntry *hashTable, const int *visibleEntryIDs,
    const Vector4u *rgb, Vector2i rgbImgSize, const float *depth, Vector2i depthImgSize, Matrix4f M_d, Matrix4f M_rgb, Vector4f projParams_d,
    Vector4f projParams_rgb, float voxelSize, float mu, int maxW)
{
    // one thread block for each voxel block
    Vector3i globalPos;
    int entryId = visibleEntryIDs[blockIdx.x];

    const ITMHashEntry &currentHashEntry = hashTable[entryId];

    if (currentHashEntry.ptr < 0) return;

    globalPos = currentHashEntry.pos.toInt() * SDF_BLOCK_SIZE;

    TVoxel *localVoxelBlock = &(localVBA[currentHashEntry.ptr * SDF_BLOCK_SIZE3]);

    // one thread for each voxel
    int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;
    integrateVoxel(x, y, z,
        stopIntegratingAtMaxW, globalPos, localVoxelBlock, voxelSize,
        M_d, projParams_d, M_rgb, projParams_rgb, mu, maxW, depth, depthImgSize, rgb, rgbImgSize);
}

__global__ void buildHashAllocAndVisibleType_device(uchar *entriesAllocType, uchar *entriesVisibleType, Vector4s *blockCoords, const float *depth,
    Matrix4f invM_d, Vector4f projParams_d, float mu, Vector2i _imgSize, float _voxelSize, ITMHashEntry *hashTable, float viewFrustum_min,
    float viewFrustum_max)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > _imgSize.x - 1 || y > _imgSize.y - 1) return;

    buildHashAllocAndVisibleTypePP(entriesAllocType, entriesVisibleType, x, y, blockCoords, depth, invM_d,
        projParams_d, mu, _imgSize, _voxelSize, hashTable, viewFrustum_min, viewFrustum_max);
}

__global__ void setToType3(uchar *entriesVisibleType, int *visibleEntryIDs, int noVisibleEntries)
{
    int entryId = threadIdx.x + blockIdx.x * blockDim.x;
    if (entryId > noVisibleEntries - 1) return;
    entriesVisibleType[visibleEntryIDs[entryId]] = VT_VISIBLE_PREVIOUS_AND_UNSTREAMED;
}


template<typename TVoxel>
__global__ void allocateVoxelBlocksList_device(
    typename ITMLocalVBA<TVoxel>::VoxelAllocationList *voxelAllocationList,
    ITMVoxelBlockHash::ExcessAllocationList *excessAllocationList, ITMHashEntry *hashTable, int noTotalEntries,
    uchar *entriesAllocType, uchar *entriesVisibleType, Vector4s *blockCoords)
{
    int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (targetIdx > noTotalEntries - 1) return;

    allocateVoxelBlock<TVoxel>(targetIdx,
        voxelAllocationList,
        excessAllocationList,
        hashTable,

        entriesAllocType,
        entriesVisibleType,
        blockCoords);
}

__global__ void buildVisibleList_device(ITMHashEntry *hashTable, int noTotalEntries,
    int *visibleEntryIDs, AllocationTempData *allocData, uchar *entriesVisibleType,
    Matrix4f M_d, Vector4f projParams_d, Vector2i depthImgSize, float voxelSize)
{
    int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (targetIdx > noTotalEntries - 1) return;

    __shared__ bool shouldPrefix;
    shouldPrefix = false;
    __syncthreads();

    bool visible = visibilityTestIfNeeded(
        targetIdx, entriesVisibleType, hashTable,
        M_d, projParams_d, depthImgSize, voxelSize
        );

    if (visible) shouldPrefix = true;

    __syncthreads();

    if (shouldPrefix)
    {
        int offset = computePrefixSum_device<int>(visible, &allocData->noVisibleEntries, blockDim.x * blockDim.y, threadIdx.x);
        if (offset != -1) visibleEntryIDs[offset] = targetIdx;
    }
}

// host methods

template<class TVoxel>
ITMSceneReconstructionEngine_CUDA<TVoxel,ITMVoxelBlockHash>::ITMSceneReconstructionEngine_CUDA(void) 
{
	ITMSafeCall(cudaMalloc((void**)&allocationTempData_device, sizeof(AllocationTempData)));
	ITMSafeCall(cudaMallocHost((void**)&allocationTempData_host, sizeof(AllocationTempData)));

	int noTotalEntries = ITMVoxelBlockHash::noTotalEntries;
	ITMSafeCall(cudaMalloc((void**)&entriesAllocType_device, noTotalEntries));
	ITMSafeCall(cudaMalloc((void**)&blockCoords_device, noTotalEntries * sizeof(Vector4s)));
}

template<class TVoxel>
ITMSceneReconstructionEngine_CUDA<TVoxel,ITMVoxelBlockHash>::~ITMSceneReconstructionEngine_CUDA(void) 
{
	ITMSafeCall(cudaFreeHost(allocationTempData_host));
	ITMSafeCall(cudaFree(allocationTempData_device));
	ITMSafeCall(cudaFree(entriesAllocType_device));
	ITMSafeCall(cudaFree(blockCoords_device));
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel,ITMVoxelBlockHash>::ResetScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene)
{
	int numBlocks = scene->index.getNumAllocatedVoxelBlocks();
	int blockSize = scene->index.getVoxelBlockSize();

    // Reset all voxels in all voxel blocks
	TVoxel *voxelBlocks_ptr = scene->localVBA.GetVoxelBlocks();
	memsetKernel<TVoxel>(voxelBlocks_ptr, TVoxel(), numBlocks * blockSize);

    // Reset voxel allocation list
    scene->localVBA.voxelAllocationList->Reset();

    // Reset hash entries
    ITMHashEntry tmpEntry = ITMHashEntry::createIllegalEntry();
	ITMHashEntry *hashEntry_ptr = scene->index.GetEntries();
	memsetKernel<ITMHashEntry>(hashEntry_ptr, tmpEntry, scene->index.noTotalEntries);

    // Reset excess allocation list
    scene->index.excessAllocationList->Reset();
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel, ITMVoxelBlockHash>::AllocateSceneFromDepth(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view, 
	const ITMTrackingState *trackingState, ITMRenderState *renderState)
{
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, invM_d;
	Vector4f projParams_d, invProjParams_d;

	M_d = trackingState->pose_d->GetM(); M_d.inv(invM_d);

	projParams_d = view->calib->intrinsics_d.projectionParamsSimple.all;
	invProjParams_d = projParams_d;
	invProjParams_d.x = 1.0f / invProjParams_d.x;
	invProjParams_d.y = 1.0f / invProjParams_d.y;

	float mu = scene->sceneParams->mu;

	float *depth = view->depth->GetData(MEMORYDEVICE_CUDA);
    auto voxelAllocationList = scene->localVBA.voxelAllocationList;
    auto excessAllocationList = scene->index.excessAllocationList;
	ITMHashEntry *hashTable = scene->index.GetEntries();

	int noTotalEntries = scene->index.noTotalEntries;

	int *visibleEntryIDs = renderState->GetVisibleEntryIDs();
	uchar *entriesVisibleType = renderState->GetEntriesVisibleType();

	dim3 cudaBlockSizeHV(16, 16);
	dim3 gridSizeHV((int)ceil((float)depthImgSize.x / (float)cudaBlockSizeHV.x), (int)ceil((float)depthImgSize.y / (float)cudaBlockSizeHV.y));

	dim3 cudaBlockSizeAL(256, 1);
	dim3 gridSizeAL((int)ceil((float)noTotalEntries / (float)cudaBlockSizeAL.x));

	dim3 cudaBlockSizeVS(256, 1);
	dim3 gridSizeVS((int)ceil((float)renderState->noVisibleEntries / (float)cudaBlockSizeVS.x));

	float oneOverVoxelSize = 1.0f / (voxelSize * SDF_BLOCK_SIZE);

	AllocationTempData *tempData = (AllocationTempData*)allocationTempData_host;
	tempData->noVisibleEntries = 0;
	ITMSafeCall(cudaMemcpyAsync(allocationTempData_device, tempData, sizeof(AllocationTempData), cudaMemcpyHostToDevice));

	ITMSafeCall(cudaMemsetAsync(entriesAllocType_device, 0, sizeof(unsigned char)* noTotalEntries));

    // Mark previously visible entries as such.
	if (gridSizeVS.x > 0) setToType3 << <gridSizeVS, cudaBlockSizeVS >> > (entriesVisibleType, visibleEntryIDs, renderState->noVisibleEntries);

    // Determine blocks currently visible in depth map and prepare allocation list
	buildHashAllocAndVisibleType_device << <gridSizeHV, cudaBlockSizeHV >> >(entriesAllocType_device, entriesVisibleType, 
		blockCoords_device, depth, invM_d, invProjParams_d, mu, depthImgSize, oneOverVoxelSize, hashTable,
		scene->sceneParams->viewFrustum_min, scene->sceneParams->viewFrustum_max);

    // Do allocation
    allocateVoxelBlocksList_device<TVoxel> << <gridSizeAL, cudaBlockSizeAL >> >(voxelAllocationList, excessAllocationList, hashTable,
		noTotalEntries, entriesAllocType_device, entriesVisibleType,
		blockCoords_device);

    // Visibility test for remaining blocks and count visible entries
	buildVisibleList_device<< <gridSizeAL, cudaBlockSizeAL >> >(hashTable, noTotalEntries, visibleEntryIDs,
			(AllocationTempData*)allocationTempData_device, entriesVisibleType, M_d, projParams_d, depthImgSize, voxelSize);
	

	ITMSafeCall(cudaMemcpy(tempData, allocationTempData_device, sizeof(AllocationTempData), cudaMemcpyDeviceToHost));
	renderState->noVisibleEntries = tempData->noVisibleEntries;
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel, ITMVoxelBlockHash>::IntegrateIntoScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view,
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

	float *depth = view->depth->GetData(MEMORYDEVICE_CUDA);
	Vector4u *rgb = view->rgb->GetData(MEMORYDEVICE_CUDA);
	TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
	ITMHashEntry *hashTable = scene->index.GetEntries();

	int *visibleEntryIDs = (int*)renderState->GetVisibleEntryIDs();

	dim3 cudaBlockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
	dim3 gridSize(renderState->noVisibleEntries);

#define integrateIntoScene_d(stopIntegratingAtMaxW, approximateIntegration) \
    integrateIntoScene_device<TVoxel, stopIntegratingAtMaxW, approximateIntegration> << <gridSize, cudaBlockSize >> >(\
    localVBA, hashTable, visibleEntryIDs,\
        rgb, rgbImgSize, depth, depthImgSize, M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW)

	if (scene->sceneParams->stopIntegratingAtMaxW)
		if (trackingState->requiresFullRendering)
            integrateIntoScene_d(true, false);
		else
            integrateIntoScene_d(true, true);
	else
		if (trackingState->requiresFullRendering)
            integrateIntoScene_d(false, false);
		else
            integrateIntoScene_d(false, true);
}


template class ITMLib::Engine::ITMSceneReconstructionEngine_CUDA<ITMVoxel, ITMVoxelIndex>;

