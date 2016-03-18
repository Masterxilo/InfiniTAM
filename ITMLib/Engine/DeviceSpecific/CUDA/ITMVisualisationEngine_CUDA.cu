// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMVisualisationEngine_CUDA.h"
#include "ITMCUDAUtils.h"
#include "../../DeviceAgnostic/ITMRepresentationAccess.h"
#include "../../DeviceAgnostic/ITMVisualisationEngine.h"
#include "../../DeviceAgnostic/ITMSceneReconstructionEngine.h"

using namespace ITMLib::Engine;

inline dim3 getGridSize(dim3 taskSize, dim3 blockSize)
{
	return dim3((taskSize.x + blockSize.x - 1) / blockSize.x, (taskSize.y + blockSize.y - 1) / blockSize.y, (taskSize.z + blockSize.z - 1) / blockSize.z);
}

inline dim3 getGridSize(Vector2i taskSize, dim3 blockSize) { return getGridSize(dim3(taskSize.x, taskSize.y), blockSize); }

//device implementations

__global__ void buildVisibleList_device(const ITMHashEntry *hashTable, int noTotalEntries,
    int *visibleEntryIDs, int *noVisibleEntries, uchar *entriesVisibleType, Matrix4f M, Vector4f projParams, Vector2i imgSize, float voxelSize)
{
    int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (targetIdx > noTotalEntries - 1) return;

    __shared__ bool shouldPrefix;

    unsigned char hashVisibleType = 0; //entriesVisibleType[targetIdx];
    const ITMHashEntry &hashEntry = hashTable[targetIdx];

    shouldPrefix = false;
    __syncthreads();

    if (hashEntry.ptr >= 0)
    {
        shouldPrefix = true;

        bool isVisible;
        checkBlockVisibility(isVisible, hashEntry.pos, M, projParams, voxelSize, imgSize);

        hashVisibleType = isVisible;
    }

    if (hashVisibleType > 0) shouldPrefix = true;

    __syncthreads();

    if (shouldPrefix)
    {
        int offset = computePrefixSum_device<int>(hashVisibleType > 0, noVisibleEntries, blockDim.x * blockDim.y, threadIdx.x);
        if (offset != -1) visibleEntryIDs[offset] = targetIdx;
    }
}

__global__ void projectAndSplitBlocks_device(const ITMHashEntry *hashEntries, const int *visibleEntryIDs, int noVisibleEntries,
    const Matrix4f pose_M, const Vector4f intrinsics, const Vector2i imgSize, float voxelSize, RenderingBlock *renderingBlocks,
    uint *noTotalBlocks)
{
    int in_offset = threadIdx.x + blockDim.x * blockIdx.x;

    const ITMHashEntry & blockData(hashEntries[visibleEntryIDs[in_offset]]);

    Vector2i upperLeft, lowerRight;
    Vector2f zRange;
    bool validProjection = false;
    if (in_offset < noVisibleEntries) if (blockData.ptr >= 0)
        validProjection = ProjectSingleBlock(blockData.pos, pose_M, intrinsics, imgSize, voxelSize, upperLeft, lowerRight, zRange);

    Vector2i requiredRenderingBlocks(ceilf((float)(lowerRight.x - upperLeft.x + 1) / renderingBlockSizeX),
        ceilf((float)(lowerRight.y - upperLeft.y + 1) / renderingBlockSizeY));

    size_t requiredNumBlocks = requiredRenderingBlocks.x * requiredRenderingBlocks.y;
    if (!validProjection) requiredNumBlocks = 0;

    int out_offset = computePrefixSum_device<uint>(requiredNumBlocks, noTotalBlocks, blockDim.x, threadIdx.x);
    if (!validProjection) return;
    if ((out_offset == -1) || (out_offset + requiredNumBlocks > MAX_RENDERING_BLOCKS)) return;

    CreateRenderingBlocks(renderingBlocks, out_offset, upperLeft, lowerRight, zRange);
}

__global__ void fillBlocks_device(const uint *noTotalBlocks, const RenderingBlock *renderingBlocks,
    Vector2i imgSize, 
    Vector2f *minmaxData //!< [out]
    )
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    int block = blockIdx.x * 4 + blockIdx.y;
    if (block >= *noTotalBlocks) return;

    const RenderingBlock & b(renderingBlocks[block]);
    int xpos = b.upperLeft.x + x;
    if (xpos > b.lowerRight.x) return;
    int ypos = b.upperLeft.y + y;
    if (ypos > b.lowerRight.y) return;

    Vector2f & pixel(minmaxData[xpos + ypos*imgSize.x]);
    atomicMin(&pixel.x, b.zRange.x); atomicMax(&pixel.y, b.zRange.y);
}

__global__ void genericRaycast_device(Vector4f *out_ptsRay, const ITMVoxel *voxelData, const typename ITMVoxelIndex::IndexData *voxelIndex,
    Vector2i imgSize, Matrix4f invM, Vector4f invProjParams, float oneOverVoxelSize, const Vector2f *minmaximg, float mu)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

    if (x >= imgSize.x || y >= imgSize.y) return;

    int locId = x + y * imgSize.x;
    int locId2 = (int)floor((float)x / minmaximg_subsample) + (int)floor((float)y / minmaximg_subsample) * imgSize.x;

    castRay(out_ptsRay[locId], x, y, voxelData, voxelIndex, invM, invProjParams, oneOverVoxelSize, mu, minmaximg[locId2]);
}

__global__ void renderICP_device(Vector4u *outRendering, Vector4f *pointsMap, Vector4f *normalsMap, const Vector4f *pointsRay,
    float voxelSize, Vector2i imgSize, Vector3f lightSource)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

    if (x >= imgSize.x || y >= imgSize.y) return;

    processPixelICP<true>(outRendering, pointsMap, normalsMap, pointsRay, imgSize, x, y, voxelSize, lightSource);
}

/*
renderGrey_device, processPixelGrey
renderColourFromNormal_device, processPixelNormal
renderColour_device, processPixelColour
*/
#define RENDER_PROCESS_PIXEL(RENDERFUN, PROCESSPIXELFUN) \
__global__ void RENDERFUN ## _device(Vector4u *outRendering, const Vector4f *ptsRay, const ITMVoxel *voxelData,\
    const typename ITMVoxelIndex::IndexData *voxelIndex, Vector2i imgSize, Vector3f lightSource) { \
    int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);\
    if (x >= imgSize.x || y >= imgSize.y) return;\
    int locId = pixelLocId(x, y, imgSize);\
    Vector4f ptRay = ptsRay[locId];\
    PROCESSPIXELFUN(outRendering[locId], ptRay.toVector3(), ptRay.w > 0, voxelData, voxelIndex, lightSource);\
}

RENDER_PROCESS_PIXEL(renderGrey, processPixelGrey)
RENDER_PROCESS_PIXEL(renderColourFromNormal, processPixelNormal)
RENDER_PROCESS_PIXEL(renderColour, processPixelColour)

// class implementation
ITMVisualisationEngine_CUDA::ITMVisualisationEngine_CUDA(ITMScene *scene) : ITMVisualisationEngine(scene)
{
	ITMSafeCall(cudaMalloc((void**)&renderingBlockList_device, sizeof(RenderingBlock) * MAX_RENDERING_BLOCKS));
	ITMSafeCall(cudaMalloc((void**)&noTotalBlocks_device, sizeof(uint)));
	ITMSafeCall(cudaMalloc((void**)&noVisibleEntries_device, sizeof(uint)));
}

ITMVisualisationEngine_CUDA::~ITMVisualisationEngine_CUDA(void)
{
	ITMSafeCall(cudaFree(noTotalBlocks_device));
	ITMSafeCall(cudaFree(renderingBlockList_device));
	ITMSafeCall(cudaFree(noVisibleEntries_device));
}

ITMRenderState* ITMVisualisationEngine_CUDA::CreateRenderState(const Vector2i & imgSize) const
{
	return new ITMRenderState(
		ITMVoxelBlockHash::noTotalEntries,
        imgSize,
        this->scene->sceneParams->viewFrustum_min, this->scene->sceneParams->viewFrustum_max, MEMORYDEVICE_CUDA
	);
}

void ITMVisualisationEngine_CUDA::FindVisibleBlocks(const ITMPose *pose, const ITMIntrinsics *intrinsics, ITMRenderState *renderState) const
{
	const ITMHashEntry *hashTable = this->scene->index.GetEntries();
	int noTotalEntries = this->scene->index.noTotalEntries;
	float voxelSize = this->scene->sceneParams->voxelSize;
	Vector2i imgSize = renderState->renderingRangeImage->noDims;

	Matrix4f M = pose->GetM();
	Vector4f projParams = intrinsics->projectionParamsSimple.all;


    // Brute-force full visibility test
	ITMSafeCall(cudaMemset(noVisibleEntries_device, 0, sizeof(int)));

	dim3 cudaBlockSizeAL(256, 1);
	dim3 gridSizeAL((int)ceil((float)noTotalEntries / (float)cudaBlockSizeAL.x));
	buildVisibleList_device << <gridSizeAL, cudaBlockSizeAL >> >(hashTable, noTotalEntries,
        renderState->GetVisibleEntryIDs(), noVisibleEntries_device, renderState->GetEntriesVisibleType(), M, projParams,
		imgSize, voxelSize);

    ITMSafeCall(cudaMemcpy(&renderState->noVisibleEntries, noVisibleEntries_device, sizeof(int), cudaMemcpyDeviceToHost));
}

void ITMVisualisationEngine_CUDA::CreateExpectedDepths(const ITMPose *pose, const ITMIntrinsics *intrinsics, 
	ITMRenderState *renderState) const
{
	float voxelSize = this->scene->sceneParams->voxelSize;

	Vector2i imgSize = renderState->renderingRangeImage->noDims;

	//go through list of visible 8x8x8 blocks
	{
		const ITMHashEntry *hash_entries = this->scene->index.GetEntries();
		const int *visibleEntryIDs = renderState->GetVisibleEntryIDs();
		int noVisibleEntries = renderState->noVisibleEntries;

		dim3 blockSize(256);
		dim3 gridSize((int)ceil((float)noVisibleEntries / (float)blockSize.x));
		ITMSafeCall(cudaMemset(noTotalBlocks_device, 0, sizeof(uint)));
		projectAndSplitBlocks_device << <gridSize, blockSize >> >(hash_entries, visibleEntryIDs, noVisibleEntries, pose->GetM(),
			intrinsics->projectionParamsSimple.all, imgSize, voxelSize, renderingBlockList_device, noTotalBlocks_device);
	}

	uint noTotalBlocks;
	ITMSafeCall(cudaMemcpy(&noTotalBlocks, noTotalBlocks_device, sizeof(uint), cudaMemcpyDeviceToHost));
	if (noTotalBlocks > (unsigned)MAX_RENDERING_BLOCKS) noTotalBlocks = MAX_RENDERING_BLOCKS;

	// go through rendering blocks and fill minmaxData
    Vector2f * const minmaxData = renderState->renderingRangeImage->GetData(MEMORYDEVICE_CUDA);
    memsetKernel<Vector2f>(minmaxData, Vector2f(FAR_AWAY, VERY_CLOSE), renderState->renderingRangeImage->dataSize);


	dim3 blockSize(16, 16);
	dim3 gridSize((unsigned int)ceil((float)noTotalBlocks / 4.0f), 4);
	fillBlocks_device << <gridSize, blockSize >> >(noTotalBlocks_device, renderingBlockList_device, imgSize, minmaxData);
}

static void GenericRaycast(const ITMScene *scene, const Vector2i& imgSize, const Matrix4f& invM, Vector4f projParams, const ITMRenderState *renderState)
{
	const float voxelSize = scene->sceneParams->voxelSize;
	const float oneOverVoxelSize = 1.0f / voxelSize;

	projParams.x = 1.0f / projParams.x;
	projParams.y = 1.0f / projParams.y;

	dim3 cudaBlockSize(16, 12);
	dim3 gridSize((int)ceil((float)imgSize.x / (float)cudaBlockSize.x), (int)ceil((float)imgSize.y / (float)cudaBlockSize.y));
    genericRaycast_device<< <gridSize, cudaBlockSize >> >(
		renderState->raycastResult->GetData(MEMORYDEVICE_CUDA),
		scene->localVBA.GetVoxelBlocks(),
		scene->index.getIndexData(),
		imgSize,
		invM,
		projParams,
		oneOverVoxelSize,
		renderState->renderingRangeImage->GetData(MEMORYDEVICE_CUDA),
		scene->sceneParams->mu
	);
}

static void RenderImage_common(const ITMScene *scene, const ITMPose *pose, const ITMIntrinsics *intrinsics, const ITMRenderState *renderState,
    ITMUChar4Image *outputImage, ITMVisualisationEngine::RenderImageType type)
{
	Vector2i imgSize = outputImage->noDims;
	Matrix4f invM = pose->GetInvM();

	GenericRaycast(scene, imgSize, invM, intrinsics->projectionParamsSimple.all, renderState);

	Vector3f lightSource = -Vector3f(invM.getColumn(2));
	Vector4u *outRendering = outputImage->GetData(MEMORYDEVICE_CUDA);
	Vector4f *pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CUDA);

	dim3 cudaBlockSize(8, 8);
	dim3 gridSize((int)ceil((float)imgSize.x / (float)cudaBlockSize.x), (int)ceil((float)imgSize.y / (float)cudaBlockSize.y));
    
	switch (type) {
	case ITMVisualisationEngine::RENDER_COLOUR_FROM_VOLUME:
        renderColour_device << <gridSize, cudaBlockSize >> >(outRendering, pointsRay, scene->localVBA.GetVoxelBlocks(),
			scene->index.getIndexData(), imgSize, lightSource);
		break;
	case ITMVisualisationEngine::RENDER_COLOUR_FROM_NORMAL:
        renderColourFromNormal_device << <gridSize, cudaBlockSize >> >(outRendering, pointsRay, scene->localVBA.GetVoxelBlocks(),
			scene->index.getIndexData(), imgSize, lightSource);
		break;
	case ITMVisualisationEngine::RENDER_SHADED_GREYSCALE:
	default:
        renderGrey_device<< <gridSize, cudaBlockSize >> >(outRendering, pointsRay, scene->localVBA.GetVoxelBlocks(),
			scene->index.getIndexData(), imgSize, lightSource);
		break;
	}
}

void CreateICPMaps_common(const ITMScene *scene, Vector4f intrinsics_d, ITMTrackingState *trackingState, ITMRenderState *renderState)
{
	Vector2i imgSize = renderState->raycastResult->noDims;
	Matrix4f invM = trackingState->pose_d->GetInvM();

    GenericRaycast(scene, imgSize, invM, intrinsics_d, renderState);
	
    // Remember the pose from which this point cloud was rendered
    trackingState->pointCloud->pose_pointCloud->SetFrom(trackingState->pose_d);

	Vector4f *pointsMap = trackingState->pointCloud->locations->GetData(MEMORYDEVICE_CUDA);
    Vector4f *normalsMap = trackingState->pointCloud->normals->GetData(MEMORYDEVICE_CUDA);
	Vector4u *outRendering = renderState->raycastImage->GetData(MEMORYDEVICE_CUDA);
	Vector4f *pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CUDA);

	Vector3f lightSource = -Vector3f(invM.getColumn(2));

	dim3 cudaBlockSize(16, 12);
	dim3 gridSize((int)ceil((float)imgSize.x / (float)cudaBlockSize.x), (int)ceil((float)imgSize.y / (float)cudaBlockSize.y));
	renderICP_device<<<gridSize, cudaBlockSize>>>(outRendering, pointsMap, normalsMap, pointsRay,
		scene->sceneParams->voxelSize, imgSize, lightSource);
}

void ITMVisualisationEngine_CUDA::RenderImage(const ITMPose *pose, const ITMIntrinsics *intrinsics, 
	ITMRenderState *renderState, ITMUChar4Image *outputImage, ITMVisualisationEngine::RenderImageType type) const
{
    CreateExpectedDepths(pose, intrinsics, renderState);
	RenderImage_common(this->scene, pose, intrinsics, renderState, outputImage, type);
}

void ITMVisualisationEngine_CUDA::CreateICPMaps(
    const ITMIntrinsics * intrinsics_d,
    ITMTrackingState *trackingState,
	ITMRenderState *renderState) const
{
    CreateExpectedDepths(trackingState->pose_d, intrinsics_d, renderState);
    CreateICPMaps_common(this->scene, intrinsics_d->projectionParamsSimple.all, trackingState, renderState);
}

