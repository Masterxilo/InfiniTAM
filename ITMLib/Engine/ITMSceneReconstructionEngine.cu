// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMSceneReconstructionEngine.h"
#include "ITMCUDAUtils.h"
#include "ITMLibDefines.h"
#include "ITMPixelUtils.h"
#include "ITMRepresentationAccess.h"
#include "ITMLocalVBA.h"

using namespace ITMLib::Engine;



/// Fusion Stage - Camera Data Integration
/// \returns \f$\eta\f$, -1 on failure
// Note that the stored T-SDF values are normalized to lie
// in [-1,1] within the truncation band.
CPU_AND_GPU inline float computeUpdatedVoxelDepthInfo(
    DEVICEPTR(ITMVoxel) &voxel, //!< X
    const THREADPTR(Vector4f) & pt_model, //!< voxel location X
    const CONSTPTR(Matrix4f) & M_d, //!< depth camera pose
    const CONSTPTR(Vector4f) & projParams_d, //!< intrinsic camera parameters \f$K_d\f$
    float mu, int maxW, const CONSTPTR(float) *depth, const CONSTPTR(Vector2i) & imgSize)
{

    float depth_measure, eta, oldF, newF;
    int oldW, newW;

    // project point into depth image
    /// X_d, depth camera coordinate system
    Vector4f pt_camera;
    /// \pi(K_dX_d), projection into the depth image
    Vector2f pt_image;
    if (!projectModel(projParams_d, M_d,
        imgSize, pt_model, pt_camera, pt_image)) return -1;

    // get measured depth from image, no interpolation
    /// I_d(\pi(K_dX_d))
    depth_measure = sampleNearest(depth, pt_image, imgSize);
    if (depth_measure <= 0.0) return -1;

    /// I_d(\pi(K_dX_d)) - X_d^(z)          (3)
    eta = depth_measure - pt_camera.z;
    // check whether voxel needs updating
    if (eta < -mu) return eta;

    // compute updated SDF value and reliability (number of observations)
    /// D(X), w(X)
    oldF = voxel.getSDF();
    oldW = voxel.w_depth;

    // newF, normalized for -1 to 1
    newF = MIN(1.0f, eta / mu);
    newW = 1;

    updateVoxelDepthInformation(
        voxel,
        oldF, oldW, newF, newW, maxW);

    return eta;
}

/// \returns early on failure
CPU_AND_GPU inline void computeUpdatedVoxelColorInfo(DEVICEPTR(ITMVoxel) &voxel, const THREADPTR(Vector4f) & pt_model, const CONSTPTR(Matrix4f) & M_rgb,
    const CONSTPTR(Vector4f) & projParams_rgb, float mu, uchar maxW, float eta, const CONSTPTR(Vector4u) *rgb, const CONSTPTR(Vector2i) & imgSize)
{
    Vector4f pt_camera; Vector2f pt_image;
    Vector3f oldC, newC;
    int newW, oldW;

    if (!projectModel(projParams_rgb, M_rgb,
        imgSize, pt_model, pt_camera, pt_image)) return;

    oldW = (float)voxel.w_color;
    oldC = TO_FLOAT3(voxel.clr);

    /// Like formula (4) for depth
    newC = TO_VECTOR3(interpolateBilinear<Vector4f>(rgb, pt_image, imgSize));
    newW = 1;

    updateVoxelColorInformation(
        voxel,
        oldC, oldW, newC, newW, maxW);
}


CPU_AND_GPU static void computeUpdatedVoxelInfo(
    DEVICEPTR(ITMVoxel) & voxel, //!< [in, out] updated voxel
    const THREADPTR(Vector4f) & pt_model,
    const THREADPTR(Matrix4f) & M_d, const THREADPTR(Vector4f) & projParams_d,
    const THREADPTR(Matrix4f) & M_rgb, const THREADPTR(Vector4f) & projParams_rgb,
    float mu, int maxW,
    const CONSTPTR(float) *depth, const CONSTPTR(Vector2i) & imgSize_d,
    const CONSTPTR(Vector4u) *rgb, const THREADPTR(Vector2i) & imgSize_rgb)
{
    float eta = computeUpdatedVoxelDepthInfo(voxel, pt_model, M_d, projParams_d, mu, maxW, depth, imgSize_d);

    // Only the voxels withing +- 25% mu of the surface get color
    if ((eta > mu) || (fabs(eta / mu) > 0.25f)) return;
    computeUpdatedVoxelColorInfo(voxel, pt_model, M_rgb, projParams_rgb, mu, maxW, eta, rgb, imgSize_rgb);
}

// alloc types
#define AT_NEEDS_ALLOC_FITS 1 //needs allocation, fits in the ordered list
#define AT_NEEDS_ALLOC_EXCESS 2 //needs allocation in the excess list

/// For allocation and visibility determination. 
///
/// Determine the blocks around a given depth sample that are currently visible
/// and need to be allocated.
/// Builds hashVisibility and entriesAllocType.
/// \param x,y [in] loop over depth image.
CPU_AND_GPU inline void buildHashAllocAndVisibleTypePP(
    DEVICEPTR(uchar) * const entriesAllocType, //!< [out] allocation type (AT_*) for each hash table bucket, indexed by values computed from hashIndex, or in excess part
    const int x, const int y,
    DEVICEPTR(Vector4s) * const  blockCoords, //!< [out] blockPos coordinate of each voxel block that needs allocation, indexed by values computed from hashIndex, or in excess part
    const CONSTPTR(float) * const  depth,
    const Matrix4f invM_d, //!< depth to world transformation
    const Vector4f invProjParams_d, //!< Note: Inverse projection parameters to avoid division by fx, fy.
    const float mu,
    const Vector2i imgSize,
    const float oneOverVoxelBlockWorldspaceSize, //!< 1 / (voxelSize * SDF_BLOCK_SIZE)
    const CONSTPTR(ITMHashEntry) *const hashTable, //<! [in] hash table buckets, indexed by values computed from hashIndex
    const float viewFrustum_min, //!< znear
    const float viewFrustum_max  //!< zfar
    )
{
    float depth_measure; unsigned int hashIdx; int noSteps;
    Vector4f pt_camera_f; Vector3f point_e, point, direction; Vector3s blockPos;

    // Find 3d position of depth pixel xy
    depth_measure = depth[x + y * imgSize.x];
    if (depth_measure <= 0 || (depth_measure - mu) < 0 || (depth_measure - mu) < viewFrustum_min || (depth_measure + mu) > viewFrustum_max) return;

    pt_camera_f = depthTo3DInvProjParams(invProjParams_d, x, y, depth_measure);

    // distance from camera
    float norm = length(pt_camera_f.toVector3());

    // Transform into block coordinates the found point +- mu
    // TODO why /norm? An adhoc fix to not allocate too much when far away and allocate more when nearby?
    point = TO_VECTOR3(invM_d * (pt_camera_f * (1.0f - mu / norm))) * oneOverVoxelBlockWorldspaceSize;
    point_e = TO_VECTOR3(invM_d * (pt_camera_f * (1.0f + mu / norm))) * oneOverVoxelBlockWorldspaceSize;

    // We will step along point -> point_e and add all voxel blocks we encounter to the visible list
    // "Create a segment on the line of sight in the range of the T-SDF truncation band"
    direction = point_e - point;
    norm = length(direction);
    noSteps = (int)ceil(2.0f*norm);

    direction /= (float)(noSteps - 1);

    //add neighbouring blocks
    for (int i = 0; i < noSteps; i++)
    {
        // "take the block coordinates of voxels on this line segment"
        blockPos = TO_SHORT_FLOOR3(point);

        //compute index in hash table
        hashIdx = hashIndex(blockPos);

        //check if hash table contains entry (block has already been allocated)
        bool isFound = false;

        ITMHashEntry hashEntry;

        // whether we find blockPos at the current hashIdx
#define check_found(BREAK) \
            hashEntry = hashTable[hashIdx]; \
            if (IS_EQUAL3(hashEntry.pos, blockPos) && hashEntry.isAllocated()) \
                        {\
                isFound = true; \
                BREAK;\
                        }

        check_found(NULL);

        if (!isFound)
        {
            bool isExcess = false;
            if (hashEntry.isAllocated()) //seach excess list only if there is no room in ordered part
            {
                isExcess = true;
                while (hashEntry.hasExcessListOffset())
                {
                    hashIdx = hashEntry.getHashIndexOfNextExcessEntry();
                    check_found(break);
                }
            }

            if (!isFound) //still not found: needs allocation 
            {
                entriesAllocType[hashIdx] = isExcess ? AT_NEEDS_ALLOC_EXCESS : AT_NEEDS_ALLOC_FITS; //needs allocation
                blockCoords[hashIdx] = Vector4s(blockPos, 1);
            }
        }

        point += direction;
    }
#undef check_found
}

#include <cuda_runtime.h>

/// \returns false when the list is full

inline
__device__
void allocateVoxelBlock(
int targetIdx,

typename ITMLocalVBA::VoxelAllocationList* voxelAllocationList,
ITMVoxelBlockHash::ExcessAllocationList* excessAllocationList,
ITMHashEntry *hashTable,
ITMVoxelBlock *localVBA,

uchar *entriesAllocType,
Vector4s *blockCoords)
{
    const unsigned char hashChangeType = entriesAllocType[targetIdx];
    if (hashChangeType == 0) return;
    const int ptr = voxelAllocationList->Allocate();
    assert(ptr >= 0 && ptr < SDF_LOCAL_BLOCK_NUM);
    //if (ptr < 0) return; //there is no room in the voxel block array


    ITMHashEntry hashEntry;
    hashEntry.pos = TO_SHORT3(blockCoords[targetIdx]);
    hashEntry.ptr = ptr;
    hashEntry.offset = 0;

    // Allocated voxel block - back-reference to key:
    assert(localVBA[ptr].pos == INVALID_VOXEL_BLOCK_POS); // make sure this was free before
    localVBA[ptr].pos = hashEntry.pos;

    if (hashChangeType == AT_NEEDS_ALLOC_EXCESS) { //needs allocation in the excess list
        const int exlOffset = excessAllocationList->Allocate();

        if (exlOffset >= 0) //there is room in the excess list
        {
            hashTable[targetIdx].offset = exlOffset + 1; //connect parent to child

            targetIdx = SDF_BUCKET_NUM + exlOffset; // target index is in excess part
        }
    }

    hashTable[targetIdx] = hashEntry;
}


CPU_AND_GPU inline void integrateVoxel(int x, int y, int z,
    Vector3i globalPos,
    ITMVoxelBlock *localVoxelBlock,
    float voxelSize,

    const CONSTPTR(Matrix4f) & M_d, const CONSTPTR(Vector4f) & projParams_d,
    const CONSTPTR(Matrix4f) & M_rgb, const CONSTPTR(Vector4f) & projParams_rgb,
    float mu, int maxW,
    const CONSTPTR(float) *depth, const CONSTPTR(Vector2i) & depthImgSize,
    const CONSTPTR(Vector4u) *rgb, const CONSTPTR(Vector2i) & rgbImgSize
    ) {
    const int locId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

    // Voxel's world coordinates, for later projection into depth and color image
    const Vector4f pt_model = Vector4f(
        (globalPos.toFloat() + Vector3f((float)x, (float)y, (float)z)) * voxelSize, 1.f);

    computeUpdatedVoxelInfo(
        localVoxelBlock->blockVoxels[locId],
        pt_model,
        M_d,
        projParams_d, M_rgb, projParams_rgb, mu, maxW, depth, depthImgSize, rgb, rgbImgSize);
}

// device functions
KERNEL integrateIntoScene_device(
    ITMVoxelBlock *localVBA, //!< [out]

    const ITMHashEntry * const hashTable,
    const Vector4u * const rgb,
    const Vector2i rgbImgSize, 
    const float * const depth,
    const Vector2i depthImgSize, const Matrix4f M_d, const Matrix4f M_rgb, const Vector4f projParams_d,
    const Vector4f projParams_rgb, const float voxelSize, const float mu, const int maxW)
{
    // one thread block for each voxel block
    ITMVoxelBlock * const localVoxelBlock = &(localVBA[blockIdx.x]); 
    if (localVoxelBlock->pos == INVALID_VOXEL_BLOCK_POS) return; 

    // one thread for each voxel
    const int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;
    integrateVoxel(x, y, z,
        localVoxelBlock->pos.toInt() * SDF_BLOCK_SIZE,
        localVoxelBlock, voxelSize,
        M_d, projParams_d, M_rgb, projParams_rgb, mu, maxW, depth, depthImgSize, rgb, rgbImgSize);
}

KERNEL buildHashAllocAndVisibleType_device(
    uchar *entriesAllocType, //!< [out]
    Vector4s *blockCoords,//!< [out]
    const float *depth,
    Matrix4f invM_d, Vector4f projParams_d, float mu, Vector2i imgSize, float voxelSize, ITMHashEntry *hashTable, float viewFrustum_min,
    float viewFrustum_max)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > imgSize.x - 1 || y > imgSize.y - 1) return;

    buildHashAllocAndVisibleTypePP(entriesAllocType, x, y, blockCoords, depth, invM_d,
        projParams_d, mu, imgSize, voxelSize, hashTable, viewFrustum_min, viewFrustum_max);
}


KERNEL allocateVoxelBlocksList_device(
    typename ITMLocalVBA::VoxelAllocationList * const voxelAllocationList,
    ITMVoxelBlockHash::ExcessAllocationList * const excessAllocationList,
    ITMHashEntry *hashTable,
    ITMVoxelBlock* localVBA,
    uchar *entriesAllocType,
    Vector4s *blockCoords)
{
    const int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (targetIdx > SDF_GLOBAL_BLOCK_NUM - 1) return;

    allocateVoxelBlock(targetIdx,
        voxelAllocationList,
        excessAllocationList,
        hashTable,
        localVBA,

        entriesAllocType,
        blockCoords);
}


// host methods

ITMSceneReconstructionEngine::ITMSceneReconstructionEngine(void) 
{
    cudaSafeCall(cudaMalloc((void**)&entriesAllocType_device, SDF_GLOBAL_BLOCK_NUM));
    cudaSafeCall(cudaMalloc((void**)&blockCoords_device, SDF_GLOBAL_BLOCK_NUM * sizeof(Vector4s)));
}

ITMSceneReconstructionEngine::~ITMSceneReconstructionEngine(void) 
{
	cudaSafeCall(cudaFree(entriesAllocType_device));
	cudaSafeCall(cudaFree(blockCoords_device));
}

/// thread blocks 0:numBlocks-1, threads 0:SDF_BLOCK_SIZE3-1
KERNEL resetVoxelBlocks(ITMVoxelBlock *voxelBlocks_ptr) {
    voxelBlocks_ptr[blockIdx.x].blockVoxels[threadIdx.x] = ITMVoxel();

    if (threadIdx.x == 0) voxelBlocks_ptr[blockIdx.x].pos = INVALID_VOXEL_BLOCK_POS;
}

void ITMSceneReconstructionEngine::ResetScene(ITMScene *scene)
{
    printf("ResetScene\n");

    // Reset sdf data of all voxels in all voxel blocks
    ITMVoxelBlock *voxelBlocks_ptr = scene->localVBA.GetVoxelBlocks();
    resetVoxelBlocks << <SDF_LOCAL_BLOCK_NUM, SDF_BLOCK_SIZE3 >> >(voxelBlocks_ptr);

    // Reset voxel allocation list
    scene->localVBA.voxelAllocationList->Reset();

    // Reset hash entries
    memsetKernel<ITMHashEntry>(
        scene->index.GetEntries(),
        ITMHashEntry::createIllegalEntry(),
        SDF_GLOBAL_BLOCK_NUM);

    // Reset excess allocation list
    scene->index.excessAllocationList->Reset();
    printf("ResetScene done\n");
}

void ITMSceneReconstructionEngine::AllocateSceneFromDepth(
    ITMScene *scene,
    const ITMView *view, 
	const ITMTrackingState *trackingState)
{
	const Vector2i depthImgSize = view->depth->noDims;
    const float voxelSize = scene->sceneParams->voxelSize;
    
    const Matrix4f M_d = trackingState->pose_d->GetM();
    const Matrix4f invM_d = trackingState->pose_d->GetInvM();

    const Vector4f invProjParams_d = view->calib->intrinsics_d.getInverseProjParams();

	const float mu = scene->sceneParams->mu;

    float * const depth = view->depth->GetData(MEMORYDEVICE_CUDA);
    auto voxelAllocationList = scene->localVBA.voxelAllocationList;
    auto excessAllocationList = scene->index.excessAllocationList;
    ITMHashEntry * const hashTable = scene->index.GetEntries();


    const dim3 cudaBlockSizeHV(16, 16);
    const dim3 gridSizeHV((int)ceil((float)depthImgSize.x / (float)cudaBlockSizeHV.x), (int)ceil((float)depthImgSize.y / (float)cudaBlockSizeHV.y));

	const dim3 cudaBlockSizeAL(256, 1);
	const dim3 gridSizeAL((int)ceil((float)SDF_GLOBAL_BLOCK_NUM / (float)cudaBlockSizeAL.x));

    const float oneOverVoxelSize = 1.0f / (voxelSize * SDF_BLOCK_SIZE);


    // Determine blocks currently visible in depth map but not allocated  for preparing allocation list

    cudaSafeCall(cudaMemsetAsync(entriesAllocType_device, 0, sizeof(unsigned char)* SDF_GLOBAL_BLOCK_NUM));

	buildHashAllocAndVisibleType_device << <gridSizeHV, cudaBlockSizeHV >> >(
        entriesAllocType_device,
		blockCoords_device,
        depth, invM_d, invProjParams_d, mu, depthImgSize, oneOverVoxelSize, hashTable,
		scene->sceneParams->viewFrustum_min,
        scene->sceneParams->viewFrustum_max);

    // Do allocation
    allocateVoxelBlocksList_device << <gridSizeAL, cudaBlockSizeAL >> >(
        voxelAllocationList, 
        excessAllocationList, 
        hashTable,
        scene->localVBA.GetVoxelBlocks(),
        entriesAllocType_device, 
		blockCoords_device);
}

void ITMSceneReconstructionEngine::IntegrateIntoScene(ITMScene *scene, const ITMView *view,
	const ITMTrackingState *trackingState)
{
	Vector2i rgbImgSize = view->rgb->noDims;
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, M_rgb;
	Vector4f projParams_d, projParams_rgb;

	M_d = trackingState->pose_d->GetM();
    M_rgb = view->calib->trafo_rgb_to_depth.calib_inv * M_d;

	projParams_d = view->calib->intrinsics_d.projectionParamsSimple.all;
	projParams_rgb = view->calib->intrinsics_rgb.projectionParamsSimple.all;

	float mu = scene->sceneParams->mu; int maxW = scene->sceneParams->maxW;

	float *depth = view->depth->GetData(MEMORYDEVICE_CUDA);
	Vector4u *rgb = view->rgb->GetData(MEMORYDEVICE_CUDA);
    ITMVoxelBlock *localVBA = scene->localVBA.GetVoxelBlocks();
	ITMHashEntry *hashTable = scene->index.GetEntries();

	dim3 cudaBlockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
    dim3 gridSize(SDF_LOCAL_BLOCK_NUM);

    integrateIntoScene_device<< <gridSize, cudaBlockSize >> >(
        localVBA, hashTable, 
        rgb, rgbImgSize, depth, depthImgSize, M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
}

