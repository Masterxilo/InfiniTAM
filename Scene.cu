#include "Scene.h"

//__device__ Scene* currentScene;
//__host__
__managed__ Scene* currentScene = 0; // TODO use __const__ memory, since this value is not changeable from gpu!

CPU_AND_GPU Scene* Scene::getCurrentScene() {
    return currentScene;
}

void Scene::setCurrentScene(Scene* s) {
    cudaDeviceSynchronize(); // want to write managed currentScene 
    currentScene = s;
}


// performAllocations -- private:
__managed__ ITMVoxelBlock* currentLocalVBA = 0; //!< used for passing localVBA to allocate, called when doing voxel allocations by HashMap
__device__ void Scene::AllocateVB::allocate(VoxelBlockPos pos, int sequenceId) {
    assert(currentLocalVBA);

    currentLocalVBA[sequenceId].reinit(pos);
}

void Scene::performAllocations() {
    assert(!currentLocalVBA);
    currentLocalVBA = localVBA;
    voxelBlockHash->performAllocations(); // will call Scene::AllocateVB::allocate for all outstanding allocations
    currentLocalVBA = 0;
}
//

Scene::Scene() {
    initCoordinateSystems();
    assert(mu > voxelSize * 2);
    voxelBlockHash = new HashMap<Z3Hasher, AllocateVB>(SDF_EXCESS_LIST_SIZE);
    cudaSafeCall(cudaMalloc(&localVBA, sizeof(ITMVoxelBlock) *SDF_LOCAL_BLOCK_NUM));
}

Scene::~Scene() {
    delete voxelBlockHash;
    cudaFree(localVBA);
}


static GPU_ONLY inline VoxelBlockPos pointToVoxelBlockPos(
    const THREADPTR(Vector3i) & point //!< [in] in voxel coordinates
    ) {
    // "The 3D voxel block location is obtained by dividing the voxel coordinates with the block size along each axis."
    VoxelBlockPos blockPos;
    // if SDF_BLOCK_SIZE == 8, then -3 should go to block -1, so we need to adjust negative values 
    // (C's quotient-remainder division gives -3/8 == 0)
    blockPos.x = ((point.x < 0) ? point.x - SDF_BLOCK_SIZE + 1 : point.x) / SDF_BLOCK_SIZE;
    blockPos.y = ((point.y < 0) ? point.y - SDF_BLOCK_SIZE + 1 : point.y) / SDF_BLOCK_SIZE;
    blockPos.z = ((point.z < 0) ? point.z - SDF_BLOCK_SIZE + 1 : point.z) / SDF_BLOCK_SIZE;
    return blockPos;
}

GPU_ONLY ITMVoxel* Scene::getVoxel(Vector3i point) {
    VoxelBlockPos blockPos = pointToVoxelBlockPos(point);

    ITMVoxelBlock* b = getVoxelBlock(blockPos);
    if (b == NULL) return NULL;

    Vector3i localPos = point - blockPos.toInt() * SDF_BLOCK_SIZE; // localized coordinate
    return b->getVoxel(localPos);
}

GPU_ONLY ITMVoxelBlock* Scene::getVoxelBlockForSequenceNumber(unsigned int sequenceNumber) {
    assert(sequenceNumber < SDF_LOCAL_BLOCK_NUM);
    assert(sequenceNumber < voxelBlockHash->getLowestFreeSequenceNumber());
    return &localVBA[sequenceNumber];
}

/// Returns NULL if the voxel block is not allocated
GPU_ONLY ITMVoxelBlock* Scene::getVoxelBlock(VoxelBlockPos pos) {
    int sequenceNumber = voxelBlockHash->getSequenceNumber(pos); // returns 0 if pos is not allocated
    if (sequenceNumber == 0) return NULL;
    return &localVBA[sequenceNumber];
}

GPU_ONLY void Scene::requestVoxelBlockAllocation(VoxelBlockPos pos) {
    voxelBlockHash->requestAllocation(pos);
}

/// --- dumping ---
#include "fileutils.h"
#include <stdio.h>
int fileExists(TCHAR * file);

void Scene::dump(std::string filename) {
    FILE* file = fopen(filename.c_str(), "wb");
    assert(file);
    const int N = voxelBlockHash->getLowestFreeSequenceNumber();
    fwrite(&N, sizeof(N), 1, file);

    auto cpu_localVBA = new MemoryBlock<ITMVoxelBlock>(N);
    cudaMemcpy(cpu_localVBA->GetData(), localVBA, N*sizeof(ITMVoxelBlock), cudaMemcpyDeviceToHost); // this takes forever -- copy only N not all!
    assert(cpu_localVBA->GetData()[1].pos_ != cpu_localVBA->GetData()[2].pos_);
    /*
    for (int i = 0; i < N; i++) { // dont care that 0 is actually never used
        fwrite(&cpu_localVBA->GetData()[i], sizeof(ITMVoxelBlock), 1, file);
    }*/
    fwrite(cpu_localVBA->GetData(), sizeof(ITMVoxelBlock), N, file);
    // no need to write hashmap -- transparent to the application

    fclose(file);
    assert(fileExists((char*)filename.c_str()));
    assert(dump::fileSize(filename) == sizeof(int) + N*sizeof(ITMVoxelBlock));

    delete cpu_localVBA;
}

static __managed__ int counter = 0;
static KERNEL requestAllocation(VoxelBlockPos vbp) {
    Scene::requestCurrentSceneVoxelBlockAllocation(vbp);
}

static KERNEL countIfAllocated(VoxelBlockPos vbp) {
    if (Scene::getCurrentSceneVoxel(vbp.toInt() * SDF_BLOCK_SIZE)) {
        atomicAdd(&counter, 1);
    }
}

static __managed__ char submittedVb_[sizeof(ITMVoxelBlock)];

static KERNEL fill() {
    ITMVoxelBlock& submittedVb = *(ITMVoxelBlock*)submittedVb_;
    VoxelBlockPos p = submittedVb.pos;
    printf("%d %d %d\n", xyz(p));
    assert(p.x < 10000); // sanity

    // assume threads run for each voxel
    ITMVoxel* voxel = Scene::getCurrentSceneVoxel(
        submittedVb.pos.toInt() * SDF_BLOCK_SIZE +
        Vector3i(threadIdx_xyz));

    assert(voxel);
    ITMVoxel* svoxel = submittedVb.getVoxel(Vector3i(threadIdx_xyz));
    *voxel = *svoxel; // copy state
}

KERNEL checkSdf(Vector3i voxelp, float expected_sdf) {
    assert(abs(
        expected_sdf - Scene::getCurrentSceneVoxel(voxelp)->getSDF()
        ) < 0.0001);
}
void pauseit() {
    while (1);
    system("pause");
}
// works on the current scene, assumes scene is empty so far
void Scene::restore(std::string filename) {
    atexit(pauseit);
    assert(getCurrentScene() == this);
    assert(voxelBlockHash->getLowestFreeSequenceNumber() == 1);

    FILE* file = fopen(filename.c_str(), "rb");
    assert(file);
    int N; fread(&N, sizeof(N), 1, file);
    assert(N > 1);
    auto cpu_localVBA = new MemoryBlock<ITMVoxelBlock>(N); // allocate only N not SDF_LOCAL_BLOCK_NUM, takes forever (many constructors)
    fread(cpu_localVBA->GetData(), sizeof(ITMVoxelBlock), N, file);
    fclose(file);
    
    assert(cpu_localVBA->GetData()[1].pos_ != cpu_localVBA->GetData()[2].pos_);

    // allocate
    do {
        for (int j = 1; j < N; j++) {
            cudaDeviceSynchronize();
            requestAllocation << <1, 1 >> >(cpu_localVBA->GetData()[j].pos_);
        cudaDeviceSynchronize();
        }
        Scene::performCurrentSceneAllocations();
        cudaDeviceSynchronize();

        counter = 0;
        for (int j = 1; j < N; j++) {
            cudaDeviceSynchronize();
            countIfAllocated << <1, 1 >> >(cpu_localVBA->GetData()[j].pos_);
            cudaDeviceSynchronize();
        }
        cudaDeviceSynchronize();
    } while (counter != N-1);

    // fill
    for (int j = 1; j < N; j++) {
        cudaDeviceSynchronize();
        //submittedVb = cpu_localVBA->GetData()[j];
        memcpy(submittedVb_, &cpu_localVBA->GetData()[j], sizeof(ITMVoxelBlock));
        fill << <1, dim3(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE) >> >();
    }

    // verify
    cudaDeviceSynchronize();
    Vector3i voxelp = cpu_localVBA->GetData()[1].pos_.toInt() * SDF_BLOCK_SIZE;
    float sdf = cpu_localVBA->GetData()[1].blockVoxels[0].getSDF();
    checkSdf << <1, 1 >> >(voxelp, sdf);

    // done

    cudaDeviceSynchronize();
    assert(voxelBlockHash->getLowestFreeSequenceNumber() == N);
}