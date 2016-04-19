#pragma once 
#include "itmlibdefines.h"
#include "cudadefines.h"
#include "hashmap.h"

// see doForEachAllocatedVoxel for T
#define doForEachAllocatedVoxel_process() static GPU_ONLY void process(const ITMVoxelBlock* vb, ITMVoxel* v, const Vector3i localPos)
template<typename T>
KERNEL doForEachAllocatedVoxel(
    ITMVoxelBlock* localVBA,
    int nextFreeSequenceId) {
    int index = blockIdx.x;
    if (index <= 0 || index >= nextFreeSequenceId) return;

    ITMVoxelBlock* vb = &localVBA[index];
    Vector3i localPos(threadIdx_xyz);
    T::process(vb, vb->getVoxel(localPos), localPos);
}

#define doForEachAllocatedVoxelBlock_process() static GPU_ONLY void process(ITMVoxelBlock* voxelBlock)
// see doForEachAllocatedVoxel for T
template<typename T>
KERNEL doForEachAllocatedVoxelBlock(
    ITMVoxelBlock* localVBA,
    int nextFreeSequenceId) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index <= 0 || index >= nextFreeSequenceId) return;

    ITMVoxelBlock* vb = &localVBA[index];
    T::process(vb);
}


/// Must be heap-allocated
class Scene : public Managed {
public:
    /// \returns NULL when the voxel was not found
    GPU_ONLY ITMVoxel* getVoxel(Vector3i pos);

    /// \returns a voxel block from the localVBA
    GPU_ONLY ITMVoxelBlock* getVoxelBlockForSequenceNumber(unsigned int sequenceNumber);

    /// Returns NULL if the voxel block is not allocated
    GPU_ONLY void requestVoxelBlockAllocation(VoxelBlockPos pos);
    void performAllocations();

    Scene();
    virtual ~Scene();

    void dump(std::string filename);
    void restore(std::string filename);

    /*
    void reset() {
        // TODO implement
        printf("Scene::reset not implemented\n");
    }*/

    /// T must have an operator(ITMVoxelBlock*, ITMVoxel*, Vector3i localPos)
    /// where localPos will run from 0,0,0 to (SDF_BLOCK_SIZE-1)^3
    /// runs threadblock per voxel block and thread per thread
    template<typename T>
    void doForEachAllocatedVoxel() {
        LAUNCH_KERNEL(
            ::doForEachAllocatedVoxel<T>,
            SDF_LOCAL_BLOCK_NUM,
            dim3(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE),
            localVBA,
            voxelBlockHash->getLowestFreeSequenceNumber()
            );
    }

    /// T must have an operator(ITMVoxelBlock*)
    template<typename T>
    void doForEachAllocatedVoxelBlock() {

        dim3 blockSize(256);
        dim3 gridSize((int)ceil((float)SDF_LOCAL_BLOCK_NUM / (float)blockSize.x));
        LAUNCH_KERNEL(
            ::doForEachAllocatedVoxelBlock<T>,
            gridSize,
            blockSize,
            localVBA,
            voxelBlockHash->getLowestFreeSequenceNumber()
            );
    }

    static GPU_ONLY ITMVoxel* getCurrentSceneVoxel(Vector3i pos) {
        assert(getCurrentScene());
        return getCurrentScene()->getVoxel(pos);
    }
    static GPU_ONLY void requestCurrentSceneVoxelBlockAllocation(VoxelBlockPos pos) {
        assert(getCurrentScene());
        return getCurrentScene()->requestVoxelBlockAllocation(pos);
    }

    static void Scene::performCurrentSceneAllocations() {
        assert(getCurrentScene());
        getCurrentScene()->performAllocations();
    }


    /** !private! But has to be placed in public for HashMap to access it - unless we make that a friend */
    struct Z3Hasher {
        typedef VoxelBlockPos KeyType;
        static const uint BUCKET_NUM = SDF_BUCKET_NUM; // Number of Hash Bucket, must be 2^n (otherwise we have to use % instead of & below)

        static GPU_ONLY uint hash(const VoxelBlockPos& blockPos) {
            return (((uint)blockPos.x * 73856093u) ^ ((uint)blockPos.y * 19349669u) ^ ((uint)blockPos.z * 83492791u))
                &
                (uint)(BUCKET_NUM - 1);
        }
    };

    /** !private! But has to be placed in public for HashMap to access it - unless we make that a friend*/
    struct AllocateVB {
        static __device__ void allocate(VoxelBlockPos pos, int sequenceId);
    };

    // Scene is mostly fixed. // TODO prefer using a scoping construct that lives together with the call stack!
    // Having it globally accessible heavily reduces having
    // to pass parameters.
    static CPU_AND_GPU Scene* getCurrentScene();

    /// Change current scene for the current block/scope
    class CurrentSceneScope {
    public:
        CurrentSceneScope(Scene* const newCurrentScene) : 
            oldCurrentScene(Scene::getCurrentScene()) {
            Scene::setCurrentScene(newCurrentScene);
        }
        ~CurrentSceneScope() {
            Scene::setCurrentScene(oldCurrentScene);
        }

    private:
        Scene* const oldCurrentScene;
    };
#define CURRENT_SCENE_SCOPE(s) Scene::CurrentSceneScope currentSceneScope(s);
private:

    static void setCurrentScene(Scene* s);

    GPU_ONLY DEVICEPTR(ITMVoxelBlock*) getVoxelBlock(VoxelBlockPos pos);

     public: // these two could be private where it not for testing/debugging
    DEVICEPTR(ITMVoxelBlock*) localVBA;
   
    /// Gives indices into localVBA for allocated voxel blocks
    HashMap<Z3Hasher, AllocateVB>* voxelBlockHash;

};
