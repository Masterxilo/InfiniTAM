#pragma once 
#include "itmlibdefines.h"
#include "cudadefines.h"
#include "hashmap.h"

// T must have a function process(ITMVoxelBlock*)
template<typename T>
KERNEL doForEachAllocatedBlock(
    ITMVoxelBlock* localVBA,
    int nextFreeSequenceId) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index <= 0 || index >= nextFreeSequenceId) return;

    T::process(&localVBA[index]);
}


/// Must be heap-allocated
class Scene : public Managed {
public:
    /// \returns NULL when the voxel was not found
    GPU_ONLY ITMVoxel* getVoxel(Vector3i pos);

    /// Returns NULL if the voxel block is not allocated
    GPU_ONLY void requestVoxelBlockAllocation(VoxelBlockPos pos);
    void performAllocations();

    Scene();
    virtual ~Scene();


    // T must have an operator(ITMVoxelBlock*)
    template<typename T>
    void doForEachAllocatedBlock() {
        const float blockSize = 256.f;
        int lfs = voxelBlockHash->getLowestFreeSequenceNumber();

        int gridSize = ceil(voxelBlockHash->getLowestFreeSequenceNumber() / blockSize);
        
        assert(gridSize);
        LAUNCH_KERNEL(
            ::doForEachAllocatedBlock<T>, 
            gridSize,
            blockSize,
            localVBA,
            voxelBlockHash->getLowestFreeSequenceNumber()
            );
    }

    // Scene is mostly fixed. // TODO prefer using a scoping construct that lives together with the call stack!
    // Having it globally accessible heavily reduces having
    // to pass parameters.
    static CPU_AND_GPU Scene* getCurrentScene();
    static GPU_ONLY ITMVoxel* getCurrentSceneVoxel(Vector3i pos) {
        return getCurrentScene()->getVoxel(pos);
    }
    static GPU_ONLY void requestCurrentSceneVoxelBlockAllocation(VoxelBlockPos pos) {
        return getCurrentScene()->requestVoxelBlockAllocation(pos);
    }


    static void setCurrentScene(Scene* s);

    /** !private! */
    struct Z3Hasher {
        typedef VoxelBlockPos KeyType;
        static const uint BUCKET_NUM = SDF_BUCKET_NUM; // Number of Hash Bucket, must be 2^n (otherwise we have to use % instead of & below)

        static GPU_ONLY uint hash(const VoxelBlockPos& blockPos) {
            return (((uint)blockPos.x * 73856093u) ^ ((uint)blockPos.y * 19349669u) ^ ((uint)blockPos.z * 83492791u))
                &
                (uint)(BUCKET_NUM - 1);
        }
    };

    /** !private! */
    struct AllocateVB {
        static __device__ void allocate(VoxelBlockPos pos, int sequenceId);
    };
private:
    GPU_ONLY DEVICEPTR(ITMVoxelBlock*) getVoxelBlock(VoxelBlockPos pos);
    DEVICEPTR(ITMVoxelBlock*) localVBA;

    /// Gives indices into localVBA for allocated voxel blocks
    HashMap<Z3Hasher, AllocateVB>* voxelBlockHash;

};
