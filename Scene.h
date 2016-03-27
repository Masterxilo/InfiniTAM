#pragma once 
#include "itmlibdefines.h"
#include "cudadefines.h"
#include "hashmap.h"
/// Must be head allocated
class Scene : public Managed {
public:
    /// \returns NULL when the voxel was not found
    GPU_ONLY ITMVoxel* getVoxel(Vector3i pos);

    /// Returns NULL if the voxel block is not allocated
    GPU_ONLY void requestVoxelBlockAllocation(VoxelBlockPos pos);
    void performAllocations();

    Scene();
    virtual ~Scene();

    static CPU_AND_GPU Scene* getCurrentScene();
    static void setCurrentScene(Scene* s);

    /* !private! */
    struct Z3Hasher {
        typedef VoxelBlockPos KeyType;
        static const uint BUCKET_NUM = SDF_BUCKET_NUM; // Number of Hash Bucket, must be 2^n (otherwise we have to use % instead of & below)

        static GPU_ONLY uint hash(const VoxelBlockPos& blockPos) {
            return (((uint)blockPos.x * 73856093u) ^ ((uint)blockPos.y * 19349669u) ^ ((uint)blockPos.z * 83492791u))
                &
                (uint)(BUCKET_NUM - 1);
        }
    };

private:
    GPU_ONLY ITMVoxelBlock* getVoxelBlock(VoxelBlockPos pos);

    DEVICEPTR(ITMVoxelBlock*) localVBA;

   

    /// Gives indices into localVBA for allocated voxel blocks
    HashMap<Z3Hasher>* voxelBlockHash;

};
