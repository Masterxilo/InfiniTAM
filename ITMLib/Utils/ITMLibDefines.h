#pragma once

#include "PlatformIndependence.h"
#include "ITMMath.h"
#include "cudadefines.h"


//////////////////////////////////////////////////////////////////////////
// Voxel Hashing definition and helper functions
//////////////////////////////////////////////////////////////////////////

#define SDF_BLOCK_SIZE 8
#define SDF_BLOCK_SIZE3 (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE)
#define SDF_LOCAL_BLOCK_NUM 0x40000		// Number of locally stored blocks (maximum load the hash-table can have)

#define SDF_BUCKET_NUM 0x100000			// Number of Hash Bucket, must be 2^n and bigger than SDF_LOCAL_BLOCK_NUM
#define SDF_HASH_MASK (SDF_BUCKET_NUM-1)// Used for get hashing value of the bucket index, "x & (uint)SDF_HASH_MASK" is the same as "x % SDF_BUCKET_NUM"
#define SDF_EXCESS_LIST_SIZE 0x20000	// Size of excess list, used to handle collisions. Also max offset (unsigned short) value.

#define SDF_GLOBAL_BLOCK_NUM (SDF_BUCKET_NUM+SDF_EXCESS_LIST_SIZE)	// Number of globally stored blocks == size of ordered + unordered part of hash table


//////////////////////////////////////////////////////////////////////////
// Voxel Hashing data structures
//////////////////////////////////////////////////////////////////////////

/** 
Coarsest integer grid laid over 3d space.

Multiply by SDF_BLOCK_SIZE to get voxel coordinates,
    and then by ITMSceneParams::voxelSize to get world coordinates.

using short to reduce storage of hash map. // TODO could use another type for accessing convenience/alignment speed
*/
typedef Vector3s VoxelBlockPos;
// Default voxel block pos, used for debugging
#define INVALID_VOXEL_BLOCK_POS Vector3s(SHRT_MIN, SHRT_MIN, SHRT_MIN)

/** 
    A single entry in the hash table (hash table bucket).
*/
struct ITMHashEntry
{
	/** Position of the corner of the 8x8x8 volume, that identifies the entry. 
    In voxel-block coordinates. Multiply by SDF_BLOCK_SIZE to get voxel coordinates,
    and then by ITMSceneParams::voxelSize to get world coordinates.
    */
    VoxelBlockPos pos;

	/** 1-based position of the 'next'
    entry in the excess list. 
    Used as SDF_BUCKET_NUM + hashEntry.offset - 1
    to compute the next hashIdx.
    This value is at most SDF_EXCESS_LIST_SIZE.*/
	int offset;
    CPU_AND_GPU bool hasExcessListOffset() { return offset >= 1; }
	CPU_AND_GPU int getHashIndexOfNextExcessEntry() {
        return SDF_BUCKET_NUM + offset - 1;
    }
    
    /** Offset into the voxel block array (* voxelData).
	    - >= 0 identifies an actual allocated entry in the voxel block array
	    - -1 identifies an entry that has been removed (swapped out)
	    - <-1 identifies an unallocated block

        Multiply with SDF_BLOCK_SIZE3 and add offset to access individual TVoxels.

        "store a pointer to the location in a large vloxel block array, where the T-SDF data of all the
        blocks is serially stored"
	*/
    int ptr;

    CPU_AND_GPU bool isAllocatedAndActive() const { return ptr >= 0; }

    /// Was once allocated, but is maybe not in active memory.
    /// But this space is permanently reserved.
    CPU_AND_GPU bool isAllocated() const { return ptr >= -1; }

    CPU_AND_GPU bool isUnallocated() const { return ptr < -1; }

    // an unallocated entry, used for resetting
    static ITMHashEntry createIllegalEntry() {
        ITMHashEntry tmpEntry;
        memset(&tmpEntry, 0, sizeof(ITMHashEntry));
        tmpEntry.ptr = -2;
        return tmpEntry;
    }
};

// #define USE_FLOAT_SDF_STORAGE // uses 4 instead of 2 bytes
/** \brief
    Stores the information of a single voxel in the volume
*/
class ITMVoxel
{   
private:
    // signed distance
    short sdf;  // saving storage
public:
    /** Value of the truncated signed distance transformation, in [-1, 1] (scaled by truncation mu when storing) */
	CPU_AND_GPU void setSDF_initialValue() { sdf = 32767; }
    CPU_AND_GPU float getSDF() { return (float)(sdf) / 32767.0f; }
    CPU_AND_GPU void setSDF(float x) {
        assert(x >= -1 && x <= 1);
        sdf = (short)((x)* 32767.0f);
    }

	/** Number of fused observations that make up @p sdf. */
	uchar w_depth;

	/** RGB colour information stored for this voxel, 0-255 per channel. */
	Vector3u clr; // C(v) 

	/** Number of observations that made up @p clr. */
	uchar w_color;

    //! unknowns of our objective
    float luminanceAlbedo; // a(v)
    //float refinedDistance; // D'(v)

    // chromaticity and intensity are
    // computed from C(v) on-the-fly

    /// \f$\Gamma(v)\f$
    float intensity() {
        // TODO is this how luminance should be computed?
        Vector3f color = clr.toFloat() / 255.f;
        return (color.r + color.g + color.b) / 3.f;
    }

    Vector3f chromaticity() {
        return clr.toFloat() / intensity();
    }

    GPU_ONLY ITMVoxel()
	{
        setSDF_initialValue();
		w_depth = 0;
		clr = (uchar)0;
		w_color = 0;
	}
};

struct ITMVoxelBlock {
    GPU_ONLY void resetVoxels() {
        for (auto& i : blockVoxels) i = ITMVoxel();
    }
    /// compute voxelLocalId to access blockVoxels
    GPU_ONLY ITMVoxel* getVoxel(Vector3i localPos) {
        assert(localPos.x >= 0 && localPos.x < SDF_BLOCK_SIZE);
        assert(localPos.y >= 0 && localPos.y < SDF_BLOCK_SIZE);
        assert(localPos.z >= 0 && localPos.z < SDF_BLOCK_SIZE);

        return &blockVoxels[
            localPos.x + localPos.y * SDF_BLOCK_SIZE + localPos.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE
        ];
    }

    GPU_ONLY VoxelBlockPos getPos() const {
        return pos_;
    }

    __declspec(property(get = getPos)) VoxelBlockPos pos;

    /// Initialize pos and reset data
    GPU_ONLY void reinit(VoxelBlockPos pos) {
        pos_ = pos;

        // resetVoxels
        for (auto& i : blockVoxels) i = ITMVoxel();
    }

private:
    /// pos is Mutable, 
    /// because this voxel block might represent any part of space
    VoxelBlockPos pos_;

    ITMVoxel blockVoxels[SDF_BLOCK_SIZE3];
};


/// The tracker iteration type used to define the tracking iteration regime
enum TrackerIterationType
{
    /// Update only the current rotation estimate. This is preferable for the coarse solution stages.
    TRACKER_ITERATION_ROTATION = 1,
    TRACKER_ITERATION_BOTH = 3,
    TRACKER_ITERATION_NONE = 4
};

#include "Image.h"

#define ITMFloatImage ORUtils::Image<float>
#define ITMFloat2Image ORUtils::Image<Vector2f>
#define ITMFloat4Image ORUtils::Image<Vector4f>
#define ITMShortImage ORUtils::Image<short>
#define ITMShort3Image ORUtils::Image<Vector3s>
#define ITMShort4Image ORUtils::Image<Vector4s>
#define ITMUShortImage ORUtils::Image<ushort>
#define ITMUIntImage ORUtils::Image<uint>
#define ITMIntImage ORUtils::Image<int>
#define ITMUCharImage ORUtils::Image<uchar>
#define ITMUChar4Image ORUtils::Image<Vector4u>
#define ITMBoolImage ORUtils::Image<bool>

#include "ITMLibSettings.h" // must be included after tracker iteration type is defined

#define INVALID_DEPTH (-1.f)


#include "CoordinateSystem.h"

/// (0,0,0) is the lower corner of the first voxel block, (1,1,1) its upper corner,
/// corresponding to (voxelBlockSize, voxelBlockSize, voxelBlockSize) in world coordinates.
extern __managed__ CoordinateSystem* voxelBlockCoordinates;

/// (0,0,0) is the lower corner of the voxel, (1,1,1) its upper corner,
/// corresponding to (voxelSize, voxelSize, voxelSize) in world coordinates.
extern __managed__ CoordinateSystem* voxelCoordinates;
extern void initCoordinateSystems();
