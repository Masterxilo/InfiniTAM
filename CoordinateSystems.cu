#include "itmlibdefines.h"

__managed__ CoordinateSystem* globalcs = 0;


/// (0,0,0) is the lower corner of the first voxel block, (1,1,1) its upper corner,
/// corresponding to (voxelBlockSize, voxelBlockSize, voxelBlockSize) in world coordinates.
__managed__ CoordinateSystem* voxelBlockCoordinates = 0;

/// (0,0,0) is the lower corner of the voxel, (1,1,1) its upper corner,
/// corresponding to (voxelSize, voxelSize, voxelSize) in world coordinates.
__managed__ CoordinateSystem* voxelCoordinates = 0;
void initCoordinateSystems() {
    CoordinateSystem::global(); // make sure it exists

    Matrix4f m;

    if (voxelBlockCoordinates) {
        assert(voxelCoordinates); return;
    }
    m.setIdentity(); m.setScale(voxelBlockSize);
    voxelBlockCoordinates = new CoordinateSystem(m);

    m.setIdentity(); m.setScale(voxelSize);
    voxelCoordinates = new CoordinateSystem(m);
}