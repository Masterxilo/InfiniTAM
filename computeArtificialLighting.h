#include "itmlibdefines.h"
/**
computes max(voxelNormal . lightNormal, 0) at each voxel in the current scene
where the normal can be computed and stores it as its color
*/
void computeArtificialLighting(Vector3f lightNormal);