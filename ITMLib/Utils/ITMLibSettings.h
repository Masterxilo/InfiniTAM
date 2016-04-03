#pragma once

/// depth threashold for the ICP tracker
/// For ITMDepthTracker: ICP distance threshold
#define depthTrackerICPThreshold (0.1f * 0.1f)
/// For ITMDepthTracker: ICP iteration termination threshold
#define depthTrackerTerminationThreshold 1e-3f
/** @} */
/** \brief
Encodes the width of the band of the truncated
signed distance transform that is actually stored
in the volume. This is again usually specified in
meters. The resulting width in voxels is @ref mu
divided by @ref voxelSize.

Must be greater than voxelSize.
*/
#define mu 0.02f
/** \brief
Up to @ref maxW observations per voxel are averaged.
Beyond that a sliding average is computed.
*/
#define maxW 100
/// Size of a voxel, usually given in meters.
/// In world space coordinates. 
#define voxelSize 0.005f
#define oneOverVoxelSize (1.0f / voxelSize)
/// In world space coordinates.
#define voxelBlockSize (voxelSize*SDF_BLOCK_SIZE)
#define oneOverVoxelBlockWorldspaceSize (1.0f / (voxelBlockSize))
/** @{ */
/** \brief
Fallback parameters: consider only parts of the
scene from @p viewFrustum_min in front of the camera
to a distance of @p viewFrustum_max. Usually the
actual depth range should be determined
automatically by a ITMLib::Engine::ITMVisualisationEngine.
*/
#define viewFrustum_min 0.2f
#define viewFrustum_max 3.0f
