#pragma once

/// depth threashold for the ICP tracker
/// For ITMDepthTracker: ICP distance threshold for lowest resolution (later iterations use lower distances)
/// In world space squared -- TODO maybe define heuristically from voxelsize/mus
#define depthTrackerICPMaxThreshold (0.1f * 0.1f)

/// For ITMDepthTracker: ICP iteration termination threshold
#define depthTrackerTerminationThreshold 1e-3f
/** @} */
/** \brief
Encodes the width of the band of the truncated
signed distance transform that is actually stored
in the volume. This is again usually specified in
meters (world coordinates). 
Note that thus, the resulting width in voxels is @ref mu
divided by @ref voxelSize (times two -> on both sides of the surface).
Also, a voxel storing the value 1 has world-space-distance mu from the surface.

Must be greater than voxelSize.

TODO define from voxelSize
*/
#define mu 0.02f

/**
Size of the thin shell region for volumetric refinement-from-shading computation.
Must be smaller than mu and should be bigger than voxelSize

In world space coordinates (meters).
*/
#define t_shell (mu/2.f)

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
to a distance of @p viewFrustum_max (world-space distance). Usually the
actual depth range should be determined
automatically by a ITMLib::Engine::ITMVisualisationEngine.
*/
#define viewFrustum_min 0.2f
#define viewFrustum_max 3.0f
