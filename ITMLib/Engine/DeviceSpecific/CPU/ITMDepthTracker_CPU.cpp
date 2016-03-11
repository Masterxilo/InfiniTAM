// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMDepthTracker_CPU.h"
#include "../../DeviceAgnostic/ITMDepthTracker.h"

using namespace ITMLib::Engine;

ITMDepthTracker_CPU::ITMDepthTracker_CPU(Vector2i imgSize, TrackerIterationType *trackingRegime, int noHierarchyLevels, int noICPRunTillLevel,
	float distThresh, float terminationThreshold, const ITMLowLevelEngine *lowLevelEngine) :ITMDepthTracker(imgSize, trackingRegime, noHierarchyLevels,
	noICPRunTillLevel, distThresh, terminationThreshold, lowLevelEngine, MEMORYDEVICE_CPU) { }

ITMDepthTracker_CPU::~ITMDepthTracker_CPU(void) { }

ITMDepthTracker::AccuCell ITMDepthTracker_CPU::ComputeGandH(Matrix4f T_g_k_estimate) {
    // Alias/0-init variables
	Vector4f *pointsMap = sceneHierarchyLevel->pointsMap->GetData(MEMORYDEVICE_CPU);
	Vector4f *normalsMap = sceneHierarchyLevel->normalsMap->GetData(MEMORYDEVICE_CPU);
	Vector4f sceneIntrinsics = sceneHierarchyLevel->intrinsics;
	Vector2i sceneImageSize = sceneHierarchyLevel->pointsMap->noDims;

	float *depth = viewHierarchyLevel->depth->GetData(MEMORYDEVICE_CPU);
	Vector4f viewIntrinsics = viewHierarchyLevel->intrinsics;
	Vector2i viewImageSize = viewHierarchyLevel->depth->noDims;

    AccuCell accu = make0Accu();

    // Loop over pixels (image domain u \in U)
	for (int y = 0; y < viewImageSize.y; y++) for (int x = 0; x < viewImageSize.x; x++)
	{
        AccuCell u_accu = make0Accu();
        // Get f (summand of energy), ATA and ATb for current pixel u if u isValidPoint
#define iteration(iterationType) \
			iterationType: u_accu.noValidPoints = computePerPointGH_Depth<iterationType>(u_accu.ATb, u_accu.AT_A_tri, u_accu.f, x, y, depth[x + y * viewImageSize.x], viewImageSize,\
        viewIntrinsics, sceneImageSize, sceneIntrinsics, T_g_k_estimate, scenePose, pointsMap, normalsMap, distThresh[levelId]); break;
		switch (iterationType) {
            case iteration(TRACKER_ITERATION_ROTATION);
            case iteration(TRACKER_ITERATION_TRANSLATION);
            case iteration(TRACKER_ITERATION_BOTH);
		}
#undef iteration
        addTo(accu, u_accu);
	}
    return accu;
}
