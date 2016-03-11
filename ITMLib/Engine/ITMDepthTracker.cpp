/// \file c.f. newcombe_etal_ismar2011.pdf
// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMDepthTracker.h"
#include "../../ORUtils/Cholesky.h"

#include <math.h>

using namespace ITMLib::Engine;

ITMDepthTracker::ITMDepthTracker(Vector2i imgSize, TrackerIterationType *trackingRegime, int noHierarchyLevels, int noICPRunTillLevel, float distThresh,
	float terminationThreshold, const ITMLowLevelEngine *lowLevelEngine, MemoryDeviceType memoryType)
{
	viewHierarchy = new ITMImageHierarchy<ITMTemplatedHierarchyLevel<ITMFloatImage> >(
        imgSize, trackingRegime, noHierarchyLevels, memoryType, true);
	sceneHierarchy = new ITMImageHierarchy<ITMSceneHierarchyLevel>(imgSize, trackingRegime, noHierarchyLevels, memoryType, true);

	this->lowLevelEngine = lowLevelEngine;

	this->noICPLevel = noICPRunTillLevel;

	this->terminationThreshold = terminationThreshold;

    // Init per level noIterationsPerLevel
	this->noIterationsPerLevel = new int[noHierarchyLevels];
	
	this->noIterationsPerLevel[0] = 2; //TODO -> make parameter
	for (int levelId = 1; levelId < noHierarchyLevels; levelId++)
	{
		noIterationsPerLevel[levelId] = noIterationsPerLevel[levelId - 1] + 2;
	}

    // Init per level distThresh
    this->distThresh = new float[noHierarchyLevels];
	float distThreshStep = distThresh / noHierarchyLevels;
	this->distThresh[noHierarchyLevels - 1] = distThresh;
	for (int levelId = noHierarchyLevels - 2; levelId >= 0; levelId--)
		this->distThresh[levelId] = this->distThresh[levelId + 1] - distThreshStep;
}

ITMDepthTracker::~ITMDepthTracker(void) 
{ 
	delete this->viewHierarchy;
	delete this->sceneHierarchy;

	delete[] this->noIterationsPerLevel;
	delete[] this->distThresh;
}

void ITMDepthTracker::SetEvaluationData(ITMTrackingState *trackingState, const ITMView *view)
{
	this->view = view;

	sceneHierarchy->levels[0]->intrinsics = view->calib->intrinsics_d.projectionParamsSimple.all;
	viewHierarchy->levels[0]->intrinsics = view->calib->intrinsics_d.projectionParamsSimple.all;

	// the image hierarchy allows pointers to external data at level 0
	viewHierarchy->levels[0]->depth = view->depth;
	sceneHierarchy->levels[0]->pointsMap = trackingState->pointCloud->locations;
	sceneHierarchy->levels[0]->normalsMap = trackingState->pointCloud->colours;

	scenePose = trackingState->pose_pointCloud->GetM();
}

void ITMDepthTracker::PrepareForEvaluation()
{
	for (int i = 1; i < viewHierarchy->noLevels; i++)
	{
		ITMTemplatedHierarchyLevel<ITMFloatImage> *currentLevelView = viewHierarchy->levels[i], *previousLevelView = viewHierarchy->levels[i - 1];
		lowLevelEngine->FilterSubsampleWithHoles(currentLevelView->depth, previousLevelView->depth);
		currentLevelView->intrinsics = previousLevelView->intrinsics * 0.5f;

		ITMSceneHierarchyLevel *currentLevelScene = sceneHierarchy->levels[i], *previousLevelScene = sceneHierarchy->levels[i - 1];
		//lowLevelEngine->FilterSubsampleWithHoles(currentLevelScene->pointsMap, previousLevelScene->pointsMap);
		//lowLevelEngine->FilterSubsampleWithHoles(currentLevelScene->normalsMap, previousLevelScene->normalsMap);
		currentLevelScene->intrinsics = previousLevelScene->intrinsics * 0.5f;
	}
}

void ITMDepthTracker::SetEvaluationParams(int levelId)
{
	this->levelId = levelId;
	this->iterationType = viewHierarchy->levels[levelId]->iterationType;
	this->sceneHierarchyLevel = sceneHierarchy->levels[0];
	this->viewHierarchyLevel = viewHierarchy->levels[levelId];
}

void ITMDepthTracker::ComputeDelta(float *step, float *nabla, float *hessian) const
{
	for (int i = 0; i < 6; i++) step[i] = 0;

    if (shortIteration())
	{
        // Keep only upper 3x3 part of hessian
		float smallHessian[3 * 3];
		for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) smallHessian[r + c * 3] = hessian[r + c * 6];

        ORUtils::Cholesky::solve(smallHessian, 3, nabla, step);
	}
	else
	{
        ORUtils::Cholesky::solve(hessian, 6, nabla, step);
	}
}

bool ITMDepthTracker::HasConverged(float *step) const
{
    // Compute ||step||_2^2
	float stepLength = 0.0f;
	for (int i = 0; i < 6; i++) stepLength += step[i] * step[i];

    // heuristic? Why /6?
	if (sqrt(stepLength) / 6 < terminationThreshold) return true; //converged

	return false;
}

Matrix4f ITMDepthTracker::ComputeTinc(const float *delta) const
{
    // step is T_inc, expressed as a parameter vector 
    // (beta, gamma, alpha, tx,ty, tz)
    // beta, gamma, alpha parametrize the rotation axis
	float step[6];

    // Depending on the iteration type, fill in 0 for values that where not computed.
	switch (iterationType)
	{
	case TRACKER_ITERATION_ROTATION:
		step[0] = (float)(delta[0]); step[1] = (float)(delta[1]); step[2] = (float)(delta[2]);
		step[3] = 0.0f; step[4] = 0.0f; step[5] = 0.0f;
		break;
	case TRACKER_ITERATION_TRANSLATION:
		step[0] = 0.0f; step[1] = 0.0f; step[2] = 0.0f;
		step[3] = (float)(delta[0]); step[4] = (float)(delta[1]); step[5] = (float)(delta[2]);
		break;
	default:
	case TRACKER_ITERATION_BOTH:
		step[0] = (float)(delta[0]); step[1] = (float)(delta[1]); step[2] = (float)(delta[2]);
		step[3] = (float)(delta[3]); step[4] = (float)(delta[4]); step[5] = (float)(delta[5]);
		break;
	}

    // Incremental pose update assuming small angles.
	Matrix4f Tinc;

	Tinc.m00 = 1.0f;		Tinc.m10 = step[2];		Tinc.m20 = -step[1];	Tinc.m30 = step[3];
	Tinc.m01 = -step[2];	Tinc.m11 = 1.0f;		Tinc.m21 = step[0];		Tinc.m31 = step[4];
	Tinc.m02 = step[1];		Tinc.m12 = -step[0];	Tinc.m22 = 1.0f;		Tinc.m32 = step[5];
	Tinc.m03 = 0.0f;		Tinc.m13 = 0.0f;		Tinc.m23 = 0.0f;		Tinc.m33 = 1.0f;
    return Tinc;
}

/// \file c.f. newcombe_etal_ismar2011.pdf, Sensor Pose Estimation section
void ITMDepthTracker::TrackCamera(ITMTrackingState *trackingState, const ITMView *view)
{
    this->SetEvaluationData(trackingState, view);
	this->PrepareForEvaluation();

    // Coarse to fine
	for (int levelId = viewHierarchy->noLevels - 1; levelId >= noICPLevel; levelId--)
	{
		if (iterationType == TRACKER_ITERATION_NONE) continue;
		this->SetEvaluationParams(levelId);

#define T_k_g_estimate trackingState->pose_d
        // T_g_k_estimate caches T_k_g_estimate->GetInvM()
        Matrix4f T_g_k_estimate = T_k_g_estimate->GetInvM();

#define set_T_k_g_estimate(x)\
        T_k_g_estimate->SetFrom(&x);
        T_g_k_estimate = T_k_g_estimate->GetInvM();

#define set_T_k_g_estimate_from_T_g_k_estimate(x) \
        T_k_g_estimate->SetInvM(x);\
		T_k_g_estimate->Coerce(); /* and make sure we've got an SE3*/\
        T_g_k_estimate = T_k_g_estimate->GetInvM();

        // We will 'accept' updates into trackingState->pose_d and T_g_k_estimate
        // before we know whether they actually decrease the energy.
        // When they did not in fact, we will revert to this value that was known to have less energy 
        // than all previous estimates.
		ITMPose least_energy_T_k_g_estimate(*(trackingState->pose_d));
		
        // Track least energy we measured so far to see whether we improved
        float f_old = 1e20f;

        // current levenberg-marquart style damping parameter, often called mu.
		float lambda = 1.0;

        // Iterate as required
		for (int iterNo = 0; iterNo < noIterationsPerLevel[levelId]; iterNo++)
        {
            if (iterationType == TRACKER_ITERATION_NONE) continue;

            // [ this takes most time. 
            // Computes f(x) as well as A^TA and A^Tb for next computation of delta_x as
            // (A^TA + lambda * diag(A^TA)) delta_x = A^T b
            // if f decreases, the delta is applied definitely, otherwise x is reset.
            // So we do:
            /*
            x = x_best;
            lambda = 1;
            f_best = infinity

            repeat:
            compute f_new, A^TA_new, A^T b_new

            if (f_new > f_best) {x = x_best; lambda *= 10;}
            else {
            x_best = x;
            A^TA = A^TA_new
            A^Tb = A^Tb_new
            }

            solve (A^TA + lambda * diag(A^TA)) delta_x = A^T b
            x += delta_x;

            */


			// evaluate error function at currently accepted
            // T_g_k_estimate
            // and compute information for next update
            float f_new;
            int noValidPoints;
            float new_sum_ATb[6];
            float new_sum_AT_A[6 * 6];
            noValidPoints = this->ComputeGandH(f_new, new_sum_ATb, new_sum_AT_A, T_g_k_estimate);
            // ]]

            float least_energy_sum_AT_A[6 * 6], 
                damped_least_energy_sum_AT_A[6 * 6];
            float least_energy_sum_ATb[6];

			// check if energy actually *increased* with the last update
            // Note: This happens rarely, namely when the blind 
            // gauss-newton step actually leads to an *increase in energy
            // because the damping was too small
            if ((noValidPoints <= 0) || (f_new > f_old)) {
                // If so, revert pose and discard/ignore new_sum_AT_A, new_sum_ATb
                // TODO would it be worthwhile to not compute these when they are not going to be used?
                set_T_k_g_estimate(least_energy_T_k_g_estimate);
                // Increase damping, then solve normal equations again with old matrix (see below)
				lambda *= 10.0f;
            }
            else {
                f_old = f_new;
                least_energy_T_k_g_estimate.SetFrom(T_k_g_estimate);

                // Prepare to solve a new system

                // Preconditioning
                for (int i = 0; i < 6 * 6; ++i) least_energy_sum_AT_A[i] = new_sum_AT_A[i] / noValidPoints;
                for (int i = 0; i < 6; ++i) least_energy_sum_ATb[i] = new_sum_ATb[i] / noValidPoints;

                // Accept and decrease damping
				lambda /= 10.0f;
			}
            // Solve normal equations

            // Apply levenberg-marquart style damping (multiply diagonal of ATA by 1.0f + lambda)
            for (int i = 0; i < 6 * 6; ++i) damped_least_energy_sum_AT_A[i] = least_energy_sum_AT_A[i];
            for (int i = 0; i < 6; ++i) damped_least_energy_sum_AT_A[i + i * 6] *= 1.0f + lambda;

			// compute the update step parameter vector x
            float x[6];
            ComputeDelta(x, 
                least_energy_sum_ATb,
                damped_least_energy_sum_AT_A);

            // Apply the corresponding Tinc
            set_T_k_g_estimate_from_T_g_k_estimate(
                /* T_g_k_estimate = */
                ComputeTinc(x) * T_g_k_estimate
                );

			// if step is small, assume it's going to decrease the error and finish
			if (HasConverged(x)) break;
		}
	}

    // Convert T_g_k (k to global) to T_k_g (global to k)
}

