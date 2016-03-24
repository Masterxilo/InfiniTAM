// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMLibDefines.h"
#include "ITMTracker.h"
#include "ITMLowLevelEngine.h"

using namespace ITMLib::Objects;

#include <vector>

namespace ITMLib
{
	namespace Engine
	{
		/** Performing ICP based depth tracking. 
            Implements the original KinectFusion tracking algorithm.

            c.f. newcombe_etal_ismar2011.pdf section "Sensor Pose Estimation"

            6-d parameter vector "x" is (beta, gamma, alpha, tx, ty, tz)
		*/
		class ITMDepthTracker : public ITMTracker
        {
        public:

            void TrackCamera(
                ITMTrackingState *trackingState, //!< [in,out] in: the current and out: the computed best fit adjusted camera pose for the new view
                const ITMView *view //<! latest camera data, for which the camera pose shall be adjusted
                );


            ITMTrackingState *BuildTrackingState() const
            {
                return new ITMTrackingState(trackingLevels[0].depth->noDims);
            }

            ITMDepthTracker(
                Vector2i depthImgSize,
                float distThresh,
                float terminationThreshold,
                const ITMLowLevelEngine *lowLevelEngine);
            virtual ~ITMDepthTracker(void);

            /** !private! */
            struct AccuCell {
                int noValidPoints;
                float f;
                // ATb
                float ATb[6];
                // AT_A_tri, upper right triangular part of AT_A
                float AT_A_tri[1 + 2 + 3 + 4 + 5 + 6];
            };
        private:

            AccuCell make0Accu() {
                AccuCell accu;
                accu.noValidPoints = 0;
                accu.f = 0.0f;
                memset(accu.AT_A_tri, 0, sizeof(float) * noParaSQ());
                memset(accu.ATb, 0, sizeof(float) * noPara());
                return accu;
            }
            /// Add all attributes of b to a if b.noValidPoints > 0
            void addTo(AccuCell& a, const AccuCell& b) {
                if (b.noValidPoints == 0) return;
                a.noValidPoints += b.noValidPoints;
                a.f += b.f;
                for (int i = 0; i < noPara(); i++) a.ATb[i] += b.ATb[i];
                for (int i = 0; i < noParaSQ(); i++) a.AT_A_tri[i] += b.AT_A_tri[i];
            }


			const ITMLowLevelEngine *lowLevelEngine;
            /// Passed on last call to TrackCamera
            const ITMView *view;

			const float terminationThreshold;

            /// Solves hessian.step = nabla
            /// \param delta output array of 6 floats 
            /// \param hessian 6x6
            /// \param delta 3 or 6
            /// \param nabla 3 or 6
            /// \param shortIteration whether there are only 3 parameters
            void ComputeDelta(float *delta, float *nabla, float *hessian) const; 
            Matrix4f ComputeTinc(const float *delta) const;
			bool HasConverged(float *step) const;

            struct TrackingLevel {
                /// FilterSubsampleWithHoles result of one level higher
                ITMFloatImage* depth;
                /// Half of the intrinsics of one level higher
                Vector4f intrinsics;
                float distanceThreshold;
                int numberOfIterations;
                TrackerIterationType iterationType;
                TrackingLevel() {}
                TrackingLevel(int numberOfIterations, TrackerIterationType iterationType, float distanceThreshold, Vector2i depthImageSize) :
                    numberOfIterations(numberOfIterations), iterationType(iterationType), distanceThreshold(distanceThreshold) {
                    depth = new ITMFloatImage(depthImageSize, MEMORYDEVICE_CUDA);
                }
            };
            // ViewHierarchy, 0 is highest resolution
            std::vector<TrackingLevel> trackingLevels;
            TrackingLevel* currentTrackingLevel;

            Vector4f getViewIntrinsics() {
                return view->calib->intrinsics_d.projectionParamsSimple.all;
            }

            TrackerIterationType iterationType() const {
                return currentTrackingLevel->iterationType;
            }
            bool shortIteration() const {
                return (iterationType() == TRACKER_ITERATION_ROTATION) ||
                    (iterationType() == TRACKER_ITERATION_TRANSLATION);
            }

            int noPara() const  {
                return shortIteration() ? 3 : 6;
            }

            int noParaSQ() const  {
                return shortIteration() ? 3 + 2 + 1 : 6 + 5 + 4 + 3 + 2 + 1;
            }

            /// Of trackingState passed to TrackCamera
            ORUtils::Image<Vector4f> *pointsMap, *normalsMap;
            Matrix4f scenePose;

            /// evaluate error function at the current T_g_k_estimate, 
            /// compute sum_ATb and sum_AT_A, the system we need to solve to compute the
            /// next update step (note: this system is not yet solved and we don't know the new energy yet!)
            /// \returns noValidPoints
            int ComputeGandH(
                float &f,
                float *sum_ATb,
                float *sum_AT_A,
                Matrix4f T_g_k_estimate) {
                AccuCell accu = ComputeGandH(T_g_k_estimate);

                memcpy(sum_ATb, accu.ATb, sizeof(float) * noPara());

                // output sum_AT_A, sum_ATb
                // Construct full output (hessian) matrix from accumulated sum
                // lower right triangular part
                for (int r = 0, counter = 0; r < noPara(); r++)
                    for (int c = 0; c <= r; c++)
                        sum_AT_A[r * 6 + c] = accu.AT_A_tri[counter++]; // here, r is bigger than c 
                // Symmetric part
                for (int r = 0; r < noPara(); ++r)
                    for (int c = r + 1; c < noPara(); c++)
                        sum_AT_A[r * 6 + c] = sum_AT_A[c * 6 + r]; // here, c is bigger than r, that part was initialized above

                // Output energy -- if we have very few points, output some high energy
                f = (accu.noValidPoints > 100) ? sqrt(accu.f) / accu.noValidPoints : 1e5f;

                return accu.noValidPoints;
            }

            /// CUDA managed host/device memory (call cudaDeviceSynchronize before using on host)
            AccuCell *accu;

            AccuCell ComputeGandH(Matrix4f T_g_k_estimate);

		};
	}
}
