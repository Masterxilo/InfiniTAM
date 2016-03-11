/// \file please document this class
// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Utils/ITMLibDefines.h"

#include "../Objects/ITMImageHierarchy.h"
#include "../Objects/ITMViewHierarchyLevel.h"

#include "../Engine/ITMTracker.h"
#include "../Engine/ITMLowLevelEngine.h"

using namespace ITMLib::Objects;

namespace ITMLib
{
	namespace Engine
	{
		/** Base class for engines performing point based colour
		    tracking. Implementations would typically project down a
		    point cloud into observed images and try to minimize the
		    reprojection error.

            f is the objective function: \$R^3 \cross R^3 \to R\$
            we optimize for, (camera translation and rotation),
            g is its gradient.
		*/
		class ITMColorTracker : public ITMTracker
		{
		private:
			const ITMLowLevelEngine *lowLevelEngine;

            /// Computes the subsampled images and gradients for each hierarchy level.
			void PrepareForEvaluation(const ITMView *view);

		protected: 
			TrackerIterationType iterationType;
			ITMTrackingState *trackingState; const ITMView *view;
			ITMImageHierarchy<ITMViewHierarchyLevel> *viewHierarchy;
			int levelId;

			int countedPoints_valid;
		public:
            /// Represents f and its gradient g evaluated at
            /// some point \f$x \in R^3 \cross R^3\$ = ITMPose.
            /// x === para
			class EvaluationPoint
			{
			public:
				float f(void) { return cacheF; }
				const float* nabla_f(void) { if (cacheNabla == NULL) computeGradients(false); return cacheNabla; }

				const float* hessian_GN(void) { if (cacheHessian == NULL) computeGradients(true); return cacheHessian; }
				/// parameter passed at initialization
                const ITMPose & getParameter(void) const { return *mPara; }

                /// \param para Manages the memory pointed to by para, expected to lie on the heap.
                /// calls F_oneLevel
                EvaluationPoint(ITMPose *para, const ITMColorTracker *f_parent);
				~EvaluationPoint(void)
				{
					delete mPara;
					if (cacheNabla != NULL) delete[] cacheNabla;
					if (cacheHessian != NULL) delete[] cacheHessian;
				}

			protected:
                /// calls G_oneLevel, computes gradient and hessian.
				void computeGradients(bool requiresHessian);

				ITMPose *mPara;
				const ITMColorTracker *mParent;

				float cacheF;
				float *cacheNabla;
				float *cacheHessian;
			};

            /// Constructs an EvaluationPoint with the passed parameter.
			EvaluationPoint* evaluateAt(ITMPose *para) const
			{
				return new EvaluationPoint(para, this);
			}

			int numParameters(void) const { return (iterationType == TRACKER_ITERATION_ROTATION) ? 3 : 6; }

            /// f is a single float
            virtual void F_oneLevel(float *f, const ITMPose& pose) = 0;
            virtual void G_oneLevel(float *gradient, float *hessian, const ITMPose& pose) const = 0;

            /// Adjusts projection parameters for a certain level of the pyramid.
            Vector4f projParamsForLevelId(int levelId) const {
                Vector4f projParams = view->calib->intrinsics_rgb.projectionParamsSimple.all;
                projParams.x /= 1 << levelId; projParams.y /= 1 << levelId;
                projParams.z /= 1 << levelId; projParams.w /= 1 << levelId;
                return projParams;
            }

            /// delta === float tx, float ty, float tz, float rx, float ry, float rz
            /// para_new = delta . para_old
            /// Where some parameters of delta are ignored based on iterationType
			void ApplyDelta(const ITMPose & para_old, const float *delta, ITMPose & para_new) const;

			void TrackCamera(ITMTrackingState *trackingState, const ITMView *view);

			ITMColorTracker(Vector2i imgSize, TrackerIterationType *trackingRegime, int noHierarchyLevels,
				const ITMLowLevelEngine *lowLevelEngine, MemoryDeviceType memoryType);
			virtual ~ITMColorTracker(void);
		};
	}
}
