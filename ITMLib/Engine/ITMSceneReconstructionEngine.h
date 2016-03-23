// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include <math.h>

#include "ITMLibDefines.h"

#include "ITMScene.h"
#include "ITMView.h"
#include "ITMTrackingState.h"
#include "ITMRenderState.h"

using namespace ITMLib::Objects;

namespace ITMLib
{
	namespace Engine
	{
		/** \brief
		    Interface to engines implementing the main KinectFusion
		    depth integration process.

		    These classes basically manage
		    an ITMLib::Objects::ITMScene and fuse new image information
		    into them.
		*/
		class ITMSceneReconstructionEngine
		{
		public:
			/** Clear and reset a scene to set up a new empty
			    one.
			*/
            void ResetScene(
                ITMScene * const scene //<! scene to be reset. 
                );

            /// Fusion stage of the system
            void ProcessFrame(
                const ITMView * const view,
                const ITMTrackingState * const trackingState,
                ITMScene * const scene
                )
            {
                // allocation & visible list update
                AllocateSceneFromDepth(scene, view, trackingState);

                // camera data integration
                IntegrateIntoScene(scene, view, trackingState);
            }

            ITMSceneReconstructionEngine(void);
            virtual ~ITMSceneReconstructionEngine(void);

        private:

            /** Given a view with a new depth image, compute the
            visible blocks, allocate them and update the hash
            table so that the new image data can be integrated.
            */
            void AllocateSceneFromDepth(
                ITMScene *scene,
                const ITMView *view,
                const ITMTrackingState *trackingState
                );

            /** Update the voxel blocks by integrating depth and
            possibly colour information from the given view.
            */
            void IntegrateIntoScene(
                ITMScene *scene,
                const ITMView *view,
                const ITMTrackingState *trackingState);

            void *allocationTempData_device;
            void *allocationTempData_host;
            unsigned char *entriesAllocType_device;
            Vector4s *blockCoords_device;
		};
	}
}
