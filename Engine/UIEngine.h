// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../ITMLib/Engine/ITMMainEngine.h"
#include "../ITMLib/Utils/ITMLibSettings.h"
#include "FileUtils.h"
#include "NVTimer.h"

#include "ImageSourceEngine.h"

#include <vector>

namespace InfiniTAM
{
	namespace Engine
	{
		class UIEngine
		{
        public:

            enum MainLoopAction
            {
                PROCESS_PAUSED, PROCESS_FRAME, PROCESS_VIDEO, EXIT
            }mainLoopAction;
        private:

            void setFreeviewFromLive() {
                freeviewPose.SetFrom(mainEngine->GetTrackingState()->pose_d);
                assert(mainEngine->GetView() != NULL);
                freeviewIntrinsics = mainEngine->GetView()->calib->intrinsics_d;
                windows[0].outImage->ChangeDims(mainEngine->GetView()->depth->noDims);
            }
			static UIEngine* instance;


			struct UIColourMode {
				const char *name;
				ITMMainEngine::GetImageType type;
				UIColourMode(const char *_name, ITMMainEngine::GetImageType _type)
				 : name(_name), type(_type)
				{}
			};
			std::vector<UIColourMode> colourModes;
			int currentColourMode;

			ITMLibSettings internalSettings;
			ImageSourceEngine *imageSource;
			ITMMainEngine *mainEngine;

			StopWatchInterface *timer_instant;
			StopWatchInterface *timer_average;

		private: 
            // For UI layout
            Vector2i winSize;
			static const int NUM_WIN = 3;
            struct Window {
                Vector4f winReg; // (x1, y1, x2, y2)
                uint textureId;
                ITMUChar4Image *outImage;
                ITMMainEngine::GetImageType outImageType;
            };
            Window windows[NUM_WIN];

			ITMUChar4Image *inputRGBImage; ITMShortImage *inputRawDepthImage;

			bool freeviewActive;
			bool intergrationActive;
			ITMPose freeviewPose;
			ITMIntrinsics freeviewIntrinsics;

            enum MOUSESTATE { MLEFT, MMIDDLE, MRIGHT, MNONE };
            MOUSESTATE mouseState;
			Vector2i mouseLastClick;


		public:
			int currentFrameNo; 
			static UIEngine* Instance(void) {
				if (instance == NULL) instance = new UIEngine();
				return instance;
			}



			static void glutDisplayFunction();
			static void glutIdleFunction();
			static void glutKeyUpFunction(unsigned char key, int x, int y);
			static void glutMouseButtonFunction(int button, int state, int x, int y);
			static void glutMouseMoveFunction(int x, int y);
			static void glutMouseWheelFunction(int button, int dir, int x, int y);

			const Vector2i & getWindowSize(void) const
			{ return winSize; }

			int processedFrameNo;
			bool needsRefresh;

			void Initialise(int & argc, char** argv, ImageSourceEngine *imageSource, ITMMainEngine *mainEngine,
				const char *outFolder);
			void Shutdown();

			void Run();
			void ProcessFrame();
		};
	}
}
