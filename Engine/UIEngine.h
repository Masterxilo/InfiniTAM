// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../ITMLib/Engine/ITMMainEngine.h"
#include "../ITMLib/Utils/ITMLibSettings.h"
#include "FileUtils.h"

#include "ImageSourceEngine.h"

#include <vector>

class UIEngine
{
public:
private:

    void setFreeviewFromLive() {
        freeviewPose.SetM(
            mainEngine->GetView()->depthImage->eyeCoordinates->fromGlobal
            );
        freeviewIntrinsics = mainEngine->GetView()->calib->intrinsics_d;
        freeviewDim = mainEngine->GetView()->depthImage->imgSize();
    }
	static UIEngine* instance;

	ImageFileReader *imageSource;
	ITMMainEngine *mainEngine;

private: 

	ITMUChar4Image *inputRGBImage;
    ITMShortImage *inputRawDepthImage;

    ITMUChar4Image *outputImage;

    Vector2i freeviewDim;
	bool freeviewActive;
	ITMPose freeviewPose;
	ITMIntrinsics freeviewIntrinsics;
    unsigned int textureId;

    int mouseLastClickButton, mouseLastClickState;
	Vector2i mouseLastClickPos;


public:
	static UIEngine* Instance(void) {
		if (instance == NULL) instance = new UIEngine();
		return instance;
	}

	static void glutDisplayFunction();
	static void glutKeyUpFunction(unsigned char key, int x, int y);
	static void glutMouseButtonFunction(int button, int state, int x, int y);
	static void glutMouseMoveFunction(int x, int y);
	static void glutMouseWheelFunction(int button, int dir, int x, int y);

    void Initialise(int & argc, char** argv, ImageFileReader *imageSource, ITMMainEngine *mainEngine);
	void Shutdown();

	void Run();
	void ProcessFrame();
};
