// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "UIEngine.h"

#include <sstream>
#include <string>
#include <string.h>
#include <GL/glut.h>
#include <GL/freeglut.h>

#include "../Utils/FileUtils.h"

using namespace InfiniTAM::Engine;
UIEngine* UIEngine::instance;

static void safe_glutBitmapString(void *font, const char *str)
{
    while (*str) {
		glutBitmapCharacter(font, *str++);
	}
}

// Called from void ImageFileReader::loadIntoCache(void) when all files have been read
void endOfFiles() {
    UIEngine::glutKeyUpFunction('f',0,0);
}

const char* KEY_HELP_STR =
"n - next frame \t "
"b - all frames \t "
"e/esc - exit \t "
"f - follow camera/free viewpoint \t "
"c - colours (currently %s) \t "
"t - turn fusion %s\t "
"s - record \t "
"w - write mesh to disk \t "
"o - save current images to disk\t "
"r - reset data \t ";
extern const char *calibFile;

void UIEngine::glutKeyUpFunction(unsigned char key, int x, int y)
{
    UIEngine *uiEngine = UIEngine::Instance();

    switch (key)
    {
    case 'n':
        printf("processing one frame ...\n");
        uiEngine->mainLoopAction = UIEngine::PROCESS_FRAME;
        break;
    case 'b':
        printf("processing input source ...\n");
        uiEngine->mainLoopAction = UIEngine::PROCESS_VIDEO;
        break;
    case 'r':
        printf("clearing data ...\n");
        uiEngine->mainEngine->ResetScene();
        break;
    case 'e':
    case 27: // esc key
        printf("exiting ...\n");
        uiEngine->mainLoopAction = UIEngine::EXIT;
        break;
    case 'f':
        if (uiEngine->freeviewActive)
        {
            uiEngine->windows[0].outImageType = ITMMainEngine::InfiniTAM_IMAGE_SCENERAYCAST;
            uiEngine->freeviewActive = false;
        }
        else
        {
            uiEngine->windows[0].outImageType = ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_SHADED;

            uiEngine->freeviewPose.SetFrom(uiEngine->mainEngine->GetTrackingState()->pose_d);
            if (uiEngine->mainEngine->GetView() != NULL) {
                uiEngine->freeviewIntrinsics = uiEngine->mainEngine->GetView()->calib->intrinsics_d;
                uiEngine->windows[0].outImage->ChangeDims(uiEngine->mainEngine->GetView()->depth->noDims);
            }
            uiEngine->freeviewActive = true;
        }
        uiEngine->needsRefresh = true;
        break;
    case 'c':
        uiEngine->currentColourMode++; 
        if ((unsigned)uiEngine->currentColourMode >= uiEngine->colourModes.size()) 
            uiEngine->currentColourMode = 0;
        uiEngine->needsRefresh = true;
        break;
    default:
        break;
    }

    if (uiEngine->freeviewActive) {
        uiEngine->windows[0].outImageType = uiEngine->colourModes[uiEngine->currentColourMode].type;
    }
}

void UIEngine::glutDisplayFunction()
{
	UIEngine *uiEngine = UIEngine::Instance();
    uiEngine->needsRefresh = false;

	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0f, 1.0f, 1.0f);
	glEnable(GL_TEXTURE_2D);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	{
		glLoadIdentity();
		glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		{
			glEnable(GL_TEXTURE_2D);

            for (auto& window : uiEngine->windows) {
                if (window.outImageType == ITMMainEngine::InfiniTAM_IMAGE_UNKNOWN) continue;

                // Draw each sub window
                // get updated images from processing thread
                uiEngine->mainEngine->GetImage(
                    window.outImage,
                    window.outImageType,
                    &uiEngine->freeviewPose, &uiEngine->freeviewIntrinsics);

                Vector4f winReg = window.winReg;

                glBindTexture(GL_TEXTURE_2D, window.textureId);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 
                    window.outImage->noDims.x,
                    window.outImage->noDims.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, 
                    window.outImage->GetData(MEMORYDEVICE_CPU));

				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glBegin(GL_QUADS); {
                    glTexCoord2f(0, 1); glVertex2f(winReg.v[0], winReg.v[1]); // glVertex2f(0, 0);
                    glTexCoord2f(1, 1); glVertex2f(winReg.v[2], winReg.v[1]); // glVertex2f(1, 0);
                    glTexCoord2f(1, 0); glVertex2f(winReg.v[2], winReg.v[3]); // glVertex2f(1, 1);
                    glTexCoord2f(0, 0); glVertex2f(winReg.v[0], winReg.v[3]); // glVertex2f(0, 1);
				}
				glEnd();
			}
			glDisable(GL_TEXTURE_2D);
		}
		glPopMatrix();
	}
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glColor3f(1.0f, 0.0f, 0.0f); 
    
    glRasterPos2f(0.85f, -0.962f);
	char str[2000]; sprintf(str, "%04.2lf", 
        //sdkGetTimerValue(&timer_instant);
        sdkGetAverageTimerValue(&uiEngine->timer_average)
        );
	safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const char*)str);

	glRasterPos2f(-0.95f, -0.95f);
	sprintf(str, 
		KEY_HELP_STR
		, uiEngine->colourModes[uiEngine->currentColourMode].name, 
		uiEngine->intergrationActive ? "off" : "on");
	safe_glutBitmapString(GLUT_BITMAP_HELVETICA_12, (const char*)str);

	glutSwapBuffers();
}

void UIEngine::glutIdleFunction()
{
	UIEngine *uiEngine = UIEngine::Instance();

	switch (uiEngine->mainLoopAction)
	{
	case PROCESS_FRAME:
		uiEngine->ProcessFrame(); uiEngine->processedFrameNo++;
		uiEngine->mainLoopAction = PROCESS_PAUSED;
		uiEngine->needsRefresh = true;
		break;
	case PROCESS_VIDEO:
		uiEngine->ProcessFrame(); uiEngine->processedFrameNo++;
		uiEngine->needsRefresh = true;
		break;
	case EXIT:
		glutLeaveMainLoop();
		break;
	case PROCESS_PAUSED:
	default:
		break;
	}

	if (uiEngine->needsRefresh) {
		glutPostRedisplay();
	}
}

void UIEngine::glutMouseButtonFunction(int button, int state, int x, int y)
{
	UIEngine *uiEngine = UIEngine::Instance();

	if (state == GLUT_DOWN)
	{
		switch (button)
		{
        case GLUT_LEFT_BUTTON: uiEngine->mouseState = MLEFT; break;
        case GLUT_MIDDLE_BUTTON: uiEngine->mouseState = MMIDDLE; break;
        case GLUT_RIGHT_BUTTON: uiEngine->mouseState = MRIGHT; break;
		default: break;
		}
		uiEngine->mouseLastClick.x = x;
		uiEngine->mouseLastClick.y = y;
	}
    else if (state == GLUT_UP) uiEngine->mouseState = MNONE;
}

static inline Matrix3f createRotation(const Vector3f & _axis, float angle)
{
    // TODO leverage ITMPose which does this conversion given r = axis * angle
	Vector3f axis = normalize(_axis);
	float si = sinf(angle);
	float co = cosf(angle);

	Matrix3f ret;
	ret.setIdentity();

	ret *= co;
	for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) ret.at(c, r) += (1.0f - co) * axis[c] * axis[r];

	Matrix3f skewmat;
	skewmat.setZeros();
	skewmat.at(1, 0) = -axis.z;
	skewmat.at(0, 1) = axis.z;
	skewmat.at(2, 0) = axis.y;
	skewmat.at(0, 2) = -axis.y;
	skewmat.at(2, 1) = axis.x;
	skewmat.at(1, 2) = -axis.x;
	skewmat *= si;
	ret += skewmat;

	return ret;
}

void UIEngine::glutMouseMoveFunction(int x, int y)
{
	UIEngine *uiEngine = UIEngine::Instance();

	if (!uiEngine->freeviewActive) return;

	Vector2i movement;
	movement.x = x - uiEngine->mouseLastClick.x;
	movement.y = y - uiEngine->mouseLastClick.y;
	uiEngine->mouseLastClick.x = x;
	uiEngine->mouseLastClick.y = y;

	if ((movement.x == 0) && (movement.y == 0)) return;

	static const float scale_rotation = 0.005f;
	static const float scale_translation = 0.0025f;

	switch (uiEngine->mouseState)
	{
	case MLEFT:
	{
		// rotation
		Vector3f axis((float)-movement.y, (float)-movement.x, 0.0f);
		float angle = scale_rotation * sqrt((float)(movement.x * movement.x + movement.y*movement.y));
		Matrix3f rot = createRotation(axis, angle);
		uiEngine->freeviewPose.SetRT(rot * uiEngine->freeviewPose.GetR(), rot * uiEngine->freeviewPose.GetT());
		uiEngine->freeviewPose.Coerce();

		uiEngine->needsRefresh = true;
		break;
	}
	case MRIGHT:
	{
		// right button: translation in x and y direction
		uiEngine->freeviewPose.SetT(uiEngine->freeviewPose.GetT() + scale_translation * Vector3f((float)movement.x, (float)movement.y, 0.0f));
		uiEngine->needsRefresh = true;
		break;
	}
	case MMIDDLE:
	{
		// middle button: translation along z axis
		uiEngine->freeviewPose.SetT(uiEngine->freeviewPose.GetT() + scale_translation * Vector3f(0.0f, 0.0f, (float)movement.y));
		uiEngine->needsRefresh = true;
		break;
	}
	default: break;
	}
}

void UIEngine::glutMouseWheelFunction(int button, int dir, int x, int y)
{
	UIEngine *uiEngine = UIEngine::Instance();

	static const float scale_translation = 0.05f;

	uiEngine->freeviewPose.SetT(uiEngine->freeviewPose.GetT() + scale_translation * Vector3f(0.0f, 0.0f, (dir > 0) ? -1.0f : 1.0f));
	uiEngine->needsRefresh = true;
}

void UIEngine::Initialise(int & argc, char** argv, ImageSourceEngine *imageSource, ITMMainEngine *mainEngine,
	const char *outFolder)
{
	this->freeviewActive = false;
	this->intergrationActive = true;
	this->currentColourMode = 0;
	this->colourModes.push_back(UIColourMode("shaded greyscale", ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_SHADED));
	this->colourModes.push_back(UIColourMode("integrated colours", ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME));
	this->colourModes.push_back(UIColourMode("surface normals", ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL));

	this->imageSource = imageSource;
	this->mainEngine = mainEngine;

	int textAreaHeight = 30; 
	winSize.x = (int)(1.5f * (float)(imageSource->getDepthImageSize().x));
	winSize.y = imageSource->getDepthImageSize().y + textAreaHeight;
	float h1 = textAreaHeight / (float)winSize.y, h2 = (1.f + h1) / 2;
    windows[0].winReg = Vector4f(0.0f, h1, 0.665f, 1.0f);   // Main render
    windows[1].winReg = Vector4f(0.665f, h2, 1.0f, 1.0f);   // Side sub window 0
    windows[2].winReg = Vector4f(0.665f, h1, 1.0f, h2);     // Side sub window 2

    windows[0].outImageType = ITMMainEngine::InfiniTAM_IMAGE_SCENERAYCAST;
    windows[1].outImageType = ITMMainEngine::InfiniTAM_IMAGE_ORIGINAL_DEPTH;
    windows[2].outImageType = ITMMainEngine::InfiniTAM_IMAGE_ORIGINAL_RGB;

	this->currentFrameNo = 0;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(winSize.x, winSize.y);
	glutCreateWindow("InfiniTAM");
    for (auto& window : windows) 
        glGenTextures(1, &window.textureId);

	glutDisplayFunc(UIEngine::glutDisplayFunction);
	glutKeyboardUpFunc(UIEngine::glutKeyUpFunction);
	glutMouseFunc(UIEngine::glutMouseButtonFunction);
	glutMotionFunc(UIEngine::glutMouseMoveFunction);
	glutIdleFunc(UIEngine::glutIdleFunction);

	glutMouseWheelFunc(UIEngine::glutMouseWheelFunction);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, 1);


	bool allocateGPU = true;
    for (auto& window : windows) 
        window.outImage = new ITMUChar4Image(imageSource->getDepthImageSize(), true, allocateGPU);

	inputRGBImage = new ITMUChar4Image(imageSource->getRGBImageSize(), true, allocateGPU);
	inputRawDepthImage = new ITMShortImage(imageSource->getDepthImageSize(), true, allocateGPU);


	mainLoopAction = PROCESS_PAUSED;
	mouseState = MNONE;
	needsRefresh = false;
	processedFrameNo = 0;

	ITMSafeCall(cudaThreadSynchronize());

	sdkCreateTimer(&timer_instant);
	sdkCreateTimer(&timer_average);

	sdkResetTimer(&timer_average);

	printf("initialised.\n");
}

void UIEngine::ProcessFrame()
{
	if (!imageSource->hasMoreImages()) return;
	imageSource->getImages(inputRGBImage, inputRawDepthImage);

	sdkResetTimer(&timer_instant);
	sdkStartTimer(&timer_instant); sdkStartTimer(&timer_average);

	//actual processing on the mainEngine
    mainEngine->ProcessFrame(inputRGBImage, inputRawDepthImage);

	ITMSafeCall(cudaThreadSynchronize());
	sdkStopTimer(&timer_instant); sdkStopTimer(&timer_average);
    

	currentFrameNo++;
}

void UIEngine::Run() { glutMainLoop(); }
void UIEngine::Shutdown()
{
	sdkDeleteTimer(&timer_instant);
	sdkDeleteTimer(&timer_average);

	delete inputRGBImage;
	delete inputRawDepthImage;

	delete instance;
}
