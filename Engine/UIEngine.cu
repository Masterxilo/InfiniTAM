#include "UIEngine.h"

#include <sstream>
#include <string>
#include <string.h>
#include <GL/glut.h>
#include <GL/freeglut.h>

#include "FileUtils.h"

UIEngine* UIEngine::instance;

static void safe_glutBitmapString(void *font, const char *str)
{
    while (*str) {
		glutBitmapCharacter(font, *str++);
	}
}

const char* KEY_HELP_STR =
"n - next frame \t "
"f - follow camera/free viewpoint \t ";
void UIEngine::glutKeyUpFunction(unsigned char key, int x, int y)
{
    UIEngine *uiEngine = UIEngine::Instance();

    switch (key)
    {
    case 'n':
        //printf("processing one frame ...\n");
        uiEngine->ProcessFrame();
        glutPostRedisplay();
        break;
    case 'f':
        uiEngine->freeviewActive = !uiEngine->freeviewActive;
        glutPostRedisplay();
        break;
    }
}

char shader[100] = "renderColour" //renderGrey" // renderColour // renderColourFromNormal
;

#include "fileutils.h"
#include "visualizeCoordinateSystem.h"
void UIEngine::glutDisplayFunction()
{
	UIEngine *uiEngine = UIEngine::Instance();

    if (!uiEngine->freeviewActive) {
        if (!uiEngine->mainEngine->GetView() || uiEngine->mainEngine->GetView()->colorImage->imgSize().area() <= 1)
            return;

        // Restore viewpoint to live when freeview is not active
        uiEngine->setFreeviewFromLive();
    }

    auto outputImage = new ITMUChar4Image(uiEngine->freeviewDim);
    auto outputDepthImage = new ITMFloatImage(uiEngine->freeviewDim);

    uiEngine->mainEngine->GetImage(
        outputImage, outputDepthImage, 
        &uiEngine->freeviewPose,
        &uiEngine->freeviewIntrinsics,
        shader
        );

    BeginGLRender(
        outputImage,
        outputDepthImage,
        uiEngine->freeviewIntrinsics,
        &uiEngine->freeviewPose
        );

    /*
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

            uiEngine->outputImage = new ITMUChar4Image(uiEngine->freeviewDim);
            uiEngine->mainEngine->GetImage(
                uiEngine->outputImage,
                &uiEngine->freeviewPose,
                &uiEngine->freeviewIntrinsics,
                "renderColour" //renderGrey" // renderColour // renderColourFromNormal
                );
            //png::SaveImageToFile(uiEngine->outputImage, "out.png");

            glBindTexture(GL_TEXTURE_2D, uiEngine->textureId);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 
                uiEngine->outputImage->noDims.x,
                uiEngine->outputImage->noDims.y, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                uiEngine->outputImage->GetData(MEMORYDEVICE_CPU));

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glBegin(GL_QUADS); {
                glTexCoord2f(0, 1); glVertex2f(0, 0);
                glTexCoord2f(1, 1); glVertex2f(1, 0);
                glTexCoord2f(1, 0); glVertex2f(1, 1);
                glTexCoord2f(0, 0); glVertex2f(0, 1);
			}
			glEnd();

			glDisable(GL_TEXTURE_2D);
		}
		glPopMatrix();
	}
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

    */


    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

	glColor3f(1.0f, 0.0f, 0.0f); 

	glRasterPos2f(-0.95f, -0.95f);
	char str[2000];
	sprintf(str, 
		KEY_HELP_STR);
	safe_glutBitmapString(GLUT_BITMAP_HELVETICA_12, (const char*)str);

    glFinish();
	glutSwapBuffers();

    glutKeyUpFunction('n', 0, 0);
}

void UIEngine::glutMouseButtonFunction(int button, int state, int x, int y)
{
	UIEngine *uiEngine = UIEngine::Instance();
    uiEngine->mouseLastClickButton = button;
    uiEngine->mouseLastClickState = state;
    uiEngine->mouseLastClickPos = Vector2i(x, y);
}

/// Alternative/external implementation of axis-angle rotation matrix construction
/// axis does not need to be normalized.
Matrix3f createRotation(const Vector3f & _axis, float angle)
{
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
    skewmat.at(2, 1) = axis.x; // should be -axis.x;
    skewmat.at(1, 2) = -axis.x;// should be axis.x;
    skewmat *= si;
    ret += skewmat;

    return ret;
}
void UIEngine::glutMouseMoveFunction(int x, int y)
{
	UIEngine *uiEngine = UIEngine::Instance();
	if (!uiEngine->freeviewActive) return;

	Vector2i movement;
	movement.x = x - uiEngine->mouseLastClickPos.x;
	movement.y = y - uiEngine->mouseLastClickPos.y;
	uiEngine->mouseLastClickPos.x = x;
	uiEngine->mouseLastClickPos.y = y;

	if ((movement.x == 0) && (movement.y == 0)) return;
    if ( uiEngine->mouseLastClickState != GLUT_DOWN) return;

    static const float scale_rotation = 0.005f;
    static const float scale_translation = 0.0025f;
    switch (uiEngine->mouseLastClickButton)
	{
    case GLUT_LEFT_BUTTON:
	{
		// rotation
		Vector3f axis((float)-movement.y, (float)-movement.x, 0.0f);
		float angle = scale_rotation * sqrt((float)(movement.x * movement.x + movement.y*movement.y));
        Matrix3f rot = createRotation(axis, angle);
		uiEngine->freeviewPose.SetRT(rot * uiEngine->freeviewPose.GetR(), rot * uiEngine->freeviewPose.GetT());
		uiEngine->freeviewPose.Coerce();

        glutPostRedisplay();
		break;
	}
    case GLUT_RIGHT_BUTTON:
	{
		// right button: translation in x and y direction
		uiEngine->freeviewPose.SetT(uiEngine->freeviewPose.GetT() + scale_translation * Vector3f((float)movement.x, (float)movement.y, 0.0f));
        glutPostRedisplay();
		break;
	}
    case GLUT_MIDDLE_BUTTON:
	{
		// middle button: translation along z axis
		uiEngine->freeviewPose.SetT(uiEngine->freeviewPose.GetT() + scale_translation * Vector3f(0.0f, 0.0f, (float)movement.y));
        glutPostRedisplay();
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
    glutPostRedisplay();
}

void UIEngine::Initialise(int & argc, char** argv, ImageFileReader *imageSource, ITMMainEngine *mainEngine ) 
{
    this->freeviewActive = 1;
    this->freeviewPose.SetT(Vector3f(0, 0,  100 * voxelSize)); // move back a bit to see something
    //viewFrustum_max
	this->imageSource = imageSource;
	this->mainEngine = mainEngine;
    this->freeviewDim = Vector2i(640, 480);

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(freeviewDim.x, freeviewDim.y);
	glutCreateWindow("InfiniTAM");

	glutDisplayFunc(UIEngine::glutDisplayFunction);
	glutKeyboardUpFunc(UIEngine::glutKeyUpFunction);
	glutMouseFunc(UIEngine::glutMouseButtonFunction);
	glutMotionFunc(UIEngine::glutMouseMoveFunction);

	glutMouseWheelFunc(UIEngine::glutMouseWheelFunction);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, 1);

    glGenTextures(1, &textureId);

    inputRGBImage = new ITMUChar4Image();
	inputRawDepthImage = new ITMShortImage();

    mouseLastClickPos = Vector2i(0, 0);
    mouseLastClickState = GLUT_UP;

    glutKeyUpFunction('n', 0, 0);
    //glutPostRedisplay();
}

void UIEngine::ProcessFrame() {
	imageSource->nextImages(inputRGBImage, inputRawDepthImage);
    mainEngine->ProcessFrame(inputRGBImage, inputRawDepthImage);
}

void UIEngine::Run() { glutMainLoop(); }
void UIEngine::Shutdown()
{
	delete inputRGBImage;
	delete inputRawDepthImage;
	delete instance;
}
