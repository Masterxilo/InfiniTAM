// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include <cstdlib>

#include "Engine/UIEngine.h"
#include "Engine/ImageSourceEngine.h"

#include "Engine/OpenNIEngine.h"
#include "Engine/Kinect2Engine.h"
#include "Engine/LibUVCEngine.h"

using namespace InfiniTAM::Engine;

bool usingImages = false;
const char *calibFile;
/** Create a default source of depth images from a list of command line
    arguments. Typically, @para arg1 would identify the calibration file to
    use, @para arg2 the colour images, @para arg3 the depth images and
    @para arg4 the IMU images. If images are omitted, some live sources will
    be tried.
*/
static void CreateDefaultImageSource(ImageSourceEngine* & imageSource, IMUSourceEngine* & imuSource, const char *arg1, const char *arg2, const char *arg3, const char *arg4)
{
    calibFile = arg1;
	const char *filename1 = arg2;
	const char *filename2 = arg3;
	const char *filename_imu = arg4;

    if (!calibFile || strlen(calibFile) == 0) {
        printf("no calib file given");
        exit(1);
    }
	printf("using calibration file: %s\n", calibFile);

	if (filename2 != NULL)
	{
		printf("using rgb images: %s\nusing depth images: %s\n", filename1, filename2);
        usingImages = true;
		if (filename_imu == NULL)
		{
			imageSource = new ImageFileReader(calibFile, filename1, filename2);
		}
		else
		{
			printf("using imu data: %s\n", filename_imu);
			imageSource = new RawFileReader(calibFile, filename1, filename2, Vector2i(320, 240), 0.5f);
			imuSource = new IMUSourceEngine(filename_imu);
		}
	}

	if (imageSource == NULL)
	{
		printf("trying OpenNI device: %s\n", (filename1==NULL)?"<OpenNI default device>":filename1);
		imageSource = new OpenNIEngine(calibFile, filename1);
		if (imageSource->getDepthImageSize().x == 0)
		{
			delete imageSource;
			imageSource = NULL;
		}
	}
	if (imageSource == NULL)
	{
		printf("trying UVC device\n");
		imageSource = new LibUVCEngine(calibFile);
		if (imageSource->getDepthImageSize().x == 0)
		{
			delete imageSource;
			imageSource = NULL;
		}
	}
	if (imageSource == NULL)
	{
		printf("trying MS Kinect 2 device\n");
		imageSource = new Kinect2Engine(calibFile);
		if (imageSource->getDepthImageSize().x == 0)
		{
			delete imageSource;
			imageSource = NULL;
		}
	}
}

void pause() {
    //system("pause");
}
void record();

// Files\Calibrations\Kinect2\calib.txt
// Files\Scenes\Teddy\calib.txt Files/Scenes/Teddy/Frames/%2504i.ppm Files/Scenes/Teddy/Frames/%2504i.pgm
int main(int argc, char** argv) {
    atexit(pause);
	const char *arg1 = "";
	const char *arg2 = NULL;
	const char *arg3 = NULL;
	const char *arg4 = NULL;

	int arg = 1;
	do {
		if (argv[arg] != NULL) arg1 = argv[arg]; else break;
		++arg;
		if (argv[arg] != NULL) arg2 = argv[arg]; else break;
		++arg;
		if (argv[arg] != NULL) arg3 = argv[arg]; else break;
		++arg;
		if (argv[arg] != NULL) arg4 = argv[arg]; else break;
	} while (false);

	if (arg == 1) {
		printf("usage: %s <calibfile> [<imagesource> [<imu source>]]\n"
            "  <calibfile>   : path to a file containing intrinsic calibration parameters. Must use backslashes (for internal copy system calls)\n"
		       "  <imagesource> : either one argument to specify OpenNI device ID\n"
		       "                  or two arguments specifying rgb and depth file masks\n"
		       "\n"
		       "examples:\n"
		       "  %s ./Files/Teddy/calib.txt ./Files/Teddy/Frames/%%04i.ppm ./Files/Teddy/Frames/%%04i.pgm\n"
		       "  %s ./Files/Teddy/calib.txt\n\n", argv[0], argv[0], argv[0]);
	}

	printf("initialising ...\n");
	ImageSourceEngine *imageSource = NULL;
	IMUSourceEngine *imuSource = NULL;

	CreateDefaultImageSource(imageSource, imuSource, arg1, arg2, arg3, arg4);
	if (imageSource==NULL)
	{
		std::cout << "failed to open any image stream" << std::endl;
		return -1;
	}

	ITMLibSettings *internalSettings = new ITMLibSettings();
	ITMMainEngine *mainEngine = 
        new ITMMainEngine(internalSettings, &imageSource->calib, imageSource->getRGBImageSize(), imageSource->getDepthImageSize());

    
	UIEngine::Instance()->Initialise(argc, argv, imageSource, imuSource, mainEngine, 
        // out dir must use backslashes for internal system("rmdir" calls
        "Files\\Scenes\\Out", 
        internalSettings->deviceType);
	
    //if (usingImages) 
    {
        // start reconstruction
        UIEngine::Instance()->mainLoopAction = UIEngine::PROCESS_VIDEO;
    }
   
    // Start recording anything that happens
    //record();

    UIEngine::Instance()->Run();
	UIEngine::Instance()->Shutdown();

	delete mainEngine;
	delete internalSettings;
	delete imageSource;
	if (imuSource != NULL) delete imuSource;
	return 0;
}

