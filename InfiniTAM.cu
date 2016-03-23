// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include <cstdlib>
#include <iostream>

#include "Engine/UIEngine.h"
#include "Engine/ImageSourceEngine.h"


using namespace InfiniTAM::Engine;

bool usingImages = false;
const char *calibFile;
/** Create a default source of depth images from a list of command line
    arguments. Typically, @para arg1 would identify the calibration file to
    use, @para arg2 the colour images, @para arg3 the depth images and
    @para arg4 the IMU images. If images are omitted, some live sources will
    be tried.
*/
static void CreateDefaultImageSource(ImageSourceEngine* & imageSource, const char *arg1, const char *arg2, const char *arg3)
{
    calibFile = arg1;
	const char *filename1 = arg2;
	const char *filename2 = arg3;

    if (!calibFile || strlen(calibFile) == 0) {
        printf("no calib file given");
        exit(1);
    }
	printf("using calibration file: %s\n", calibFile);

	if (filename2 != NULL)
	{
		printf("using rgb images: %s\nusing depth images: %s\n", filename1, filename2);
        usingImages = true;
        imageSource = new ImageFileReader(calibFile, filename1, filename2);
	}
}

void pause() {
    //system("pause");
}
void record();

void tests();
void redirectStd();
// Files\Calibrations\Kinect2\calib.txt
// Files\Scenes\Teddy\calib.txt Files/Scenes/Teddy/Frames/%2504i.ppm Files/Scenes/Teddy/Frames/%2504i.pgm
int main(int argc, char** argv) {

    redirectStd();
    tests();
    atexit(pause);
	const char *arg1 = "";
	const char *arg2 = NULL;
	const char *arg3 = NULL;

	int arg = 1;
	do {
		if (argv[arg] != NULL) arg1 = argv[arg]; else break;
		++arg;
		if (argv[arg] != NULL) arg2 = argv[arg]; else break;
		++arg;
		if (argv[arg] != NULL) arg3 = argv[arg]; else break;
	} while (false);

	if (arg == 1) {
		printf("usage: %s <calibfile> [<imagesource>]\n"
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

	CreateDefaultImageSource(imageSource, arg1, arg2, arg3);
	if (imageSource==NULL)
	{
		std::cout << "failed to open any image stream" << std::endl;
		return -1;
	}

	ITMLibSettings *internalSettings = new ITMLibSettings();
	ITMMainEngine *mainEngine = 
        new ITMMainEngine(internalSettings, &imageSource->calib, imageSource->getRGBImageSize(), imageSource->getDepthImageSize());

    
	UIEngine::Instance()->Initialise(argc, argv, imageSource, mainEngine, 
        // out dir must use backslashes for internal system("rmdir" calls
        "Files\\Scenes\\Out");
	
    //if (usingImages) 
    {
        // start reconstruction
        UIEngine::Instance()->mainLoopAction = UIEngine::PROCESS_VIDEO;
    }
   
    // Start recording anything that happens
    //record();

    UIEngine::Instance()->Run();
    // When the main window is closed/esc pressed we arrive here
	UIEngine::Instance()->Shutdown();

	delete mainEngine;
	delete internalSettings;
	delete imageSource;
	return 0;
}

