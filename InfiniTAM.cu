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
static ImageSourceEngine * CreateDefaultImageSource(
    const char *calibFile, const char *rgbMask, const char *depthMask)
{
	printf("using calibration file: %s\n", calibFile);
    printf("using rgb images: %s\nusing depth images: %s\n", rgbMask, depthMask);   
    return new ImageFileReader(calibFile, rgbMask, depthMask);
}

void pause() {
    //system("pause");
}
void record();

void tests();
void redirectStd();
#include <string>
#include <sstream>
#include <iostream>
#include <iostream>
using namespace std;
// Files\Scenes\Teddy\calib.txt Files/Scenes/Teddy/Frames/color%25i.png Files/Scenes/Teddy/Frames/depth%25i.png ConvertDisparityToDepth
// Files\Scenes\fountain\calib.txt Files/Scenes/fountain/Frames/color%25i.png Files/Scenes/fountain/Frames/depth%25i.png ScaleAndValidateDepth
int main(int argc, char** argv) {

    //redirectStd();
    tests(); // TODO enable again
    return 0;
    atexit(pause);

	if (argc != 5) {
		printf("usage: %s <calibfile> [<imagesource>]\n"
            "  <calibfile>   : path to a file containing intrinsic calibration parameters. Must use backslashes (for internal copy system calls)\n"
            "  <imagesource> : two arguments specifying rgb and depth file masks (sprintf)\n"
		       "\n"
		       "examples:\n"
		       "  see files\\scenes\\*\\parameters\n"
		       , argv[0], argv[0]);
	}

	printf("initialising ...\n");
    auto imageSource = CreateDefaultImageSource(argv[1], argv[2], argv[3]);
    ITMView::depthConversionType = argv[4];

	ITMMainEngine *mainEngine = 
        new ITMMainEngine(
        &imageSource->calib,
        imageSource->getRGBImageSize(),
        imageSource->getDepthImageSize()
        );

    
	UIEngine::Instance()->Initialise(argc, argv, imageSource, mainEngine, 
        // out dir must use backslashes for internal system("rmdir" calls
        "Files\\Scenes\\Out");
	
    //if (usingImages) 
    {
        // start reconstruction
        UIEngine::Instance()->mainLoopAction = UIEngine::PROCESS_FRAME;// PROCESS_VIDEO;
    }
   
    // Start recording anything that happens
    //record();

    UIEngine::Instance()->Run();
    // When the main window is closed/esc pressed we arrive here
	UIEngine::Instance()->Shutdown();

	delete mainEngine;
	delete imageSource;
	return 0;
}

