#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <iostream>
using namespace std;

#include "Engine/UIEngine.h"
#include "Engine/ImageSourceEngine.h"

/** Create a default source of depth images from a list of command line
    arguments. Typically, @para arg1 would identify the calibration file to
    use, @para arg2 the colour images, @para arg3 the depth images and
    @para arg4 the IMU images. If images are omitted, some live sources will
    be tried.
*/
static ImageFileReader * CreateDefaultImageSource(
    const char *calibFile, const char *rgbMask, const char *depthMask)
{
	printf("using calibration file: %s\n", calibFile);
    printf("using rgb images: %s\nusing depth images: %s\n", rgbMask, depthMask);   
    return new ImageFileReader(calibFile, rgbMask, depthMask, 1);
}

int main(int argc, char** argv) {
    assert(LoadLibraryA("PaulLanguage3.dll"));
    //tests();
    
    auto imageSource = CreateDefaultImageSource(argv[1], argv[2], argv[3]);
    ITMView::depthConversionType = argv[4];

	ITMMainEngine *mainEngine = 
        new ITMMainEngine(
        &imageSource->calib
        );
    
	UIEngine::Instance()->Initialise(argc, argv, imageSource, mainEngine);
    UIEngine::Instance()->Run();
	UIEngine::Instance()->Shutdown();

	delete mainEngine;
	delete imageSource;
	return 0;
}

