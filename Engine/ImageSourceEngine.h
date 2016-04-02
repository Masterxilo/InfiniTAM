#pragma once
#include "ITMLib.h"
#include "FileUtils.h"
#include <stdio.h>

/** Represents a source of depth & color image data, together with the calibration of those two cameras. */ 
class ImageFileReader
{
private:
    std::string rgbImageMask, depthImageMask;
    

public:int currentFrameNo;
    ITMRGBDCalib calib;
    // Returns the current frame and advances the frame counter for next time.
    /// images must exist
    void nextImages(ITMUChar4Image *rgb, ITMShortImage *rawDepth) {
        char str[2048];
        sprintf(str, rgbImageMask.c_str(), currentFrameNo);
        if (!png::ReadImageFromFile(rgb, str)) {
            if (1 == currentFrameNo) assert(false); // should have at least one image
            return;
        }

        sprintf(str, depthImageMask.c_str(), currentFrameNo);
        if (!png::ReadImageFromFile(rawDepth, str)) return;

        ++currentFrameNo;
    }

    ImageFileReader(
        std::string calibFilename,
        std::string rgbImageMask,
        std::string depthImageMask,
        int firstFrameNo
        ) : currentFrameNo(firstFrameNo), rgbImageMask(rgbImageMask), depthImageMask(depthImageMask) {
        readRGBDCalib(calibFilename, calib);
    }
};

