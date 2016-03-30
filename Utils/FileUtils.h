// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include <stdio.h>

#include "../ITMLib/Utils/ITMLibDefines.h"

/// \returns true iff operation succeeded
namespace png {
    // image must exist but can have any size. Loads rgb only.                  
    bool ReadImageFromFile(ITMUChar4Image* image, const char* fileName);
    // image must exist but can have any size.  
    bool ReadImageFromFile(ITMShortImage *image, const char *fileName);


    bool SaveImageToFile(const ITMUChar4Image* image, const char* fileName);
    bool SaveImageToFile(const ITMShortImage* image, const char* fileName);
}
