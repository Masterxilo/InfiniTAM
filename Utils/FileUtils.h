// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include <stdio.h>

#include "../ITMLib/Utils/ITMLibDefines.h"

namespace png {
    // Image must exist but can have any size. Loads rgb only.                  
    bool ReadImageFromFile(ITMUChar4Image* image, const char* fileName);
    // Image must exist but can have any size.                          
    bool ReadImageFromFile(ITMShortImage *image, const char *fileName);
}
