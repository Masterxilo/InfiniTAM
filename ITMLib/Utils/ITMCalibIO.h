// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Objects/ITMRGBDCalib.h"


namespace ITMLib
{
	namespace Objects
	{
		bool readRGBDCalib(const char *fileName, ITMRGBDCalib & dest);
    }
}

