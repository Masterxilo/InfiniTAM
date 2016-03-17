// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Utils/ITMLibDefines.h"

namespace ITMLib
{
	namespace Objects
	{
        /// Generic image (T) hierarchy for pyramid scheme/coarse-to-fine solving of various optimization problems.
        /// Level 0 has the highest resolution, then lower
		template <class T> class ITMImageHierarchy
		{
		public:
			int noLevels;
			T **levels;

			ITMImageHierarchy(Vector2i imgSize, TrackerIterationType *trackingRegime, int noHierarchyLevels, 
				MemoryDeviceType memoryType, bool skipAllocationForLevel0 = false)
			{
				this->noLevels = noHierarchyLevels;

				levels = new T*[noHierarchyLevels];

				for (int i = noHierarchyLevels - 1; i >= 0; i--)
					levels[i] = new T(imgSize, i, trackingRegime[i], memoryType, (i == 0) && skipAllocationForLevel0);
			}

			void UpdateHostFromDevice()
			{ for (int i = 0; i < noLevels; i++) this->levels[i]->UpdateHostFromDevice(); }

			void UpdateDeviceFromHost()
			{ for (int i = 0; i < noLevels; i++) this->levels[i]->UpdateDeviceFromHost(); }

			~ITMImageHierarchy(void)
			{
				for (int i = 0; i < noLevels; i++) delete levels[i];
				delete [] levels;
			}
		};
	}
}
