// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM
#include "DeviceAgnostic\ITMVisualisationEngine.h"

#include "ITMVisualisationEngine.h"

using namespace ITMLib::Engine;

/// as val goes from x0 to x1, output goes from y0 to y1 linearly
inline float interpolate(float val, float y0, float x0, float y1, float x1) {
	return (val - x0)*(y1 - y0) / (x1 - x0) + y0;
}

/**
1   ---
0__/   \___
where the angles are at
-.75, -.25, .25, .75
*/
inline float base(float val) {
	if (val <= -0.75f) return 0.0f;
	else if (val <= -0.25f) return interpolate(val, 0.0f, -0.75f, 1.0f, -0.25f);
	else if (val <= 0.25f) return 1.0f;
	else if (val <= 0.75f) return interpolate(val, 1.0f, 0.25f, 0.0f, 0.75f);
	else return 0.0;
}

void IITMVisualisationEngine::DepthToUchar4(ITMUChar4Image *dst, const ITMFloatImage *src)
{
    dst->Clear();
    Vector4u * const dest = dst->GetData(MEMORYDEVICE_CPU);
	float const * const source = src->GetData(MEMORYDEVICE_CPU);
	const int dataSize = static_cast<int>(dst->dataSize);

    // lims =  #@source & /@ {Min, Max}
	float lims[2];
	lims[0] = 100000.0f; lims[1] = -100000.0f;

	for (int idx = 0; idx < dataSize; idx++)
	{
		float sourceVal = source[idx];
		if (sourceVal > 0.0f) { lims[0] = MIN(lims[0], sourceVal); lims[1] = MAX(lims[1], sourceVal); }
	}
	if (lims[0] == lims[1]) return;

    // Rescaled rgb-converted depth
    const float scale = 1.0f / (lims[1] - lims[0]);
	for (int idx = 0; idx < dataSize; idx++)
	{
		float sourceVal = source[idx];

        if (sourceVal <= 0.0f) continue;
		sourceVal = (sourceVal - lims[0]) * scale;

        dest[idx].r = (uchar)(base(sourceVal - 0.5f) * 255.0f); // shows the range 0 to 1.25
		dest[idx].g = (uchar)(base(sourceVal) * 255.0f); // shows the range 0 to .75
		dest[idx].b = (uchar)(base(sourceVal + 0.5f) * 255.0f); // shows the range 
		dest[idx].a = 255;
	}
}


template class ITMLib::Engine::ITMVisualisationEngine<ITMVoxel, ITMVoxelIndex>;
