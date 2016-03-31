// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ImageSourceEngine.h"

#include "FileUtils.h"

#include <stdio.h>

using namespace InfiniTAM::Engine;

ImageSourceEngine::ImageSourceEngine(const char *calibFilename)
{
	readRGBDCalib(calibFilename, calib);
}

ImageFileReader::ImageFileReader(const char *calibFilename, const char *rgbImageMask, const char *depthImageMask)
	: ImageSourceEngine(calibFilename)
{
	strncpy(this->rgbImageMask, rgbImageMask, BUF_SIZE);
	strncpy(this->depthImageMask, depthImageMask, BUF_SIZE);

	currentFrameNo = 1;
	cachedFrameNo = -1;

	cached_rgb = NULL;
	cached_depth = NULL;
}

ImageFileReader::~ImageFileReader()
{
	delete cached_rgb;
	delete cached_depth;
}

void endOfFiles();
void ImageFileReader::loadIntoCache(void)
{
	if (currentFrameNo == cachedFrameNo) return;
	cachedFrameNo = currentFrameNo;

	//TODO> make nicer
    assert(cached_rgb == NULL);
    assert(cached_depth == NULL);
	cached_rgb = new ITMUChar4Image(); 
	cached_depth = new ITMShortImage();

	char str[2048];
	sprintf(str, rgbImageMask, currentFrameNo);
	if (!png::ReadImageFromFile(cached_rgb, str)) 
	{
		delete cached_rgb; cached_rgb = NULL;
		printf("error reading file '%s'\n", str);
	}

	sprintf(str, depthImageMask, currentFrameNo);
    if (!png::ReadImageFromFile(cached_depth, str))
	{
		delete cached_depth; cached_depth = NULL;
		printf("error reading file '%s'\n", str);

        printf("end of files\n", str);
        endOfFiles();
	}
}

bool ImageFileReader::hasMoreImages(void)
{
	loadIntoCache();
	return ((cached_rgb!=NULL)&&(cached_depth!=NULL));
}

void ImageFileReader::getImages(ITMUChar4Image *rgb, ITMShortImage *rawDepth)
{
    loadIntoCache();
    if (!hasMoreImages()) return;
    assert(rgb->noDims == cached_rgb->noDims);
    assert(rawDepth->noDims == cached_depth->noDims);

	rgb->SetFrom(cached_rgb, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
	delete cached_rgb;
	cached_rgb = NULL;

	rawDepth->SetFrom(cached_depth, ORUtils::MemoryBlock<short>::CPU_TO_CPU);
	delete cached_depth;
	cached_depth = NULL;

	++currentFrameNo;
}

Vector2i ImageFileReader::getDepthImageSize(void)
{
	loadIntoCache();
	return cached_depth->noDims;
}

Vector2i ImageFileReader::getRGBImageSize(void)
{
	loadIntoCache();
	if (cached_rgb != NULL) return cached_rgb->noDims;
	return cached_depth->noDims;
}
