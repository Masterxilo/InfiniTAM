#include "FileUtils.h"
#include <stdio.h>
#include <fstream>
using namespace std;

#include "lodepng.h"

namespace png {
    
    bool ReadImageFromFile(ITMUChar4Image* image, const char* fileName)
    {
        unsigned int xsize, ysize;
        unsigned char* data;
#define failed(lodepng_error) (lodepng_error != 0)
        if (failed(lodepng_decode24_file(&data, &xsize, &ysize, fileName))) return false;

        image->ChangeDims(Vector2i(xsize, ysize));
        Vector4u *dataPtr = image->GetData(MEMORYDEVICE_CPU);

        for (int i = 0; i < image->noDims.x*image->noDims.y; ++i)
        {
            dataPtr[i].x = data[i * 3 + 0];
            dataPtr[i].y = data[i * 3 + 1];
            dataPtr[i].z = data[i * 3 + 2];
            dataPtr[i].w = 255;
        }
        free(data);

        return true;
    }

    bool ReadImageFromFile(ITMShortImage *image, const char *fileName)
    {
        unsigned int xsize, ysize;
        short* data;
        if (failed(lodepng_decode_file((unsigned char**)&data, &xsize, &ysize, fileName, LCT_GREY, 16)))
            return false;

        image->ChangeDims(Vector2i(xsize, ysize));

        memcpy(image->GetData(MEMORYDEVICE_CPU),
            data,
            image->noDims.x*image->noDims.y * sizeof(short));

        free(data);

        return true;
    }

}