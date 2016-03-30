#include "FileUtils.h"
#include <stdio.h>
#include <fstream>
using namespace std;
#include "image.h"
using namespace ORUtils;

#include "lodepng.h"
// lodepng_*: "Return value: LodePNG error code (0 means no error)."
#define failed(lodepng_error) (lodepng_error != 0)

namespace png {
    template<typename T, LodePNGColorType colorType, int bitdepth_per_channel>
    static bool ReadImageFromFile_(Image<T>* image, const char* fileName)
    {
        unsigned int xsize, ysize;
        unsigned char* data = 0;
        if (failed(lodepng_decode_file(&data, &xsize, &ysize, fileName,
            colorType, bitdepth_per_channel))) return false;

        image->ChangeDims(Vector2i(xsize, ysize));

        memcpy(image->GetData(MEMORYDEVICE_CPU),
            data,
            image->noDims.x*image->noDims.y * sizeof(T));

        free(data);
        return true;
    }
    template<typename T, LodePNGColorType colorType, int bitdepth_per_channel>
    static bool SaveImageToFile_(const Image<T>* image, const char* fileName)
    {
        return !failed(lodepng_encode_file(
            fileName, 
            (unsigned char*)image->GetData(MEMORYDEVICE_CPU),
            image->noDims.x, image->noDims.y,
            colorType, bitdepth_per_channel));
    }

    bool ReadImageFromFile(ITMUChar4Image* image, const char* fileName)
    {
        return ReadImageFromFile_<Vector4u, LodePNGColorType::LCT_RGBA, 8>(image, fileName);
    }

    bool ReadImageFromFile(ITMShortImage* image, const char* fileName)
    {
        return ReadImageFromFile_<short, LodePNGColorType::LCT_GREY, 16>(image, fileName);
    }

    bool SaveImageToFile(const ITMUChar4Image* image, const char* fileName)
    {
        return SaveImageToFile_<Vector4u, LodePNGColorType::LCT_RGBA, 8>(image, fileName);
    }

    bool SaveImageToFile_(const ITMShortImage* image, const char* fileName)
    {
        return SaveImageToFile_<short, LodePNGColorType::LCT_GREY, 16>(image, fileName);
    }
}