// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include <stdio.h>

#include "../ITMLib/Utils/ITMLibDefines.h"

// .png files
/// \returns true iff operation succeeded
// image must exist and be allocated on cpu but can have any size. 
namespace png {       
    bool ReadImageFromFile(ITMUChar4Image* image, const char* fileName);
    bool ReadImageFromFile(ITMShortImage *image, const char *fileName);

    bool SaveImageToFile(const ITMUChar4Image* image, const char* fileName);
    bool SaveImageToFile(const ITMShortImage* image, const char* fileName);

}
// .dump files
namespace dump {
    template<typename T>
    inline bool ReadImageFromFile(ORUtils::Image<T>* image, const char* fileName) {
        FILE* f = fopen(fileName, "rb");
        if (!f) return false;
        Vector2i d;
        fread(&d, sizeof(d), 1, f);
        image->ChangeDims(d);
        fread(image->GetData(MEMORYDEVICE_CPU), sizeof(T) * d.area(), 1, f);
        fclose(f);
        return true;
    }
    template<typename T>
    inline bool SaveImageToFile(const ORUtils::Image<T>* image, const char* fileName) {
        FILE* f = fopen(fileName, "wb");
        if (!f) return false;
        Vector2i d = image->noDims;
        fwrite(&d, sizeof(d), 1, f);
        fwrite(image->GetData(MEMORYDEVICE_CPU), sizeof(T) * d.area(), 1, f);
        fclose(f);
        return true;
    }

    template<typename T>
    inline bool ReadPODFromFile(T* o, const char* fileName) {
        FILE* f = fopen(fileName, "rb");
        if (!f) return false;
        fread(o, sizeof(T), 1, f);
        fclose(f);
        return true;
    }
    template<typename T>
    inline bool SavePODToFile(const T* image, const char* fileName) {
        FILE* f = fopen(fileName, "wb");
        if (!f) return false;
        fwrite(o, sizeof(T), 1, f);
        fclose(f);
        return true;
    }

}