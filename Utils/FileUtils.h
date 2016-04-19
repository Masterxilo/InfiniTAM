// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include <stdio.h>
#include <io.h>
#include <string>

#include "itmview.h"
#include "itmpose.h"
#include "ITMLibDefines.h"

// .png files
/// \returns true iff operation succeeded
// image must exist and be allocated on cpu but can have any size. 
namespace png {       
    bool ReadImageFromFile(ITMUChar4Image* image, std::string fileName);
    bool ReadImageFromFile(ITMShortImage *image, std::string fileName);

    bool SaveImageToFile(const ITMUChar4Image* image, std::string fileName);
    bool SaveImageToFile(const ITMShortImage* image, std::string fileName);

}
// .dump files
namespace dump {
    template<typename T>
    static inline bool ReadImageFromFile(ORUtils::Image<T>* image, std::string fileName) {
        FILE* f = fopen(fileName.c_str(), "rb");
        if (!f) return false;
        Vector2i d;
        fread(&d, sizeof(d), 1, f);
        image->ChangeDims(d);
        fread(image->GetData(MEMORYDEVICE_CPU), sizeof(T) * d.area(), 1, f);
        fclose(f);
        return true;
    }
    template<typename T>
    static inline bool SaveImageToFile(const ORUtils::Image<T>* image, std::string fileName) {
        FILE* f = fopen(fileName.c_str(), "wb");
        if (!f) return false;
        Vector2i d = image->noDims;
        fwrite(&d, sizeof(d), 1, f);
        fwrite(image->GetData(MEMORYDEVICE_CPU), sizeof(T) * d.area(), 1, f);
        fclose(f);
        return true;
    }

    template<typename T>
    static inline bool ReadPODFromFile(T* o, std::string fileName) {
        FILE* f = fopen(fileName.c_str(), "rb");
        if (!f) return false;
        fread(o, sizeof(T), 1, f);
        fclose(f);
        return true;
    }
    template<typename T>
    static inline bool SavePODToFile(const T* o, std::string fileName) {
        FILE* f = fopen(fileName.c_str(), "wb");
        if (!f) return false;
        fwrite(o, sizeof(T), 1, f);
        fclose(f);
        return true;
    }

    // load<T>(fn)
    // store(T, fn) mechanism

    template<typename T>
    static inline T* load(std::string fn) {
        T* o = new T();
        assert(ReadPODFromFile(o, fn.c_str()));
        return o;
    }

    template<typename T>
    static inline void store(T* o, std::string fn){
        assert(SavePODToFile(o, fn.c_str()));
    }
    static int fileSize(std::string s) {
        HANDLE h = CreateFile(s.c_str(), 0, 0, 0, OPEN_EXISTING, 0, 0);
        assert(h != INVALID_HANDLE_VALUE);
        int x = GetFileSize(h, 0);
        assert(x > 0);
        CloseHandle(h);
        return x;
    }

}