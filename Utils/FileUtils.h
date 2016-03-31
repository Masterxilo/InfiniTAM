// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include <stdio.h>
#include <io.h>
#include <string>

#include "itmview.h"
#include "itmpose.h"
#include "itmtrackingstate.h"
#include "ITMLibDefines.h"
using namespace ITMLib;
using namespace ITMLib::Objects;

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

    // View and tracking state
    template<>
    static inline ITMView* load(std::string f) {
        Vector2i 
            imgSize_rgb = *load<Vector2i>(f + ".imgSize_rgb"), 
            imgSize_d = *load<Vector2i>(f + ".imgSize_d");
        ITMRGBDCalib *calib = load<ITMRGBDCalib>(f + ".calib");
        ITMView* v = new ITMView(calib, imgSize_rgb, imgSize_d);
        ReadImageFromFile(v->depth, f+".depth");
        ReadImageFromFile(v->rgb, f + ".rgb");
        return v;
    }
    template<>
    static inline void store(ITMView* v, std::string f) {
        store(&v->rgb->noDims, f + ".imgSize_rgb");
        store(&v->depth->noDims, f + ".imgSize_d");
        store(v->calib, f + ".calib");
        assert(fileSize(f + ".calib") == sizeof(ITMRGBDCalib));
        SaveImageToFile(v->depth, f + ".depth");
        SaveImageToFile(v->rgb, f + ".rgb");
    }

    template<>
    static inline ITMTrackingState* load(std::string f) {
        Vector2i
            imgSize_d = *load<Vector2i>(f + ".imgSize_d");
        ITMTrackingState* v = new ITMTrackingState(imgSize_d);


        v->pose_d = load<ITMPose>(f + ".pose_d");



        v->pointCloud->pose_pointCloud = load<ITMPose>(f + ".pose_pointCloud");
        assert(ReadImageFromFile(v->pointCloud->locations, f + ".locations"));
        assert(ReadImageFromFile(v->pointCloud->normals, f + ".normals"));
        return v;
    }
    template<>
    static inline void store(ITMTrackingState* v, std::string f) {
        store(&v->pointCloud->locations->noDims, f + ".imgSize_d");

        store(v->pose_d, f + ".pose_d");
        assert(fileSize(f + ".pose_d") == sizeof(ITMPose));
        store(v->pointCloud->pose_pointCloud, f + ".pose_pointCloud");
        assert(fileSize(f + ".pose_pointCloud") == sizeof(ITMPose));

        assert(SaveImageToFile(v->pointCloud->locations, f + ".locations"));
        assert(SaveImageToFile(v->pointCloud->normals, f + ".normals"));
    }

}