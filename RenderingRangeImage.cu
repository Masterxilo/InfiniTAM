#include "itmlibdefines.h"
#include "itmpose.h"

static __managed__ Matrix4f pose_global_to_eye;
static __managed__ Vector4f intrinsics;
static __managed__ Vector2i imgSize;
static __managed__ ITMFloatImage* zmins;
static __managed__ ITMFloatImage* zmaxs;

class RenderingRangeImage {
public:
    void build(ITMPose pose, Vector4f intrinsics, Vector2i imgSize);

private:
    void save(std::string baseFilename);
    // TODO combine into one image

};

#include "fileutils.h"
inline float interpolate(float val, float y0, float x0, float y1, float x1) {
    return (val - x0)*(y1 - y0) / (x1 - x0) + y0;
}

inline float base(float val) {
    if (val <= -0.75f) return 0.0f;
    else if (val <= -0.25f) return interpolate(val, 0.0f, -0.75f, 1.0f, -0.25f);
    else if (val <= 0.25f) return 1.0f;
    else if (val <= 0.75f) return interpolate(val, 1.0f, 0.25f, 0.0f, 0.75f);
    else return 0.0;
}

void DepthToUchar4(ITMUChar4Image *dst, ITMFloatImage *src)
{
    assert(dst->noDims == src->noDims);
    Vector4u *dest = dst->GetData(MEMORYDEVICE_CPU);
    float *source = src->GetData(MEMORYDEVICE_CPU);
    int dataSize = static_cast<int>(dst->dataSize);
    assert(dataSize > 1);
    memset(dst->GetData(MEMORYDEVICE_CPU), 0, dataSize * 4);

    Vector4u *destUC4;
    float lims[2], scale;

    destUC4 = (Vector4u*)dest;
    lims[0] = 100000.0f; lims[1] = -100000.0f;

    for (int idx = 0; idx < dataSize; idx++)
    {
        float sourceVal = source[idx];
        if (sourceVal > 0.0f) { lims[0] = MIN(lims[0], sourceVal); lims[1] = MAX(lims[1], sourceVal); }
    }

    scale = ((lims[1] - lims[0]) != 0) ? 1.0f / (lims[1] - lims[0]) : 1.0f / lims[1];

    if (lims[0] == lims[1]) assert(false); 

    for (int idx = 0; idx < dataSize; idx++)
    {
        float sourceVal = source[idx];

        if (sourceVal > 0.0f)
        {
            sourceVal = (sourceVal - lims[0]) * scale;

            destUC4[idx].r = (uchar)(base(sourceVal - 0.5f) * 255.0f);
            destUC4[idx].g = (uchar)(base(sourceVal) * 255.0f);
            destUC4[idx].b = (uchar)(base(sourceVal + 0.5f) * 255.0f);
            destUC4[idx].a = 255;
        }
    }
}
#include <memory>
void RenderingRangeImage::save(std::string baseFilename) {
    std::auto_ptr<ITMUChar4Image> depth(new ITMUChar4Image(imgSize));

    DepthToUchar4(depth.get(), zmins);
    png::SaveImageToFile(depth.get(), baseFilename + ".zmins.png");

    DepthToUchar4(depth.get(), zmaxs);
    png::SaveImageToFile(depth.get(), baseFilename + ".zmaxs.png");
}
#include "itmpixelutils.h"
struct InitZ {
    forEachPixelNoImage_process() {
        zmins->GetData()[locId] = viewFrustum_min;
        zmaxs->GetData()[locId] = viewFrustum_max;
    }
};

#include "scene.h"

GPU_ONLY inline bool ProjectSingleBlock(
    const THREADPTR(Vector3s) & blockPos,
    
    const THREADPTR(Matrix4f) & pose, const THREADPTR(Vector4f) & intrinsics, const THREADPTR(Vector2i) & imgSize, 
    
    THREADPTR(Vector2i) & upperLeft, THREADPTR(Vector2i) & lowerRight, THREADPTR(Vector2f) & zRange
    )
{
    upperLeft = imgSize;
    lowerRight = Vector2i(-1, -1);
    zRange = Vector2f(viewFrustum_max, viewFrustum_min);
    for (int corner = 0; corner < 8; ++corner)
    {
        // project all 8 corners down to 2D image
        Vector3s tmp = blockPos;
        tmp.x += (corner & 1) ? 1 : 0;
        tmp.y += (corner & 2) ? 1 : 0;
        tmp.z += (corner & 4) ? 1 : 0;
        Vector4f pt3d(TO_FLOAT3(tmp) * (float)SDF_BLOCK_SIZE * voxelSize, 1.0f);
        pt3d = pose * pt3d;
        if (pt3d.z < 1e-6) continue;

        Vector2f pt2d;
        pt2d.x = (intrinsics.x * pt3d.x / pt3d.z + intrinsics.z);
        pt2d.y = (intrinsics.y * pt3d.y / pt3d.z + intrinsics.w);

        // remember bounding box, zmin and zmax
        if (upperLeft.x > floor(pt2d.x)) upperLeft.x = (int)floor(pt2d.x);
        if (lowerRight.x < ceil(pt2d.x)) lowerRight.x = (int)ceil(pt2d.x);
        if (upperLeft.y > floor(pt2d.y)) upperLeft.y = (int)floor(pt2d.y);
        if (lowerRight.y < ceil(pt2d.y)) lowerRight.y = (int)ceil(pt2d.y);
        if (zRange.x > pt3d.z) zRange.x = pt3d.z;
        if (zRange.y < pt3d.z) zRange.y = pt3d.z;
    }

    // do some sanity checks and respect image bounds
    if (upperLeft.x < 0) upperLeft.x = 0;
    if (upperLeft.y < 0) upperLeft.y = 0;
    if (lowerRight.x >= imgSize.x) lowerRight.x = imgSize.x - 1;
    if (lowerRight.y >= imgSize.y) lowerRight.y = imgSize.y - 1;
    if (upperLeft.x > lowerRight.x) return false;
    if (upperLeft.y > lowerRight.y) return false;
    //if (zRange.y <= VERY_CLOSE) return false; never seems to happen
    if (zRange.x < viewFrustum_min) zRange.x = viewFrustum_min;
    if (zRange.y < viewFrustum_min) return false;

    return true;
}

struct DetermineBlockExtents {
    doForEachAllocatedVoxelBlock_process() {

        THREADPTR(Vector2i) upperLeft;
        THREADPTR(Vector2i) lowerRight;
        THREADPTR(Vector2f) zRange;
        if (!ProjectSingleBlock(voxelBlock->pos,
            ::pose_global_to_eye, ::intrinsics, ::imgSize,
            upperLeft, lowerRight, zRange
            ))
            return;
        assert(upperLeft.x >= 0 && upperLeft.y >= 0 && 
            upperLeft.x < imgSize.width && upperLeft.y < imgSize.height);
        assert(lowerRight.x >= 0 && lowerRight.y >= 0 &&
            lowerRight.x < imgSize.width && lowerRight.y < imgSize.height);

        assert(lowerRight.x >= upperLeft.x &&
            lowerRight.y >= upperLeft.y);
        assert(zRange.y >= zRange.x);


        // copy to image right away (todo: we can do better)
        for (int x = upperLeft.x; x < lowerRight.x; x++) {
            for (int y = upperLeft.y; y < lowerRight.y; y++) {

                atomicMin(
                    &zmins->GetData()[pixelLocId(x,y, imgSize)], 
                    zRange.x);

                atomicMax(
                    &zmaxs->GetData()[pixelLocId(x, y, imgSize)],
                    zRange.y);
            }
        }
    }
};
void RenderingRangeImage::build(ITMPose pose, Vector4f intrinsics, Vector2i imgSize) {
    ::pose_global_to_eye = pose.GetM();
    ::intrinsics = intrinsics;
    ::imgSize = imgSize;
    zmins = new ITMFloatImage(imgSize);
    zmaxs = new ITMFloatImage(imgSize);
    assert(imgSize.area() > 1);
    forEachPixelNoImage<InitZ>(imgSize);
    cudaDeviceSynchronize();
    assert(zmins->GetData()[0] == viewFrustum_min);
    assert(zmaxs->GetData()[0] == viewFrustum_max);

    Scene::getCurrentScene()->doForEachAllocatedVoxelBlock<DetermineBlockExtents>();
    /*
    // HACK to make save show something
    zmins->GetData()[0] = viewFrustum_max;
    zmaxs->GetData()[0] = viewFrustum_min;
    */

    save("RenderingRangeImage");
}
int main() {
    auto scene = new Scene();
    CURRENT_SCENE_SCOPE(scene);
    scene->restore("scenedump.dump");

    RenderingRangeImage r;
    r.build(ITMPose(), ITMIntrinsics().projectionParamsSimple.all, Vector2i(640, 480));
}