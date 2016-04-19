#include "ITMMainEngine.h"


__managed__ int counter = 0;
static KERNEL buildBlockRequests(Vector3i offset) {
    Scene::requestCurrentSceneVoxelBlockAllocation(
        VoxelBlockPos(
        offset.x + blockIdx.x,
        offset.y + blockIdx.y,
        offset.z + blockIdx.z));
}
static __managed__ float radiusInWorldCoordinates;
static KERNEL buildSphereRequests() {
    Scene::requestCurrentSceneVoxelBlockAllocation(
        VoxelBlockPos(blockIdx.x,
        blockIdx.y,
        blockIdx.z));
}

struct BuildSphere {
    doForEachAllocatedVoxel_process() {
        assert(v);
        assert(radiusInWorldCoordinates > 0);

        // world-space coordinate position of current voxel
        Vector3f voxelGlobalPos = globalPoint.location;

        // Compute distance to origin
        const float distanceToOrigin = length(voxelGlobalPos);
        // signed distance to radiusInWorldCoordinates, positive when bigger
        const float dist = distanceToOrigin - radiusInWorldCoordinates;

        // Truncate and convert to -1..1 for band of size mu
        const float eta = dist;
        v->setSDF(MAX(MIN(1.0f, eta / mu), -1.f));

        // set color as if there where a white directional light at positive x 
        Vector3f n = normalize(voxelGlobalPos); // normal is normalized worldspace position for a sphere
        float cos = n.x > 0 ? n.x : 0;
        v->clr = Vector3u(cos * 255, cos * 255, cos * 255);


        v->w_color = 1;
        v->w_depth = 1;
    }
};

static KERNEL countAllocatedBlocks(Vector3i offset) {
    if (Scene::getCurrentSceneVoxel(
        VoxelBlockPos(
        offset.x + blockIdx.x,
        offset.y + blockIdx.y,
        offset.z + blockIdx.z).toInt() * SDF_BLOCK_SIZE
        ))
        atomicAdd(&counter, 1);
}
void buildSphereScene(const float radiusInWorldCoordinates) {
    assert(radiusInWorldCoordinates > 0);
    ::radiusInWorldCoordinates = radiusInWorldCoordinates;
    const float diameterInWorldCoordinates = radiusInWorldCoordinates * 2;
    int offseti = -ceil(radiusInWorldCoordinates / voxelBlockSize) - 1; // -1 for extra space
    assert(offseti < 0);

    Vector3i offset(offseti, offseti, offseti);
    int counti = ceil(diameterInWorldCoordinates / voxelBlockSize) + 2; // + 2 for extra space
    assert(counti > 0);
    dim3 count(counti, counti, counti);
    assert(offseti + count.x == -offseti);

    // repeat allocation a few times to avoid holes
    do {
        buildBlockRequests << <count, 1 >> >(offset);
        cudaDeviceSynchronize();
        Scene::performCurrentSceneAllocations();
        cudaDeviceSynchronize();
        counter = 0;
        countAllocatedBlocks << <count, 1 >> >(offset);
        cudaDeviceSynchronize();
    } while (counter != counti*counti*counti);

    Scene::getCurrentScene()->doForEachAllocatedVoxel<BuildSphere>();
}

// assumes buildWallRequests has been executed
// followed by perform allocations
// builds a solid wall, i.e.
// an trunctated sdf reaching 0 at 
// z == (SDF_BLOCK_SIZE / 2)*voxelSize
// and negative at bigger z.
struct BuildWall {
    doForEachAllocatedVoxel_process() {
        assert(v);

        float z = (threadIdx.z) * voxelSize;
        float eta = (SDF_BLOCK_SIZE / 2)*voxelSize - z;
        v->setSDF(MAX(MIN(1.0f, eta / mu), -1.f));
    }
};
void buildWallScene() {
    // Build wall scene
    buildBlockRequests << <dim3(10, 10, 1), 1 >> >(Vector3i(0, 0, 0));
    cudaDeviceSynchronize();
    Scene::performCurrentSceneAllocations();
    cudaDeviceSynchronize();
    Scene::getCurrentScene()->doForEachAllocatedVoxel<BuildWall>();
}

//void buildSphereScene(const float radiusInWorldCoordinates);
ITMMainEngine::ITMMainEngine(const ITMRGBDCalib *calib)
{
    scene = new Scene();
    CURRENT_SCENE_SCOPE(scene);
    
    //buildSphereScene(2 * voxelBlockSize);
    //buildWallScene();

    view = new ITMView(calib); // will be allocated by the view builder
}

ITMMainEngine::~ITMMainEngine()
{
	delete scene;

    delete view;
}

// HACK:
int frameCount = 0;
Matrix4f cameraMatrices[1000];
bool computeLighting;
void estimateLightingModel_();
void computeArtificialLighting_();

void ITMMainEngine::ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage)
{

    assert(rgbImage->noDims.area() > 1);
    assert(rawDepthImage->noDims.area() > 1);

    CURRENT_SCENE_SCOPE(scene);
    currentView = view;

    currentView->ChangeImages(rgbImage, rawDepthImage);
    cudaDeviceSynchronize();
    
    Matrix4f old_M_d = currentView->depthImage->eyeCoordinates->fromGlobal;
    assert(old_M_d == currentView->depthImage->eyeCoordinates->fromGlobal);
    ImprovePose();
    assert(old_M_d != currentView->depthImage->eyeCoordinates->fromGlobal);

    Fuse();

    // record camera toGlobal matrix
    cameraMatrices[frameCount++] = view->depthImage->eyeCoordinates->toGlobal;
    if (computeLighting) {
        computeArtificialLighting_();
            estimateLightingModel_();
    }
}
#include "fileutils.h"
#include <memory>

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
        float sourceVal = source[idx]; // only depths greater than 0 are considered
        if (sourceVal > 0.0f) { lims[0] = MIN(lims[0], sourceVal); lims[1] = MAX(lims[1], sourceVal); }
    }

    scale = ((lims[1] - lims[0]) != 0) ? 1.0f / (lims[1] - lims[0]) : 1.0f / lims[1];
    
    if (lims[0] == lims[1]) 
        assert(false);

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

void ITMMainEngine::GetImage(
    ITMUChar4Image * const out,
    ITMFloatImage * const outDepth,

    const ITMPose * const pose, 
    const ITMIntrinsics * const intrinsics,
    std::string shader
    )
{
    assert(out->noDims.area() > 1);
    assert(outDepth->noDims == out->noDims);
    CURRENT_SCENE_SCOPE(scene);
	auto ci = RenderImage(pose, intrinsics, out->noDims, outDepth, shader); // <- good place to stop freeze if a current scene should be set
    
    assert(outDepth->GetData()[0] >= 0);

    //std::auto_ptr<ITMUChar4Image> cDepth (new ITMUChar4Image(out->noDims));
    //DepthToUchar4(cDepth.get(), outDepth);
    //png::SaveImageToFile(cDepth.get(), "cDepth.png");

    out->SetFrom(ci->image, MemoryCopyDirection::CUDA_TO_CPU);
    delete ci;
}