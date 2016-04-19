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
        Vector3f voxelGlobalPos = (vb->getPos().toFloat() * SDF_BLOCK_SIZE + localPos.toFloat()) * voxelSize;

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
    static GPU_ONLY void process(const ITMVoxelBlock* vb, ITMVoxel* v, const Vector3i localPos) {
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
    
    buildSphereScene(2 * voxelBlockSize);
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

void ITMMainEngine::ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage)
{
    return;

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
    if (computeLighting)
        estimateLightingModel_();
}
void ITMMainEngine::GetImage(
    ITMUChar4Image * const out,
    const ITMPose * const pose, 
    const ITMIntrinsics * const intrinsics,
    std::string shader
    )
{
    assert(out->noDims.area() > 1);
    CURRENT_SCENE_SCOPE(scene);
	auto ci = RenderImage(pose, intrinsics, out->noDims, shader);
    out->SetFrom(ci->image, MemoryCopyDirection::CUDA_TO_CPU);
    delete ci;
}