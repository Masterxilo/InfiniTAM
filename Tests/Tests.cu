
#include "Scene.h" // defines: #include "HashMap.h"
#include <stdio.h>

template<typename T>
struct Z3Hasher {
    typedef T KeyType;
    static const uint BUCKET_NUM = 0x1000; // Number of Hash Bucket, must be 2^n (otherwise we have to use % instead of & below)

    static GPU_ONLY uint hash(const T& blockPos) {
        return (((uint)blockPos.x * 73856093u) ^ ((uint)blockPos.y * 19349669u) ^ ((uint)blockPos.z * 83492791u))
            &
            (uint)(BUCKET_NUM - 1);
    }
};


KERNEL get(HashMap<Z3Hasher<Vector3s>>* myHash, Vector3s q, int* o) {
    *o = myHash->getSequenceNumber(q);
}

KERNEL alloc(HashMap<Z3Hasher<Vector3s>>* myHash) {
    int p = blockDim.x * blockIdx.x + threadIdx.x;
    myHash->requestAllocation(p);
}

#include <vector>
using namespace std;
KERNEL assertfalse() {
    assert(false);
}


void testZ3Hasher() {
    //assertfalse << <1, 1 >> >();
    //assert(false);
    // insert a lot of points into a large hash just for fun
    HashMap<Z3Hasher<Vector3s>>* myHash = new HashMap<Z3Hasher<Vector3s>>(0x2000);

    int n = 1000;
    LAUNCH_KERNEL(alloc,n, 1 ,myHash);

    myHash->performAllocations();
    puts("after alloc");
    // should be some permutation of 1:n
    vector<bool> found; found.resize(n + 1);
    int* p; cudaMallocManaged(&p, sizeof(int));
    for (int i = 0; i < n; i++) {
        LAUNCH_KERNEL(get, 
            1, 1, 
            myHash, Vector3s(i, i, i), p);
        cudaSafeCall(cudaDeviceSynchronize()); // to read managed p
        printf("Vector3s(%i,%i,%i) -> %d\n", i, i, i, *p);

        assert(!found[*p]);
        found[*p] = 1;
    }
}

// n hasher test suite
// trivial hash function n -> n
struct NHasher{
    typedef int KeyType;
    static const uint BUCKET_NUM = 1;
    static GPU_ONLY uint hash(const int& n) {
        return n % BUCKET_NUM;//& (BUCKET_NUM-1);
    }
};

KERNEL get(HashMap<NHasher>* myHash, int p, int* o) {
    *o = myHash->getSequenceNumber(p);
}

KERNEL alloc(HashMap<NHasher>* myHash, int p, int* o) {
    myHash->requestAllocation(p);
}

void testNHasher() {
    int n = NHasher::BUCKET_NUM;
    auto myHash = new HashMap<NHasher>(1 + 1); // space for BUCKET_NUM entries only, and 1 collision handling entry

    int* p; cudaMallocManaged(&p, sizeof(int));

    for (int i = 0; i < n; i++) {

        LAUNCH_KERNEL(alloc,
            1, 1,
            myHash, i, p);
    }
    myHash->performAllocations();

    // an additional alloc at another key not previously seen (e.g. BUCKET_NUM) 
    alloc << <1, 1 >> >(myHash, NHasher::BUCKET_NUM, p);
    myHash->performAllocations();

    // an additional alloc at another key not previously seen (e.g. BUCKET_NUM + 1) makes it crash cuz no excess list
    //alloc << <1, 1 >> >(myHash, NHasher::BUCKET_NUM + 1, p);
    myHash->performAllocations(); // performAllocations is always fine to call when no extra allocations where made

    puts("after alloc");
    // should be some permutation of 1:BUCKET_NUM
    bool found[NHasher::BUCKET_NUM + 1] = {0};
    for (int i = 0; i < n; i++) {
        get << <1, 1 >> >(myHash, i, p);
        cudaDeviceSynchronize();
        printf("%i -> %d\n", i, *p);
        assert(!found[*p]);
        //assert(*p != i+1); // numbers are very unlikely to be in order -- nah it happens
        found[*p] = 1;
    }

}

// zero hasher test suite
// trivial hash function with one bucket.
// This will allow the allocation of only one block at a time
// and all blocks will be in the same list.
// The numbers will be in order.
struct ZeroHasher{
    typedef int KeyType;
    static const uint BUCKET_NUM = 0x1;
    static GPU_ONLY uint hash(const int&) { return 0; }
};

KERNEL get(HashMap<ZeroHasher>* myHash, int p, int* o) {
    *o = myHash->getSequenceNumber(p);
}

KERNEL alloc(HashMap<ZeroHasher>* myHash, int p, int* o) {
    myHash->requestAllocation(p);
}

void testZeroHasher() {
    int n = 10;
    auto myHash = new HashMap<ZeroHasher>(n); // space for BUCKET_NUM(1) + excessnum(n-1) = n entries

    int* p; cudaMallocManaged(&p, sizeof(int));

    const int extra = 0; // doing one more will crash it at
    // Assertion `excessListEntry >= 1 && excessListEntry < EXCESS_NUM` failed.

    // Keep requesting allocation until all have been granted
    for (int j = 0; j < n + extra; j++) { // request & perform alloc cycle
        for (int i = 0; i < n + extra
            ; i++) {
            alloc << <1, 1 >> >(myHash, i, p); // only one of these allocations will get through at a time
        }
        myHash->performAllocations();

        puts("after alloc");
        for (int i = 0; i < n; i++) {
            get << <1, 1 >> >(myHash, i, p);
            cudaDeviceSynchronize();
            printf("%i -> %d\n", i, *p);
            // expected result
            assert(i <= j ? *p == i + 1 : *p == 0);
        }
    }
}
#include "Cholesky.h"
using namespace ORUtils;
void testCholesky() {
    float m[] = {
        1, 0,
        0, 1
    };
    float b[] = {1, 2};
    float r[2];
    Cholesky::solve(m, 2, b, r);
    assert(r[0] == b[0] && r[1] == b[1]);

}

KERNEL addSceneVB(Scene* scene) {
    assert(scene);
    scene->requestVoxelBlockAllocation(VoxelBlockPos(0, 0, 0));
    scene->requestVoxelBlockAllocation(VoxelBlockPos(1,2,3));
}

GPU_ONLY void allExist(Scene* scene, Vector3i base) {
    for (int i = 0; i < SDF_BLOCK_SIZE; i++)
        for (int j = 0; j < SDF_BLOCK_SIZE; j++)
            for (int k = 0; k < SDF_BLOCK_SIZE; k++) {
                ITMVoxel* v = scene->getVoxel(base + Vector3i(i, j, k));
                assert(v != NULL);
            }
}
KERNEL findSceneVoxel(Scene* scene) {
    allExist(scene, Vector3i(0,0,0));
    allExist(scene, Vector3i(SDF_BLOCK_SIZE, 2*SDF_BLOCK_SIZE, 3*SDF_BLOCK_SIZE));

    assert(scene->getVoxel(Vector3i(-1, 0, 0)) == NULL);
}

KERNEL checkS(Scene* scene) {
    assert(Scene::getCurrentScene() == scene);
}

struct WriteEach {
    static GPU_ONLY void process(ITMVoxelBlock* vb, ITMVoxel* v, Vector3i localPos) {
        v->setSDF((
            localPos.x + 
            localPos.y * SDF_BLOCK_SIZE + 
            localPos.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE
            )/1024.f);
    }
};

__managed__ int counter = 0;
__managed__ bool visited[SDF_BLOCK_SIZE][SDF_BLOCK_SIZE][SDF_BLOCK_SIZE] = {0};
struct DoForEach {
    static GPU_ONLY void process(ITMVoxelBlock* vb, ITMVoxel* v, Vector3i localPos) {
        assert(localPos.x >= 0 && localPos.y >= 0 && localPos.z >= 0);
        assert(localPos.x  < SDF_BLOCK_SIZE && localPos.y < SDF_BLOCK_SIZE && localPos.z < SDF_BLOCK_SIZE);

        assert(vb);
        assert(vb->pos == VoxelBlockPos(0, 0, 0) ||
            vb->pos == VoxelBlockPos(1,2,3)); 

        visited[localPos.x][localPos.y][localPos.z] = 1;

        printf("%f .. %f\n", v->getSDF(),
            (
            localPos.x +
            localPos.y * SDF_BLOCK_SIZE +
            localPos.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE
            ) / 1024.f);
        assert(abs(
            v->getSDF() -
            (
            localPos.x +
            localPos.y * SDF_BLOCK_SIZE +
            localPos.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE
            ) / 1024.f) < 0.001 // not perfectly accurate
            );
        atomicAdd(&counter, 1);
    }
};
struct DoForEachBlock {
    static GPU_ONLY void process(ITMVoxelBlock* vb) {
        assert(vb);
        assert(vb->pos == VoxelBlockPos(0, 0, 0) ||
            vb->pos == VoxelBlockPos(1, 2, 3));
        atomicAdd(&counter, 1);
    }
};

KERNEL modifyS() {
    Scene::getCurrentSceneVoxel(Vector3i(0, 0, 1))->setSDF(1.0);
}

KERNEL checkModifyS() {
    assert(Scene::getCurrentSceneVoxel(Vector3i(0, 0, 1))->getSDF() == 1.0);
}

void testScene() {
    assert(Scene::getCurrentScene() == 0);

    Scene* s = new Scene(); 
    LAUNCH_KERNEL(addSceneVB, 1, 1, s);
    s->performAllocations();
    LAUNCH_KERNEL(findSceneVoxel, 1, 1, s);

    // current scene starts out at 0
    LAUNCH_KERNEL(checkS, 1, 1, 0);

    // change current scene
    {
        LAUNCH_KERNEL(checkS, 1, 1, 0); // still 0 before scope begins

        CURRENT_SCENE_SCOPE(s);
        LAUNCH_KERNEL(checkS, 1, 1, s);
        // Nest
        {
            CURRENT_SCENE_SCOPE(0);
            LAUNCH_KERNEL(checkS, 1, 1, 0);
        }
        LAUNCH_KERNEL(checkS, 1, 1, s);
    }
    LAUNCH_KERNEL(checkS, 1, 1, 0); // 0 again

    // modify current scene
    {
        CURRENT_SCENE_SCOPE(s);
        LAUNCH_KERNEL(modifyS, 1, 1);
        LAUNCH_KERNEL(checkModifyS, 1, 1);
    }

    // do for each

    s->doForEachAllocatedVoxel<WriteEach>();

    counter = 0;
    for (int x = 0; x < SDF_BLOCK_SIZE; x++)
        for (int y = 0; y < SDF_BLOCK_SIZE; y++)
            for (int z = 0; z < SDF_BLOCK_SIZE; z++)
                assert(!visited[x][y][z]);
    s->doForEachAllocatedVoxel<DoForEach>();
    cudaDeviceSynchronize();
    assert(counter == 2 * SDF_BLOCK_SIZE3);
    for (int x = 0; x < SDF_BLOCK_SIZE; x++)
        for (int y = 0; y < SDF_BLOCK_SIZE; y++)
            for (int z = 0; z < SDF_BLOCK_SIZE; z++)
                assert(visited[x][y][z]);

    counter = 0;
    s->doForEachAllocatedVoxelBlock<DoForEachBlock>();
    assert(counter == 2);

    delete s;
}

#define W 5
#define H 7
#include "ITMPixelUtils.h"
struct DoForEachPixel {
    forEachPixelNoImage_process() {
        assert(x >= 0 && x < W);
        assert(y >= 0 && y < H);
        atomicAdd(&counter, 1);
    }
};

void testForEachPixelNoImage() {
    counter = 0;
    forEachPixelNoImage<DoForEachPixel>(Vector2i(W, H));
    cudaDeviceSynchronize();
    assert(counter == W * H);
}

#include "FileUtils.h"
#include "itmcalibio.h"
#include "itmlibdefines.h"
#include "itmview.h"
#include "ITMMath.h"

#include <fstream>
#include <vector>
#include <algorithm>
using namespace std;
using namespace ITMLib;
using namespace ITMLib::Objects;
//using namespace ITMLib::Engine;

extern void ITMSceneReconstructionEngine_ProcessFrame_pre(
    const ITMView * const view,
    Matrix4f M_d
    );

void approxEqual(float a, float b) {
    assert(abs(a-b) < 0.00001);
}

void testAllocRequests(Matrix4f M_d, 
    const char* expectedRequestsFilename, 
    const char* missedExpectedRequestsFile,
    const char* possibleExtra = 0) {
    // [[
    std::vector<VoxelBlockPos> allExpectedRequests;
    {
        ifstream expectedRequests(expectedRequestsFilename);
        assert(expectedRequests.is_open());
        while (1) {
            VoxelBlockPos expectedBlockCoord;
            expectedRequests >> expectedBlockCoord.x >> expectedBlockCoord.y >> expectedBlockCoord.z;
            if (expectedRequests.fail()) break;
            allExpectedRequests.push_back(expectedBlockCoord);
        }
    }

    std::vector<VoxelBlockPos> allMissedExpectedRequests;
    {
        ifstream expectedRequests(missedExpectedRequestsFile);
        assert(expectedRequests.is_open());
        while (1) {
            VoxelBlockPos expectedBlockCoord;
            expectedRequests >> expectedBlockCoord.x >> expectedBlockCoord.y >> expectedBlockCoord.z;
            if (expectedRequests.fail()) break;
            allMissedExpectedRequests.push_back(expectedBlockCoord);
        }
    }

    // Some requests might be lost entirely sometimes
    std::vector<VoxelBlockPos> extra;
    if (possibleExtra)
    {
        ifstream expectedRequests(possibleExtra);
        assert(expectedRequests.is_open());
        while (1) {
            VoxelBlockPos expectedBlockCoord;
            expectedRequests >> expectedBlockCoord.x >> expectedBlockCoord.y >> expectedBlockCoord.z;
            if (expectedRequests.fail()) break;
            extra.push_back(expectedBlockCoord);
        }
    }

    // ]]
    ITMUChar4Image rgb(Vector2i(1,1), true, false);
    png::ReadImageFromFile(&rgb, "Tests\\TestAllocRequests\\color1.png");
    ITMShortImage depth(Vector2i(1, 1), true, false);
    png::ReadImageFromFile(&depth, "Tests\\TestAllocRequests\\depth1.png");
    ITMRGBDCalib calib;
    readRGBDCalib("Tests\\TestAllocRequests\\calib.txt", calib);

    ITMView* view = new ITMView(&calib, rgb.noDims, depth.noDims);
    ITMView::depthConversionType = "ConvertDisparityToDepth";
    view->Update(&rgb, &depth);

    assert(view->depth->noDims == Vector2i(640, 480));
    assert(view->rgb->noDims == Vector2i(640, 480));
    assert(view->calib->intrinsics_d.getInverseProjParams() ==
        Vector4f(0.00174304086	,
        0.00174096529	,
        346.471008	,
        249.031006	));

    Scene* scene = new Scene();
    CURRENT_SCENE_SCOPE(scene);


    // test ITMSceneReconstructionEngine_ProcessFrame_pre
    ITMSceneReconstructionEngine_ProcessFrame_pre(
        view, M_d
        );

    cudaDeviceSynchronize();

    // test content of requests "allocate planned"
    uchar *entriesAllocType = (uchar *)malloc(SDF_GLOBAL_BLOCK_NUM);
    Vector3s *blockCoords = (Vector3s *)malloc(SDF_GLOBAL_BLOCK_NUM * sizeof(Vector3s));

    cudaMemcpy(entriesAllocType,
        Scene::getCurrentScene()->voxelBlockHash->needsAllocation,
        SDF_GLOBAL_BLOCK_NUM,
        cudaMemcpyDeviceToHost);

    cudaMemcpy(blockCoords,
        Scene::getCurrentScene()->voxelBlockHash->naKey,
        SDF_GLOBAL_BLOCK_NUM * sizeof(VoxelBlockPos),
        cudaMemcpyDeviceToHost);
    {
        ifstream expectedRequests(expectedRequestsFilename);
        assert(expectedRequests.is_open());
        VoxelBlockPos expectedBlockCoord;
        bool read = true;
        for (int targetIdx = 0; targetIdx < SDF_GLOBAL_BLOCK_NUM; targetIdx++) {
            if (entriesAllocType[targetIdx] == 0) continue;
            
            if (read)
                expectedRequests >> expectedBlockCoord.x >> expectedBlockCoord.y >> expectedBlockCoord.z;
            read = true;
            assert(!expectedRequests.fail());

            printf("expecting %d %d %d got %d %d %d\n", 
                xyz(expectedBlockCoord),
                xyz(blockCoords[targetIdx])
                );

            if (expectedBlockCoord != blockCoords[targetIdx]) {
                // If the expectedBlockCoord is not in this file, it must be in the missed requests - 
                // it is not deterministic which blocks will be allocated first and which on the second run
                auto i = find(allMissedExpectedRequests.begin(), allMissedExpectedRequests.end(),
                    blockCoords[targetIdx]);
                if (i == allMissedExpectedRequests.end()) {
                    auto i = find(
                        extra.begin(),
                        extra.end(),
                        blockCoords[targetIdx]);
                    read = false;
                    assert(i != extra.end());
                }
            }

            continue;
        }
        // Must have seen all requests
        int _;
        expectedRequests >> _;
        assert(expectedRequests.fail());
    }

    // do allocations
    Scene::performCurrentSceneAllocations();

    cudaDeviceSynchronize();
    // --- again!
    // test ITMSceneReconstructionEngine_ProcessFrame_pre
    ITMSceneReconstructionEngine_ProcessFrame_pre(
        view, M_d
        );

    cudaDeviceSynchronize();

    // test content of requests "allocate planned"
    cudaMemcpy(entriesAllocType,
        Scene::getCurrentScene()->voxelBlockHash->needsAllocation,
        SDF_GLOBAL_BLOCK_NUM,
        cudaMemcpyDeviceToHost);

    cudaMemcpy(blockCoords,
        Scene::getCurrentScene()->voxelBlockHash->naKey,
        SDF_GLOBAL_BLOCK_NUM * sizeof(VoxelBlockPos),
        cudaMemcpyDeviceToHost);

    {
        ifstream expectedRequests(missedExpectedRequestsFile);
        assert(expectedRequests.is_open());
        for (int targetIdx = 0; targetIdx < SDF_GLOBAL_BLOCK_NUM; targetIdx++) {
            if (entriesAllocType[targetIdx] == 0) continue;
            VoxelBlockPos expectedBlockCoord;
            expectedRequests >> expectedBlockCoord.x >> expectedBlockCoord.y >> expectedBlockCoord.z;

            if (expectedBlockCoord != blockCoords[targetIdx]) {
                // If the expectedBlockCoord is not in this file, it must be in the missed requests - 
                // it is not deterministic which blocks will be allocated first and which on the second run
                auto i = find(allExpectedRequests.begin(), allExpectedRequests.end(),
                    blockCoords[targetIdx]);
                if (i == allExpectedRequests.end()) {
                    auto i = find(
                        extra.begin(),
                        extra.end(),
                        blockCoords[targetIdx]);
                    assert(i != extra.end());
                }
            }

        }
        // Must have seen all requests
        int _;
        expectedRequests >> _;
        assert(expectedRequests.fail());
    }

    delete scene;
    delete view;
}
/// Must exist on cpu
template<typename T>
bool checkImageSame(Image<T>* a_, Image<T>* b_) {
    T* a = a_->GetData(MEMORYDEVICE_CPU);
    T* b = b_->GetData(MEMORYDEVICE_CPU);
#define failifnot(x) if (!(x)) return false;
    failifnot(a_->dataSize == b_->dataSize);
    failifnot(a_->noDims == b_->noDims);
    int s = a_->dataSize;
    while (s--) {
        if (*a != *b) {
            png::SaveImageToFile(a_, "checkImageSame_a.png");
            png::SaveImageToFile(b_, "checkImageSame_b.png");
            failifnot(false);
        }
        a++;
        b++;
    }
    return true;
}

/// Must exist on cpu
template<typename T>
void assertImageSame(Image<T>* a_, Image<T>* b_) {
    assert(checkImageSame(a_,b_));
}
ITMUChar4Image* load(const char* fn) {

    ITMUChar4Image* i = new ITMUChar4Image(Vector2i(1, 1), true, false);
    png::ReadImageFromFile(i, fn);
    return i;
}

void testImageSame() {
    auto i = load("Tests\\TestAllocRequests\\color1.png");
    assertImageSame(i,i);
    delete i;
}

#include "itmlib.h"
#include "ITMVisualisationEngine.h"

ITMUChar4Image* renderNow(Vector2i imgSize, ITMPose* pose) {
    ITMRenderState* renderState_freeview = new ITMRenderState(imgSize);
    auto render = new ITMUChar4Image(imgSize, true, true);

    ITMVisualisationEngine::RenderImage(
        pose, new ITMIntrinsics(),
        renderState_freeview,
        render,
        ITMVisualisationEngine::RENDER_SHADED_GREYSCALE);
    render->UpdateHostFromDevice();
    delete renderState_freeview;
    return render;
}

void renderExpecting(const char* fn, ITMPose* pose = new ITMPose()) {
    auto expect = load(fn);
    auto render = renderNow(expect->noDims, pose);
    assertImageSame(expect, render);
    delete expect;
    delete render;
    delete pose;
}

#define make(scene) Scene* scene = new Scene(); CURRENT_SCENE_SCOPE(scene);

void testRenderBlack() {
    make(scene);
    renderExpecting("Tests\\TestRender\\black.png");
    delete scene;
}


static KERNEL buildWallRequests() {
    Scene::requestCurrentSceneVoxelBlockAllocation(
        VoxelBlockPos(blockIdx.x,
        blockIdx.y,
        blockIdx.z));
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

void testRenderWall() {
    make(scene);

    // Build wall scene
    buildWallRequests << <dim3(10, 10, 1), 1 >> >();
    cudaDeviceSynchronize();
    Scene::performCurrentSceneAllocations();
    cudaDeviceSynchronize();
    Scene::getCurrentScene()->doForEachAllocatedVoxel<BuildWall>();

    // move everything away a bit so we can see the wall
    auto pose = new ITMPose();
    pose->SetT(Vector3f(0, 0, voxelSize * 100)); 

    renderExpecting("Tests\\TestRender\\wall.png", pose);
    delete scene;
}

void testAllocRequests() {

    // With the alignment generated by the (buggy?) original Track Camera on the first frame,
    // no conflicts occur
    Matrix4f M_d;
    M_d.m00 = 0.848863006;
    M_d.m01 = 0.441635638;
    M_d.m02 = -0.290498704;
    M_d.m03 = 0.000000000;
    M_d.m10 = -0.290498704;
    M_d.m11 = 0.848863065;
    M_d.m12 = 0.441635549;
    M_d.m13 = 0.000000000;
    M_d.m20 = 0.441635638;
    M_d.m21 = -0.290498614;
    M_d.m22 = 0.848863065;
    M_d.m23 = 0.000000000;
    M_d.m30 = -0.144862041;
    M_d.m31 = -0.144861951;
    M_d.m32 = -0.144861966;
    M_d.m33 = 1.00000000;

    Matrix4f invM_d; M_d.inv(invM_d);
    approxEqual(invM_d.m00, 0.848863125); // exactly equal should do
    approxEqual(invM_d.m01, -0.290498734);
    approxEqual(invM_d.m02, 0.441635668);
    approxEqual(invM_d.m03, 0.000000000);
    approxEqual(invM_d.m10, 0.441635668);
    approxEqual(invM_d.m11, 0.848863184);
    approxEqual(invM_d.m12, -0.290498614);
    approxEqual(invM_d.m13, 0.000000000);
    approxEqual(invM_d.m20, -0.290498734);
    approxEqual(invM_d.m21, 0.441635609);
    approxEqual(invM_d.m22, 0.848863184);
    approxEqual(invM_d.m23, 0.000000000);
    approxEqual(invM_d.m30, 0.144862026);
    approxEqual(invM_d.m31, 0.144861937);
    approxEqual(invM_d.m32, 0.144862041);
    approxEqual(invM_d.m33, 1.00000012);
    testAllocRequests(M_d, "Tests\\TestAllocRequests\\expectedRequests.txt"
        , "Tests\\TestAllocRequests\\expectedMissedRequests.txt");

}

void testAllocRequests2() {

    // With identity matrix, we have some conflicts that are only resolved on a second allocation pass
    testAllocRequests(Matrix4f(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1), "Tests\\TestAllocRequests\\expectedRequests2.txt"
        , "Tests\\TestAllocRequests\\expectedMissedRequests2.txt"
        , "Tests\\TestAllocRequests\\possibleExtraRequests2.txt"
        );
}


void testDump() {
    auto i = load("Tests\\TestAllocRequests\\color1.png");
    assert(dump::SaveImageToFile(i, "Tests\\TestAllocRequests\\color1.dump"));
    auto j = new ITMUChar4Image(Vector2i(1, 1), true, false);
    assert(dump::ReadImageFromFile(j, "Tests\\TestAllocRequests\\color1.dump"));
    assertImageSame(i, j);
    delete i;
    delete j;
}

// TODO take the tests apart, clean state inbetween
void tests() {
    assert(!checkImageSame(load("Tests\\TestRender\\wall.png"), load("Tests\\TestRender\\black.png")));
    assert(!dump::ReadImageFromFile(new ITMUChar4Image(Vector2i(1, 1), true, false), "thisimagedoesnotexist"));
    assert(!png::ReadImageFromFile(new ITMUChar4Image(Vector2i(1, 1), true, false), "thisimagedoesnotexist"));
    assert(!png::SaveImageToFile(new ITMUChar4Image(Vector2i(1, 1), true, false), "C:\\cannotSaveHere.png"));
    testImageSame();
    testDump();
    testForEachPixelNoImage();
    testRenderBlack();
    testRenderWall();
    testScene();
    testCholesky();
    testZ3Hasher();
    testNHasher();
    testZeroHasher();
    testAllocRequests();
    testAllocRequests2();

    puts("==== All tests passed ====");
}
