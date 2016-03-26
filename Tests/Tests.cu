#include "HashMap.h"
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

// TODO take the tests apart, clean state inbetween
void tests() {
    testCholesky();
    testZ3Hasher();
    testNHasher();
    testZeroHasher();
}