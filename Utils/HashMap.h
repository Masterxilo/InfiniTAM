#pragma once 
#include "CUDADefines.h"
#include "ITMMath.h"
#include "MemoryBlock.h"
#include "ITMCUDAUtils.h"

// Forward declarations
template<typename Hasher, typename AllocCallback> class HashMap;
template<typename Hasher, typename AllocCallback>
KERNEL performAllocationKernel(typename HashMap<Hasher, AllocCallback>* hashMap);

#define hprintf(...) //printf(__VA_ARGS__) // enable for verbose radio debug messages

struct VoidSequenceIdAllocationCallback {
    template<typename T>
    static __device__ void allocate(T, int sequenceId) {}
};
/**
Implements a

key -> sequence#

mapping on the GPU, where keys for which allocation is requested get assigned unique, consecutive unsigned integer numbers
starting at 1.

Stores at most Hasher::BUCKET_NUM + EXCESS_NUM - 1 entries.

After a series of requestAllocation(key) calls, performAllocations() must be called to make getSequenceId(key)
return a unique nonzero value for key.
Allocation is not guaranteed in only one requestAllocation(key) -> performAllocations() cycle:
At most one entry will be allocated per hash(key) in one such cycle.

Note: As this reads from global memory it might be advisable to cache results,
especially when it is expected that the same entry is accessed multiple times from the same thread.
TODO Can we provide this functionality from here? Maybe force the creation of some object including a cache to access this.
*/
/* Implementation: See HashMap.png */
template<
    typename Hasher, //!< must have static __device__ function uint Hasher::hash(const KeyType&) which generates values from 0 to Hasher::BUCKET_NUM-1 
    typename SequenceIdAllocationCallback = VoidSequenceIdAllocationCallback //!< must have static __device__ void  allocate(KeyType k, int sequenceId) function
>
class HashMap : public Managed {
public:
    typedef Hasher::KeyType KeyType;

private:
    static const uint BUCKET_NUM = Hasher::BUCKET_NUM;
    const uint EXCESS_NUM;
    CPU_AND_GPU uint NUMBER_TOTAL_ENTRIES() const {
        return (BUCKET_NUM + EXCESS_NUM);
    }

    struct HashEntry {
    public:
        GPU_ONLY bool isAllocated() {
            return sequenceId != 0;
        }
        GPU_ONLY bool hasNextExcessList() {
            assert(isAllocated());
            return nextInExcessList != 0;
        }
        GPU_ONLY uint getNextInExcessList() {
            assert(hasNextExcessList() && isAllocated());
            return nextInExcessList;
        }

        GPU_ONLY bool hasKey(const KeyType& key) {
            assert(isAllocated());
            return this->key == key;
        }

        GPU_ONLY void linkToExcessListEntry(const uint excessListId) {
            assert(!hasNextExcessList() && isAllocated() && excessListId >= 1);// && excessListId < EXCESS_NUM);
            // also, the excess list entry should exist and this should be the only entry linking to it
            // all entries in the excess list before this one should be allocated
            nextInExcessList = excessListId;
        }

        GPU_ONLY void allocate(const KeyType& key, const uint sequenceId) {
            assert(!isAllocated() && nextInExcessList == 0);
            assert(sequenceId > 0);
            this->key = key;
            this->sequenceId = sequenceId;

            SequenceIdAllocationCallback::allocate(key, sequenceId);

            hprintf("allocated %d\n", sequenceId);
        }

        GPU_ONLY uint getSequenceId() {
            assert(isAllocated());
            return sequenceId;
        }
    private:
        KeyType key;
        /// any of 1 to lowestFreeExcessListEntry-1
        /// 0 means this entry ends a list of excess entries
        uint nextInExcessList;
        /// any of 1 to lowestFreeSequenceNumber-1
        /// 0 means this entry is not allocated
        uint sequenceId;
    };

    /// 0 or 1, BUCKET_NUM + EXCESS_NUM many
    GPU(uchar*) needsAllocation;
    GPU(KeyType*) naKey;

    /// BUCKET_NUM + EXCESS_NUM many
    /// Indexed by Hasher::hash() return value
    // or BUCKET_NUM + HashEntry.nextInExcessList (which is any of 1 to lowestFreeExcessListEntry-1)
    GPU(HashEntry*) hashMap_then_excessList;


    GPU_ONLY HashEntry& hashMap(const uint hash) {
        assert(hash < BUCKET_NUM);
        return hashMap_then_excessList[hash];
    }
    GPU_ONLY HashEntry& excessList(const uint excessListEntry) {
        assert(excessListEntry >= 1 && excessListEntry < EXCESS_NUM);
        return hashMap_then_excessList[BUCKET_NUM + excessListEntry];
    }


    /// Sequence numbers already used up. Starts at 1 (sequence number 0 is used to signify non-allocated)
    GPU(uint*) lowestFreeSequenceNumber;

    /// Excess list slots already used up. Starts at 1 (one safeguard entry)
    GPU(uint*) lowestFreeExcessListEntry;

    /// Follows the excess list starting at hashMap[Hasher::hash(key)]
    /// until either hashEntry.key == key, returning true
    /// or until hashEntry does not exist or hashEntry.key != key but there is no further entry, returns false in that case.
    GPU_ONLY bool findEntry(const KeyType& key,//!< [in]
        HashEntry& hashEntry, //!< [out]
        uint& hashMap_then_excessList_entry //!< [out]
        ) {
        hashMap_then_excessList_entry = Hasher::hash(key);
        hprintf("%d %d\n", hashMap_then_excessList_entry, BUCKET_NUM);
        assert(hashMap_then_excessList_entry < BUCKET_NUM);
        hashEntry = hashMap(hashMap_then_excessList_entry);

        if (!hashEntry.isAllocated()) return false;
        if (hashEntry.hasKey(key)) return true;

        // try excess list
        int safe = 0;
        while (hashEntry.hasNextExcessList()) {
            hashEntry = excessList(hashMap_then_excessList_entry = hashEntry.getNextInExcessList());
            hashMap_then_excessList_entry += BUCKET_NUM; // the hashMap_then_excessList_entry must include the offset by BUCKET_NUM
            if (hashEntry.hasKey(key)) return true;
            if (safe++ > 100) assert(false);
        }
        return false;
    }

    /** Allocate a block of CUDA memory and memset it to 0 */
    template<typename T> static void zeroMalloc(T*& p, const uint count = 1) {
        cudaSafeCall(cudaMalloc(&p, sizeof(T) * count));
        cudaSafeCall(cudaMemset(p, 0, sizeof(T) * count));
    }

    GPU_ONLY void allocate(HashEntry& hashEntry, const KeyType & key) {
        hashEntry.allocate(key, atomicAdd(lowestFreeSequenceNumber, 1));
    }

    friend KERNEL performAllocationKernel<Hasher, SequenceIdAllocationCallback>(typename HashMap<Hasher, SequenceIdAllocationCallback>* hashMap);

    GPU_ONLY void performAllocation(const uint hashMap_then_excessList_entry) {
        if (hashMap_then_excessList_entry >= NUMBER_TOTAL_ENTRIES()) return;
        if (!needsAllocation[hashMap_then_excessList_entry]) return;
        assert(hashMap_then_excessList_entry != BUCKET_NUM); // never allocate guard
        hprintf("performAllocation %d\n", hashMap_then_excessList_entry);


        needsAllocation[hashMap_then_excessList_entry] = false;
        KeyType key = naKey[hashMap_then_excessList_entry];

        // Allocate in place if not allocated
        HashEntry& hashEntry = hashMap_then_excessList[hashMap_then_excessList_entry];

        if (!hashEntry.isAllocated()) {
            hprintf("not allocated\n", hashMap_then_excessList_entry);
            allocate(hashEntry, key);
            return;
        }

        hprintf("hashEntry %d\n", hashEntry.getSequenceId());

        // If existing, allocate new and link parent to child
        uint excessListId = atomicAdd(lowestFreeExcessListEntry, 1);
        HashEntry& newHashEntry = excessList(excessListId);
        assert(!newHashEntry.isAllocated());
        hashEntry.linkToExcessListEntry(excessListId);
        assert(hashEntry.getNextInExcessList() == excessListId);

        allocate(newHashEntry, key);
        hprintf("newHashEntry.getSequenceId() = %d\n", newHashEntry.getSequenceId());

#ifdef _DEBUG
        // should now find this entry:
        HashEntry e; uint _;
        bool found = findEntry(key, e, _);
        assert(found && e.getSequenceId() > 0);
        hprintf("%d = findEntry(), e.seqId = %d\n", found, e.getSequenceId());
#endif
    }


public:
    HashMap(const uint EXCESS_NUM //<! must be at least one
        ) : EXCESS_NUM(EXCESS_NUM) {
        assert(EXCESS_NUM >= 1);
        zeroMalloc(needsAllocation, NUMBER_TOTAL_ENTRIES());
        zeroMalloc(naKey, NUMBER_TOTAL_ENTRIES()); // TODO zeroing not necessary
        zeroMalloc(hashMap_then_excessList, NUMBER_TOTAL_ENTRIES());

        cudaMallocManaged(&lowestFreeSequenceNumber, sizeof(uint));
        cudaMallocManaged(&lowestFreeExcessListEntry, sizeof(uint));

        cudaDeviceSynchronize();
        *lowestFreeSequenceNumber = *lowestFreeExcessListEntry = 1;
    }

    virtual ~HashMap() {
        cudaFree(needsAllocation);
        cudaFree(naKey);
        cudaFree(hashMap_then_excessList);
        cudaFree(lowestFreeSequenceNumber);
        cudaFree(lowestFreeExcessListEntry);
    }
    
    uint getLowestFreeSequenceNumber() {
    return *lowestFreeSequenceNumber;
    }
/*
    uint countAllocatedEntries() {
    return getLowestFreeSequenceNumber() - 1;
    }
    */

    /**
    Requests allocation for a specific key.
    Only one request can be made per hash(key) before performAllocations must be called.
    Further requests will be ignored.
    */
    GPU_ONLY void requestAllocation(const KeyType& key) {
        hprintf("requestAllocation \n");

        HashEntry hashEntry; uint hashMap_then_excessList_entry;

        bool alreadyExists = findEntry(key, hashEntry, hashMap_then_excessList_entry);
        if (alreadyExists) {
            hprintf("already exists\n");
            return;
        }
        hprintf("request goes to %d\n", hashMap_then_excessList_entry);

        // not strictly necessary, ordering is random anyways
        if (needsAllocation[hashMap_then_excessList_entry]) {
            hprintf("already requested\n");
            return;
        }

        assert(hashMap_then_excessList_entry != BUCKET_NUM &&
            hashMap_then_excessList_entry < NUMBER_TOTAL_ENTRIES());

        needsAllocation[hashMap_then_excessList_entry] = true;
        naKey[hashMap_then_excessList_entry] = key;
    }
#define THREADS_PER_BLOCK 256
    /**
    Allocates entries that requested allocation. Allocates at most one entry per hash(key).
    Further requests can allocate colliding entries.
    */
    void performAllocations() {
        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaDeviceSynchronize()); // Managed this is not accessible when still in use?
        performAllocationKernel << <
            /// Scheduling strategy: Fixed number of threads per block, working on all entries (to find those that have needsAllocation set)
            (uint)ceil(NUMBER_TOTAL_ENTRIES() / (1. * THREADS_PER_BLOCK)),
            THREADS_PER_BLOCK
            >> >(this);
#ifdef _DEBUG
        cudaSafeCall(cudaDeviceSynchronize());  // detect problems (failed assertions) early where this kernel is called
#endif
        cudaSafeCall(cudaGetLastError());
    }

    /// \returns 0 if the key is not allocated
    GPU_ONLY uint getSequenceNumber(const KeyType& key) {
        HashEntry hashEntry; uint _;
        if (!findEntry(key, hashEntry, _)) return 0;
        return hashEntry.getSequenceId();
    }
};

template<typename Hasher, typename AllocCallback>
KERNEL performAllocationKernel(typename HashMap<Hasher, AllocCallback>* hashMap) {
    assert(blockDim.x == THREADS_PER_BLOCK && blockDim.y == 1 && blockDim.z == 1);
    assert(
        gridDim.x*blockDim.x >= hashMap->NUMBER_TOTAL_ENTRIES() && // all entries covered
        gridDim.y == 1 &&
        gridDim.z == 1);

    hashMap->performAllocation(blockIdx.x*THREADS_PER_BLOCK + threadIdx.x);
}
