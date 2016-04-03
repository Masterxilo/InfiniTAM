// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "PlatformIndependence.h"
#include "CUDADefines.h"
#include "itmcudautils.h"


enum MemoryDeviceType { MEMORYDEVICE_CPU, MEMORYDEVICE_CUDA };

namespace ORUtils
{
	/** \brief
	Represents memory blocks, templated on the data type, 
    behaving as if always allocated on both the gpu and the cpu, 
    but not automatically synched.

    TODO If the cpu or gpu version is never accessed, the memory never has to be allocated.
    TODO acess the data in a write-only manner so that the update..from.. dont have to be called.
	*/
	template <typename T>
	class MemoryBlock : public Managed
	{
	protected:

		/** Pointer to memory on CPU host. */
		DEVICEPTR(T)* data_cpu;

		/** Pointer to memory on GPU, if available. */
		DEVICEPTR(T)* data_cuda;

        size_t _dataSize;
    public: /* private */
        // true when cpu memory has been possibly modified but gpu not updated
        mutable bool dirtyCPU;
        mutable bool dirtyGPU;
    protected:

        
        /** Transfer data from GPU to CPU. Happens implicitly when the pointer is requested */
        void UpdateHostFromDevice() const {
            assert(dirtyGPU && !dirtyCPU);
            cudaSafeCall(cudaMemcpy(data_cpu, data_cuda, dataSizeInBytes(), cudaMemcpyDeviceToHost));
            dirtyGPU = false;
        }

    public:
        /** Transfer data from CPU to GPU. Cannot be requested from GPU */
        void UpdateDeviceFromHost() const {
            assert(!dirtyGPU && dirtyCPU);
            cudaSafeCall(cudaMemcpy(data_cuda, data_cpu, dataSizeInBytes(), cudaMemcpyHostToDevice));
            dirtyCPU = false;
        }

        CPU_AND_GPU size_t getDataSize() const { return _dataSize; } 
        /** Total #number of allocated entries in the data array. Read-only.
        Number of allocated BYTES computes as dataSize * sizeof(T)*/
        __declspec(property(get = getDataSize)) size_t dataSize;

        CPU_AND_GPU size_t dataSizeInBytes() const {
            return dataSize * sizeof(T);
        }
		enum MemoryCopyDirection { CPU_TO_CPU, CPU_TO_CUDA, CUDA_TO_CPU, CUDA_TO_CUDA };

		/** Get the data pointer on CPU or GPU. */
		CPU_AND_GPU DEVICEPTR(T)* GetData(MemoryDeviceType memoryType)
		{
			switch (memoryType)
            {
#ifndef __CUDA_ARCH__
            case MEMORYDEVICE_CPU: if (dirtyGPU) UpdateHostFromDevice(); assert(data_cpu && !dirtyGPU); dirtyCPU = true; return data_cpu;
#endif
            case MEMORYDEVICE_CUDA:
#ifndef __CUDA_ARCH__
                if (dirtyCPU) UpdateDeviceFromHost();
#endif
                assert(data_cuda && !dirtyCPU); 
                dirtyGPU = true; // TODO calling this function often will cause many unnecessary memory transfers
                return data_cuda;
			}
            assert(false);
            return 0;
        }

        /** Get the const data pointer on CPU or GPU. */
        CPU_AND_GPU const DEVICEPTR(T)* GetData(MemoryDeviceType memoryType) const
        {
            switch (memoryType)
            {
#ifndef __CUDA_ARCH__
            case MEMORYDEVICE_CPU: if (dirtyGPU) UpdateHostFromDevice(); assert(data_cpu && !dirtyGPU); return data_cpu;
#endif
            case MEMORYDEVICE_CUDA:
#ifndef __CUDA_ARCH__
                if (dirtyCPU) UpdateDeviceFromHost();
#endif
                assert(data_cuda && !dirtyCPU); // did you forget to UpdateDeviceFromHost?
                return data_cuda;
            }
            assert(false);
            return 0;
        }

#ifdef __CUDA_ARCH__
        /** Get the data pointer on CPU or GPU. */
        GPU_ONLY DEVICEPTR(T)* GetData() {return GetData(MEMORYDEVICE_CUDA);}
        GPU_ONLY const DEVICEPTR(T)* GetData() const { return GetData(MEMORYDEVICE_CUDA); }
#else
        inline DEVICEPTR(T)* GetData() { return GetData(MEMORYDEVICE_CPU); }
        inline const DEVICEPTR(T)* GetData() const { return GetData(MEMORYDEVICE_CPU); }
#endif

		/** Initialize an empty memory block of the given size,
		on CPU only or GPU only or on both. CPU might also use the
		Metal compatible allocator (i.e. with 16384 alignment).
		*/
        MemoryBlock(size_t dataSize) : data_cpu(0), data_cuda(0)
		{
			Allocate(dataSize);
			Clear();
		}

		/** Set all data to the given @p defaultValue. */
		void Clear(unsigned char defaultValue = 0)
		{
            memset(data_cpu, defaultValue, dataSizeInBytes());
            cudaSafeCall(cudaMemset(data_cuda, defaultValue, dataSizeInBytes()));
            dirtyCPU = dirtyGPU = false;
		}

		/** Copy data */
		void SetFrom(const MemoryBlock<T> *source, MemoryCopyDirection memoryCopyDirection)
		{
            assert(dataSize == source->dataSize);
            assert(dataSizeInBytes() == source->dataSizeInBytes());
			switch (memoryCopyDirection)
			{
			case CPU_TO_CPU:
                memcpy(GetData(MEMORYDEVICE_CPU), source->GetData(MEMORYDEVICE_CPU), dataSizeInBytes());
				break;

			case CPU_TO_CUDA:
                cudaSafeCall(cudaMemcpyAsync(GetData(MEMORYDEVICE_CUDA), source->GetData(MEMORYDEVICE_CPU), dataSizeInBytes(), cudaMemcpyHostToDevice));
				break;
			case CUDA_TO_CPU:
                cudaSafeCall(cudaMemcpy(GetData(MEMORYDEVICE_CPU), source->GetData(MEMORYDEVICE_CUDA), dataSizeInBytes(), cudaMemcpyDeviceToHost));
				break;
			case CUDA_TO_CUDA:
                cudaSafeCall(cudaMemcpyAsync(GetData(MEMORYDEVICE_CUDA), source->GetData(MEMORYDEVICE_CUDA), dataSizeInBytes(), cudaMemcpyDeviceToDevice));
				break;

			default: break;
			}
		}

		virtual ~MemoryBlock() { this->Free(); }

		/** Allocate image data of the specified size. If the
		data has been allocated before, the data is freed.
		*/
		void Allocate(size_t dataSize //!< 0 is not acceptable
            )
		{
            assert(dataSize);
            Free();
            this->_dataSize = dataSize;
            data_cpu = new T[dataSize];
            cudaSafeCall(cudaMalloc(&data_cuda, dataSizeInBytes()));
            dirtyCPU = dirtyGPU = true;
		}

		void Free()
		{
            delete[] data_cpu;
            data_cpu = 0;

            cudaSafeCall(cudaFree(data_cuda));
            data_cuda = 0;
		}
	};
}
