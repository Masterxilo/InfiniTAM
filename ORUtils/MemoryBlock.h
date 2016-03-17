// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "PlatformIndependence.h"
#include "CUDADefines.h"


enum MemoryDeviceType { MEMORYDEVICE_CPU, MEMORYDEVICE_CUDA };

namespace ORUtils
{
	/** \brief
	Represents memory blocks, templated on the data type
	*/
	template <typename T>
	class MemoryBlock
	{
	protected:

        bool isAllocated_CPU() const {
            return data_cpu != 0;
        }
        bool isAllocated_CUDA() const {
            return data_cuda != 0;
        }
		/** Pointer to memory on CPU host. */
		DEVICEPTR(T)* data_cpu;

		/** Pointer to memory on GPU, if available. */
		DEVICEPTR(T)* data_cuda;

	public:
		enum MemoryCopyDirection { CPU_TO_CPU, CPU_TO_CUDA, CUDA_TO_CPU, CUDA_TO_CUDA };

		/** Total number of allocated entries in the data array. */
		size_t dataSize;

		/** Get the data pointer on CPU or GPU. */
		inline DEVICEPTR(T)* GetData(MemoryDeviceType memoryType)
		{
			switch (memoryType)
			{
			case MEMORYDEVICE_CPU: return data_cpu;
			case MEMORYDEVICE_CUDA: return data_cuda;
			}
            assert(false);
            return 0;
		}

		/** Get the data pointer on CPU or GPU. */
		inline const DEVICEPTR(T)* GetData(MemoryDeviceType memoryType) const
		{
			switch (memoryType)
			{
			case MEMORYDEVICE_CPU: return data_cpu;
			case MEMORYDEVICE_CUDA: return data_cuda;
            }
            assert(false);
            return 0;
		}

		/** Initialize an empty memory block of the given size,
		on CPU only or GPU only or on both. CPU might also use the
		Metal compatible allocator (i.e. with 16384 alignment).
		*/
        MemoryBlock(size_t dataSize, bool allocate_CPU, bool allocate_CUDA) : data_cpu(0), data_cuda(0)
		{
			Allocate(dataSize, allocate_CPU, allocate_CUDA);
			Clear();
		}

		/** Initialize an empty memory block of the given size, either
		on CPU only or on GPU only.
		*/
        MemoryBlock(size_t dataSize, MemoryDeviceType memoryType) : data_cpu(0), data_cuda(0)
		{
			switch (memoryType)
			{
			case MEMORYDEVICE_CPU: Allocate(dataSize, true, false); break;
			case MEMORYDEVICE_CUDA: Allocate(dataSize, false, true); break;
            default: assert(false); break;
			}

			Clear();
		}

		/** Set all image data to the given @p defaultValue. */
		void Clear(unsigned char defaultValue = 0)
		{
			if (isAllocated_CPU()) memset(data_cpu, defaultValue, dataSize * sizeof(T));
			if (isAllocated_CUDA()) ORcudaSafeCall(cudaMemset(data_cuda, defaultValue, dataSize * sizeof(T)));
		}

		/** Transfer data from CPU to GPU, if possible. */
		void UpdateDeviceFromHost() const {
			if (isAllocated_CUDA() && isAllocated_CPU())
				ORcudaSafeCall(cudaMemcpy(data_cuda, data_cpu, dataSize * sizeof(T), cudaMemcpyHostToDevice));
		}
		/** Transfer data from GPU to CPU, if possible. */
		void UpdateHostFromDevice() const {
            if (isAllocated_CUDA() && isAllocated_CPU())
				ORcudaSafeCall(cudaMemcpy(data_cpu, data_cuda, dataSize * sizeof(T), cudaMemcpyDeviceToHost));
		}

		/** Copy data */
		void SetFrom(const MemoryBlock<T> *source, MemoryCopyDirection memoryCopyDirection)
		{
			switch (memoryCopyDirection)
			{
			case CPU_TO_CPU:
				memcpy(this->data_cpu, source->data_cpu, source->dataSize * sizeof(T));
				break;

			case CPU_TO_CUDA:
				ORcudaSafeCall(cudaMemcpyAsync(this->data_cuda, source->data_cpu, source->dataSize * sizeof(T), cudaMemcpyHostToDevice));
				break;
			case CUDA_TO_CPU:
				ORcudaSafeCall(cudaMemcpy(this->data_cpu, source->data_cuda, source->dataSize * sizeof(T), cudaMemcpyDeviceToHost));
				break;
			case CUDA_TO_CUDA:
				ORcudaSafeCall(cudaMemcpyAsync(this->data_cuda, source->data_cuda, source->dataSize * sizeof(T), cudaMemcpyDeviceToDevice));
				break;

			default: break;
			}
		}

		virtual ~MemoryBlock() { this->Free(); }

		/** Allocate image data of the specified size. If the
		data has been allocated before, the data is freed.
		*/
		void Allocate(size_t dataSize, //!< 0 is not acceptable
            bool allocate_CPU, bool allocate_CUDA)
		{
            assert(dataSize);
            assert(allocate_CPU || allocate_CUDA);
			Free();
			this->dataSize = dataSize;
			if (allocate_CPU) data_cpu = new T[dataSize];
			if (allocate_CUDA) ORcudaSafeCall(cudaMalloc((void**)&data_cuda, dataSize * sizeof(T)));
		}

		void Free()
		{
			if (isAllocated_CPU()) {
                delete[] data_cpu;
                data_cpu = 0;
			}

			if (isAllocated_CUDA()) {
                ORcudaSafeCall(cudaFree(data_cuda));
                data_cuda = 0;
			}
		}
	};
}
