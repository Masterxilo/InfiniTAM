// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "MemoryBlock.h"


namespace ORUtils
{
	/** \brief
	Represents images, templated on the pixel type
	*/
	template <typename T>
	class Image : public MemoryBlock < T >
	{
	public:
		/** Size of the image in pixels. */
		Vector2<int> noDims;

		/** Initialize an empty image of the given size, either
		on CPU only or on both CPU and GPU.
		*/
		Image(Vector2<int> noDims, bool allocate_CPU, bool allocate_CUDA)
            : MemoryBlock<T>(noDims.x * noDims.y, allocate_CPU, allocate_CUDA), noDims(noDims)
		{
		}


		Image(bool allocate_CPU, bool allocate_CUDA)
            : Image(Vector2<int>(1, 1), allocate_CPU, allocate_CUDA) {}

		Image(Vector2<int> noDims, MemoryDeviceType memoryType)
            : MemoryBlock<T>(noDims.x * noDims.y, memoryType), noDims(noDims)
		{
		}

		/** Resize an image, loosing all old image data.
		Essentially any previously allocated data is
		released, new memory is allocated.
        No-op if image already has this size
		*/
		void ChangeDims(Vector2<int> newDims)
		{
			if (newDims != noDims)
			{
				this->noDims = newDims;

                bool wasCPU = isAllocated_CPU(), wasCUDA = isAllocated_CUDA();
				this->Free();
                this->Allocate(newDims.x * newDims.y, wasCPU, wasCUDA);
			}
		}
	};
}
