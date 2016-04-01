// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "MemoryBlock.h"


namespace ORUtils
{
	/** \brief
	Represents images, templated on the pixel type

    Managed


	*/
	template <typename T>
	class Image : public MemoryBlock < T >
	{
	public:
		/** Size of the image in pixels. */
		Vector2<int> noDims;

		/** Initialize an empty image of the given size
		*/
        Image(Vector2<int> noDims = Vector2<int>(1, 1)) : MemoryBlock<T>(noDims.area()), noDims(noDims) {}

		/** Resize an image, loosing all old image data.
		Essentially any previously allocated data is
		released, new memory is allocated.
        No-op if image already has this size
		*/
		void ChangeDims(Vector2<int> newDims)
		{
            if (newDims == noDims) return;
			this->noDims = newDims;
			this->Free();
            this->Allocate(newDims.area());
            Clear();
		}

        /** Copy data, resize if needed */
        void SetFrom(const Image<T> *source, MemoryCopyDirection memoryCopyDirection)
        {
            ChangeDims(source->noDims);
            __super::SetFrom(source, memoryCopyDirection);
        }
	};
}
