#pragma once

#include <fan/types/types.hpp>

namespace fan {

#ifndef A_set_buffer
#define A_set_buffer 512
#endif

	static uint64_t _vector_calculate_buffer(uint64_t size){
		uint64_t r = A_set_buffer / size;
		if(!r){
			return 1;
		}
		return r;
	}

	typedef struct{
		uint64_t Current, Possible, Type, Buffer;
		std::vector<uint8_t> ptr;
	}vector_t;
	static void vector_init(vector_t *vec, uint64_t size){
		vec->Current = 0;
		vec->Possible = 0;
		vec->Type = size;
		vec->Buffer = _vector_calculate_buffer(size);
	}

	static void _vector_handle(vector_t *vec){
		vec->Possible = vec->Current + vec->Buffer;
		vec->ptr.resize(vec->Possible * vec->Type);
	}
	static void VEC_handle(vector_t *vec){
		if(vec->Current >= vec->Possible){
			_vector_handle(vec);
		}
	}
	static void vector_handle0(vector_t *vec, uintptr_t amount){
		vec->Current += amount;
		VEC_handle(vec);
	}

	static void vector_free(vector_t *vec){
		vec->ptr.clear();
		vec->Current = 0;
		vec->Possible = 0;
	}
}