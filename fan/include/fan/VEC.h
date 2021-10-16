#pragma once

#include <fan/types/types.hpp>

namespace fan {

#ifndef A_set_buffer
#define A_set_buffer 512
#endif

	static uint64_t _A_calculate_buffer(uint64_t size){
		uint64_t r = A_set_buffer / size;
		if(!r){
			return 1;
		}
		return r;
	}

	typedef struct{
		uint64_t Current, Possible, Type, Buffer;
		std::vector<uint8_t> ptr;
	}VEC_t;
	static void VEC_init(VEC_t *vec, uint64_t size){
		vec->Current = 0;
		vec->Possible = 0;
		vec->Type = size;
		vec->Buffer = _A_calculate_buffer(size);
	}

	static void _VEC_handle(VEC_t *vec){
		vec->Possible = vec->Current + vec->Buffer;
		vec->ptr.resize(vec->Possible * vec->Type);
	}
	static void VEC_handle(VEC_t *vec){
		if(vec->Current >= vec->Possible){
			_VEC_handle(vec);
		}
	}
	static void VEC_handle0(VEC_t *vec, uintptr_t amount){
		vec->Current += amount;
		VEC_handle(vec);
	}

	static void VEC_free(VEC_t *vec){
		vec->ptr.clear();
		vec->Current = 0;
		vec->Possible = 0;
	}
}