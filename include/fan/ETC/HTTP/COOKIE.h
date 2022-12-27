#pragma once

#include _WITCH_PATH(WITCH.h)
#include _WITCH_PATH(VEC/VEC.h)
#include _WITCH_PATH(MEM/MEM.h)

#ifndef HTTP_COOKIE_set_name_size
	#define HTTP_COOKIE_set_name_size 2048
#endif
#ifndef HTTP_COOKIE_set_value_size
	#define HTTP_COOKIE_set_value_size 2048
#endif

enum{
	HTTP_COOKIE_HttpOnly_e = 0x01,
	HTTP_COOKIE_Secure_e = 0x02
};

typedef struct{
	uint8_t name[HTTP_COOKIE_set_name_size];
	uint32_t name_size;
	uint8_t value[HTTP_COOKIE_set_value_size];
	uint32_t value_size;
	uint32_t flag;
}HTTP_Cookie_t;

typedef struct{
	/* HTTP_Cookie_t */
	VEC_t vec;
}HTTP_CookieContainer_t;

void HTTP_CookieContainer_open(HTTP_CookieContainer_t *container){
	VEC_init(&container->vec, sizeof(HTTP_Cookie_t), A_resize);
}
void HTTP_CookieContainer_close(HTTP_CookieContainer_t *container){
	VEC_free(&container->vec);
}

void HTTP_CookieContainer_add0(HTTP_CookieContainer_t *Container, HTTP_Cookie_t *PCookie){
	VEC_handle0(&Container->vec, 1);
	HTTP_Cookie_t *Cookie = &((HTTP_Cookie_t *)Container->vec.ptr)[Container->vec.Current - 1];
	Cookie->name_size = PCookie->name_size;
	Cookie->value_size = PCookie->value_size;
	MEM_copy(PCookie->name, Cookie->name, Cookie->name_size);
	MEM_copy(PCookie->value, Cookie->value, Cookie->value_size);
	Cookie->flag = PCookie->flag;
}

sint32_t HTTP_CookieContainer_add1(HTTP_CookieContainer_t *Container, const void *name, const void *value, uint32_t flag){
	uintptr_t name_size = MEM_cstreu(name);
	if(name_size > HTTP_COOKIE_set_name_size){
		return -1;
	}
	uintptr_t value_size = MEM_cstreu(value);
	if(value_size > HTTP_COOKIE_set_value_size){
		return -2;
	}
	VEC_handle0(&Container->vec, 1);
	HTTP_Cookie_t *Cookie = &((HTTP_Cookie_t *)Container->vec.ptr)[Container->vec.Current - 1];
	Cookie->name_size = name_size;
	Cookie->value_size = value_size;
	MEM_copy(name, Cookie->name, Cookie->name_size);
	MEM_copy(value, Cookie->value, Cookie->value_size);
	Cookie->flag = flag;
}
