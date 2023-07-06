#ifndef element_type
  #define element_type float
#endif

struct vector_name {
  element_type arr[vector_size];
};

#define build_multiply \
struct vector_name CONCAT(CONCAT(vector_name, _), multiply)(struct vector_name* v0, struct vector_name* v1) { \
  struct vector_name ret; \
  for (int i = 0; i < vector_size; ++i) { \
    ((element_type*)&ret)[i] = ((element_type*)v0)[i] * ((element_type*)v1)[i]; \
  } \
  return ret; \
}

build_multiply

#undef build_multiply
#undef vector_name
#undef vector_size
#undef element_type