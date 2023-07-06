#include <stdio.h>

#define _CONCAT(_0_m, _1_m) _0_m ## _1_m
#define CONCAT(_0_m, _1_m) _CONCAT(_0_m, _1_m)

#define vector_name vec2
#define vector_size 2
#define element_type int
#include "vector.h"

#define vector_name vec3
#define vector_size 3
#include "vector.h"

struct vec4 {
  static constexpr int size = 4;
  float r[4];
};

void vec_multiply_size(float* v0, float* v1, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    //...
  }
}

#define vec_multiply(v0, v1) vec_multiply_size(v0, v1, sizeof(v0) / sizeof(*v0))

int main() {
  
  float vec2_0[4];
  float vec2_1[4];
  vec_multiply(vec2_0, vec2_1);

}

