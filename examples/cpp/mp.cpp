#include fan_pch

struct temp_t {
  int x;
};

struct a_t {
  int x;
  float y;
  temp_t c;
};

struct b_t {
  a_t a;
  int* b;
  int c[2];
};

struct recursive0_t {
  b_t b;
};

struct recursive1_t {
  recursive0_t r;
};

struct recursive2_t {
  recursive1_t r;
};

struct iterate_a_t {
  int x;
};

struct iterate_b_t {
  float y;
};

struct iterate_c_t {
  iterate_a_t a;
  iterate_b_t b;
};


struct random_t {
  fan::vec2 size;
  fan::string id;
  uint32_t group_id;
};

struct color_t {
  f32_t salsa;
};

struct colors_t {
  color_t color;
};

int main() {
  fan::print(recursive2_t{});
}