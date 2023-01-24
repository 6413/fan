#include <iostream>

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)


struct stages_t {
  struct wrapper_t;

  inline static constexpr stages_t* get(auto* ptr) {
    return OFFSETLESS((wrapper_t*)ptr, stages_t, wrapper);
  }

  struct stage0_t {
    stage0_t() {
      fan::print(get(this));
    }
    uint8_t x[10];
  };
  struct stage1_t {
    stage1_t() {
      fan::print(get(this)->wrapper.x);
    }
    uint8_t y[15];
  };
  struct wrapper_t : stage0_t, stage1_t {
  }wrapper;
};

int main() {

  stages_t st;
  fan::print(&st);


  return 0;
}