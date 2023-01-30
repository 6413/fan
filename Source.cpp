// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

struct a_t {
  a_t(auto* b) {

  }
};

struct b_t : a_t{

  b_t() : a_t(this) {
    x = new int(5);
    fan::print(*x);
  }

  int* x = 0;
};

int main() {
  b_t b;
  return 0;
}