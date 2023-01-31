// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)


// how to define loco_t::a_t here

void f(a_t*) {

}

struct b_t {
  loco_t::a_t* a;
};

struct loco_t {
  struct a_t {

  }a;
};

int main() {
  loco_t l;
  f(&l.a);
  return 0;
}