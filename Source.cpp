#include <iostream>

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(io/file.h)

struct s_t {
  int f = 0;
  s_t() = default;
  s_t(int x) {

  }
  s_t operator=(int x) {
    f = x;
    return *this;
  }
};

struct a_t {
  int x;
  s_t z;
  s_t* y;
};

void push(a_t& a) {
  *a.y = a.z;
}

int main() {

  s_t t = 0;

  a_t v{
    .y = &t
  };

  push(v);
}