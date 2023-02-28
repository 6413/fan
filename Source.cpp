#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(types/masterpiece.h)

#include <iostream>
#include <stdio.h>

struct a_t {
  int x;
};

struct b_t : a_t {
  int y;
};

int main() {
  b_t b = { {.x = 5} };
}
