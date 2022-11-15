#include <iostream>
#include <algorithm>
#include <vector>
#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 1
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include <fan/types/vector.h>
#include <fan/math/math.h>

struct b_t {

};

struct a_t {
  a_t(b_t ){}
};

int main() {
  a_t x{0};
}