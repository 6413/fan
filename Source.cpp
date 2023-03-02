#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <set>

struct a_t {
  struct b_t {

  };
  struct c_t {

  };
};

template <typename T>
void f() {
  typename a_t::T var;
}

int main() {
  f<a_t::b_t>();
  f<a_t::c_t>();
}