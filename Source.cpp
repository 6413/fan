#include <type_traits>

#include <iostream>
#include <type_traits>

#include <iostream>
#include <type_traits>

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(types/masterpiece.h)

int main() {

  fan::masterpiece_t<int, double> m;

  m.iterate([](auto x, auto y) {



    if constexpr (x == 1) {
      return 1;
    }
    else {
      return;
    }
  });
}