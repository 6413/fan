#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include <fan/types/vector.h>

int main() {
  fan::vec3 x(0, 1, 2);
  fan::vec3 y = x;
  fan::vec2 v(5, 6);
  x = std::move(v);
  fan::print(x);
}