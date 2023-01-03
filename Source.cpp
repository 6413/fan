#include <iostream>
#include <format>

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include <fmt/format.h>

struct s_t {
  int x[0];
};

int main() {
  fan::string str("hello");
  str.replace(2, 2, "ddd");
  fan::print(str);
}