#include <iostream>
#include <type_traits>
#include <fan/types/types.h>
#include <fan/types/masterpiece.h>
#include <fan/graphics/opengl/gl_core.h>

int main() {
  uint32_t x = 0;
  auto l = [&] {
    fan::print(x);
  };
  x = 5;
  l();
}