#include <iostream>
#include <type_traits>
#include <fan/types/types.h>
#include <fan/types/masterpiece.h>
#include <fan/graphics/opengl/gl_core.h>

struct x{
  int a;
  int b;
};

int main() {
  fan::masterpiece_t<int, int> a;
  a.iterate([](const auto& element) {
    fan::print(element);
  });
}