#include <iostream>
#include <type_traits>
#include <fan/types/types.h>
#include <fan/types/masterpiece.h>

int main() {
  fan::masterpiece_t<int[]> x;

  x.iterate([&] (auto i) {
    fan::print(0, typeid(x.get_value<i>()).name());
  });
}