#include <iostream>
#include <fan/types/types.h>

struct st_t {
  int a;
  int b;

  template <typename T>
  static constexpr auto AN(T st_t::* x) {
    return offsetOf(x) / sizeof(int);
  }
};

int main() {
  constexpr auto a = fan::ofof(&st_t::b);
  //constexpr auto b = &st_t::b;
  //constexpr auto c = b - a;
  //constexpr auto x = st_t::AN(&st_t::a);
}