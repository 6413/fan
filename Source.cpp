#include <iostream>
#include <fan/types/types.h>

struct st_t {
  int a;

  template <typename T>
  static constexpr auto AN(T st_t::* x) {
    return fan::ofof(x) / sizeof(int);
  }
};

int main() {
  constexpr auto x = st_t::AN(&st_t::a);
}