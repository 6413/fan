#include <fan/types/types.h>

template <typename T>
struct a_t{
  void f() {
    fan::print(sizeof(T));
  }
};

struct b_t : a_t<b_t> {
  void f2() {
    fan::print(sizeof(*this));
  }
  uint32_t x{1};
};

int main() {
  b_t b;
  b.f2();
}