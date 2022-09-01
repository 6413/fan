#include <fan/types/types.h>

struct a_t{
  void f() {
    fan::print(sizeof(*this));
  }
};

struct b_t : a_t {
  void f2() {
    fan::print(sizeof(*this));
  }
};

int main() {
  b_t b;
  b.f2();
}