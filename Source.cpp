#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

struct pile_t;

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

struct a_t {
  int x;
};

struct b_t {

};

struct my_functor {
  a_t& a;
  my_functor(a_t& a) : a(a) {}
  void operator()() {
    a.x = 5;
  }
};

template <typename T>
struct gt {
  T l;
  gt(T l) : l(l) {}
  void f() {
    l();
  }
};

int main() {
  a_t a;
  my_functor f(a);
  gt<my_functor> g(f);
  g.f();
}