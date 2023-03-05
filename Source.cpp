
#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include <iostream>
#include <array>

struct element_t {
  int x;
};

template <typename T>
struct get_type {
  using type_t = T;
};

struct b_t {
  struct g_t : fan::return_type_of_t<decltype([] {
      struct {
        auto f() {
          return get_type<decltype(*this)>();
        }
      }v;
      return get_type<decltype(v)>();
    }) > {
  };
};

using type_t = b_t::g_t::type_t;

struct a_t {
  std::array<element_t, 5> x = [&] {
    fan::print(&x[0]);
    x[0].x = 5;
    return x;
  }();

};

int main() {
  a_t a;
  fan::print(&a.x[0]);;
  //b_t::get_type<>::type_t;
  //b_t::
}