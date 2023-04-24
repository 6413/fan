#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include <fan/types/vector.h>

struct Config {
  static constexpr bool hasX = false;
  static constexpr bool hasY = false;
  static constexpr bool hasZ = false;
};

struct make_x {
  int x;
};
struct empty_t {

};

template <typename C>
struct bll_maker  {
  //int salsa;

  template <typename T = C, typename = std::enable_if_t<T::hasY >>
  void f(){
    // Function body
    salsa = 5;
  }

  using temp = C;
};

struct ConfigB : Config {
  static constexpr bool hasY = false;
};
using B_t = bll_maker<ConfigB>;

int main() {
  B_t b;
  b.f();
}