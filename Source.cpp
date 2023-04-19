#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

struct pile_t;

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

struct a_t {
  a_t() {
    fan::print("a_t()");
  }
  a_t(const a_t&) {
    fan::print("copy");
  }
  a_t(a_t&&) {
    fan::print("move");
  }
  a_t& operator=(const a_t&) {
    fan::print("acopy");
    return *this;
  }
  a_t& operator=(a_t&&) {
    fan::print("amove");
    return *this;
  }
};

int main() {
  //std::vector<a_t> a;
  a_t a;
  a_t* b = (a_t*)malloc(sizeof(a_t));
  new (b) a_t(a);
  //std::construct_at(b, a);
  //new (b) a_t(a);
  //a = b;
  //a = std::move(b);
  /*for (uint32_t i = 0; i < 2; ++i) {
    a.push_back(a_t());
  }*/
}