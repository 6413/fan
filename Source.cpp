#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(types/vector.h)

void f() {

}

struct HelloWorld {
  void main() {
    f();
  }
};

struct b_t {
  int y;
  int c;
};

struct a_t {
  b_t b;
};

struct c_t {
  int x;
  b_t v;

  c_t& operator=(c_t&& c) {
    fan::print("a");
    return c;
  }
  ~c_t() {
    fan::print("d");
  }
};

int main() {
  b_t b;
  b.c = 5;
  c_t c;
  c = {
    .x = 0, 
    .v = b
  };
}