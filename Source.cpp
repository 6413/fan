#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(io/file.h)

struct a_t {
  fan::vec2 p;
  double x;
};

template <typename T, typename T2>
void set(a_t* a, T a_t::* member, T2 v) {
  a->*member = v;
}
template <typename T>
void set2(a_t* a, T a_t::* member, T v) {
  a->*member = v;
}

int main() {
  a_t x;
  set(&x, &a_t::x, 3);
} 