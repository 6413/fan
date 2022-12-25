#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define d \
  int x; \

struct t_t {
  int y;
};

struct s_t : std::conditional_t<d{
  d;
  struct {
    int x;
  };
};

int main() {
  s_t s;
  s.x = 5;
}