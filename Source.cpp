#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

struct pile_t;

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_opengl

#define loco_window
#define loco_context

#define loco_no_inline

#define loco_sprite
#define loco_button

//#include _FAN_PATH(graphics/loco.h)
#include <fan/types/masterpiece.h>


int main() {
  
  fan::masterpiece_t<int, double> x;
  x.iterate([] (const auto&d, const auto&x) {

    fan::print("a");
  });
}