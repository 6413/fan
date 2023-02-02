// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(types/vector.h)

//#define loco_vulkan

#define loco_window
#define loco_context

int main() {

  auto min = fan::min(fan::vec2(0, 1), fan::vec2(2, 3));

  return 0;
}