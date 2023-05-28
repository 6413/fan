// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(types/masterpiece.h)

int main() {
  fan::masterpiece_t<int, double, float> v;
  for (uint32_t i = 0; i < v.size(); i++) {
    v.get_value(i, [&] (const auto& d) {
      fan::print(typeid(d).name());
    });
  }
}