// rectangle text button using loco

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan

#include _FAN_PATH(system.h)

int main() {
  fan::sys::set_utf8_cout();
  fan::string str = "aö";

  uint32_t x = str.get_utf8(1);
  fan::print_no_space(fan::format("{}{}\n{:x}{:x}",
    (char)((x & 0xff00) >> 8), (char)(x & 0xff), (x & 0xff00) >> 8, x & 0xff
  )
  );
  return 0;
}