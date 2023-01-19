// rectangle text button using loco

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan

#include _FAN_PATH(system.h)

#define loco_framebuffer

#define loco_window
#define loco_context

#include <string_view>
#include <clocale>
#include <cuchar>


int main() {
  fan::sys::set_utf8_cout();
  std::u32string utf32_string = U"hello ö";
  std::string utf8_string;

  auto [p, ec] = std::to_chars(std::back_inserter(utf8_string), utf32_string.data(), utf32_string.data() + utf32_string.size(), std::chars_format::utf8);
  if (ec == std::errc()) {
    std::cout << utf8_string;
  }
}