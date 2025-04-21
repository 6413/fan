#pragma once

#include <cstdint>

namespace fan {
  namespace window_input {
    uint16_t convert_scancode_to_fan(int key);

    uint16_t convert_fan_to_scancode(int key);

    #if defined(fan_gui)
    int fan_to_imguikey(int key);
    #endif
  }
}