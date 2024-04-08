#pragma once

#include <cstdint>

#if defined(loco_imgui)
#include <fan/imgui/imgui.h>
#endif

namespace fan {
  namespace window_input {
    uint16_t convert_scancode_to_fan(int key);

    uint16_t convert_fan_to_scancode(int key);

    #if defined(loco_imgui)
    ImGuiKey fan_to_imguikey(int key);
    #endif
  }
}