module;

#if defined(FAN_GUI)

#include <cstdint>

#endif

export module fan.graphics.gui.input;

#if defined(FAN_GUI)

import fan.types.vector;
import fan.window.input_common;
import fan.graphics.gui.base;

using namespace fan::graphics;

export namespace fan::graphics::gui::input {
  bool ctrl() {
    return gui::get_io().KeyCtrl;
  }
  bool shift() {
    return gui::get_io().KeyShift;
  }
  bool alt() {
    return gui::get_io().KeyAlt;
  }
  bool left_click() {
    return gui::get_io().MouseClicked[fan::mouse_left];
  }
  bool left_down() {
    return gui::get_io().MouseDown[fan::mouse_left];
  }
  bool left_released() {
    return gui::get_io().MouseReleased[fan::mouse_left];
  }
  bool right_click() {
    return gui::get_io().MouseClicked[fan::mouse_right];
  }
  bool right_down() {
    return gui::get_io().MouseDown[fan::mouse_right];
  }
  bool right_released() {
    return gui::get_io().MouseReleased[fan::mouse_right];
  }
  f32_t scroll() {
    return gui::get_io().MouseWheel;
  }
  bool number(uint32_t& out_number) {
    for (uint32_t i = 0; i < 9; ++i) {
      if (gui::is_key_pressed((gui::key_t)fan::window::input::to_imgui_key(fan::key_1 + i), false)) {
        out_number = i;
        return true;
      }
    }
    return false;
  }
  bool hover(const fan::vec2& p_min, const fan::vec2& p_max) {
    fan::vec2 m = gui::get_io().MousePos;
    return m.x >= p_min.x && m.x <= p_max.x &&
      m.y >= p_min.y && m.y <= p_max.y;
  }
  bool drag_start(bool drag_active, bool clicked, bool slot_empty) {
    return !drag_active && clicked && !slot_empty;
  }
}

#endif