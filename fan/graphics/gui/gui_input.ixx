module;

export module fan.graphics.gui.input;

import std;

#if defined(FAN_GUI)

import fan.types;
import fan.types.vector;

export namespace fan::graphics::gui::input {
  bool ctrl();
  bool shift();
  bool alt();
  bool left_click();
  bool left_down();
  bool left_released();
  bool right_click();
  bool right_down();
  bool right_released();
  f32_t scroll();
  bool number(std::uint32_t& out_number);
  bool hover(const fan::vec2& p_min, const fan::vec2& p_max);
  bool drag_start(bool drag_active, bool clicked, bool slot_empty);
}

#endif