#pragma once

struct fan_window_loop_t{
  fan_window_loop_t(const auto& lambda) {
    gloco()->loop(lambda);
  }
};

// static called inside scope, so its fine for linking
#define fan_window_loop \
  static fan_window_loop_t __fan_window_loop_entry = [&]()

#define fan_window_close() \
  gloco()->close(); \
  return

// GUI
#define fan_graphics_gui_slider_property(shape, kind) \
  [&]{ \
    static auto val = shape.get_##kind(); \
    bool ret = fan::graphics::gui::drag(#shape "_" #kind "_slider", &val); \
    if (ret) shape.set_##kind(val);  \
    return ret; \
  }()