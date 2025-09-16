#pragma once

struct fan_window_loop_t{
  fan_window_loop_t(const auto& lambda) {
    gloco->loop(lambda);
  }
};

// static called inside scope, so its fine for linking
#define fan_window_loop \
  static fan_window_loop_t __fan_window_loop_entry = [&]()

#define fan_window_close() \
  gloco->close(); \
  return

//const std::string& window_name, bool* p_open = 0, window_flags_t window_flags = 0
#define fan_graphics_gui_window(...) \
    for (struct { \
        fan::graphics::gui::window_t __window; \
        int once; \
      }__struct_var{{__VA_ARGS__}, {(bool)__struct_var.__window}}; \
      __struct_var.once--;  \
    )

//(const std::string& window_name, const fan::vec2& size = fan::vec2(0, 0), child_window_flags_t window_flags = 0)
#define fan_graphics_gui_child_window(...) \
    for (struct { \
        fan::graphics::gui::child_window_t __window; \
        int once; \
      }__struct_var{{__VA_ARGS__}, {(bool)__struct_var.__window}}; \
      __struct_var.once--;  \
    )

#define fan_graphics_gui_table(...) \
    for (struct { \
        fan::graphics::gui::table_t __table; \
        int once; \
      }__struct_var{{__VA_ARGS__}, {(bool)__struct_var.__table}}; \
      __struct_var.once--;  \
    )

// GUI
#define fan_graphics_gui_slider_property(shape, kind) \
  [&]{ \
    static auto val = shape.get_##kind(); \
    bool ret = fan::graphics::gui::drag_float(#shape "_" #kind "_slider", &val); \
    if (ret) shape.set_##kind(val);  \
    return ret; \
  }()