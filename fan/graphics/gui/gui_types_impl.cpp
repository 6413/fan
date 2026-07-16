#if defined(FAN_GUI)
module fan.graphics.gui.types;

import std;

namespace fan::graphics::gui {
  void topmost_window_data_t::register_window(std::string_view name) {
    if (std::find(windows.begin(), windows.end(), name) == windows.end()) {
      windows.push_back(std::string(name));
    }
  }

  void topmost_window_data_t::unregister_window(std::string_view name) {
    auto it = std::find(windows.begin(), windows.end(), name);
    if (it != windows.end()) {
      windows.erase(it);
    }
  }
}

#endif