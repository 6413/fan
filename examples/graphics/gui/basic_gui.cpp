#include <fan/pch.h>

using namespace fan::graphics;

int main() {
  engine_t engine;

  fan_window_loop{
    fan_graphics_gui_window("window") {
      gui::columns(2);
      fan_graphics_gui_child_window("child_window") {
        gui::button("click me");
        gui::text("hello world");
      }
      gui::next_column();
      fan_graphics_gui_child_window("child_window2") {
        gui::button("click me");
        gui::text("hello world");
      }
      gui::columns(1);
    }
    
  };

  return 0;
}