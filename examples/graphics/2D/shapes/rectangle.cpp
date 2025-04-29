#include <vector>

import fan;

#include <fan/graphics/types.h>


int main() {
  fan::vec2 v;//
  using namespace fan::graphics;
  engine_t engine;//
  
  std::vector<rectangle_t> rects(1);
  for (int i = 0; i < 1; ++i) {
    rects[i] = { {
    .position = fan::vec3(fan::vec2(i, i), 1),
    .size = 100,
    .color = fan::colors::blue,
  } };
  }
  fan_window_loop{
    rects[0].set_color(fan::random::color());
    rects[0].set_position(fan::window::get_mouse_position());
    
  };
}