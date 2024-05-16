#include <fan/pch.h>

f32_t f(f32_t i, f32_t j) {
  return 17 * i - 14 * j;
}

int main() {
  loco_t loco;

  fan::vec2 window_size = loco.window.get_size();

  fan::vec2 v = f(1, 1);

  // draw line from top left to bottom right
  fan::graphics::line_t line{{
    .src = fan::vec3(0, 0, 0),
    .dst = window_size,
    .color = fan::colors::white
  }};

  loco.loop([&] {

  });

  return 0;
}