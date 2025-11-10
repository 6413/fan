#include <fan/utility.h>
import fan;

#include <fan/graphics/types.h>

int main() {
  fan::graphics::engine_t engine;

  fan::vec2 viewport_size = engine.window.get_size();

  static constexpr fan::vec2 grid_size = fan::vec2(64, 64);
  fan::graphics::tilemap_t tilemap(
    grid_size,
    fan::colors::red,
    viewport_size,
    { 0, 0 }
  );

  fan::graphics::circle_t circle{{
    .position  = fan::vec3(0, 0, 3),
    .radius    = 128,
    .color     = fan::colors::blue - fan::color(0, 0, 0, 0.3),
    .blending  = true
  }};

  engine.set_vsync(false);

  fan::graphics::line_t line;

  fan_window_loop {
    tilemap.reset_colors(fan::colors::red);

    fan::vec2 world_pos = engine.get_mouse_position();
    circle.set_position(world_pos);

    tilemap.highlight(circle, fan::colors::green);
    tilemap.highlight(line, fan::colors::blue);
  };
}