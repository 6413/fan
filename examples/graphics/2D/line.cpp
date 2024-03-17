#include fan_pch

f32_t f(f32_t i, f32_t j) {
  return 17 * i - 14 * j;
}

int main() {
  loco_t loco;

  fan::vec2 window_size = loco.window.get_size();
  loco.default_camera->camera.set_ortho(
    fan::vec2(0, window_size.x),
    fan::vec2(0, window_size.y)
  );

  fan::vec2 v = f(14, 15);

  // draw line from top left to bottom right
  fan::graphics::line_t line{{
    .src = v,
    .dst = v * 100,
    .color = fan::colors::white
  }};

  loco.loop([&] {

  });

  return 0;
}