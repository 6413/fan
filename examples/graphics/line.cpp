#include fan_pch

int main() {
  loco_t loco;

  fan::vec2 window_size = loco.get_window()->get_size();
  loco.default_camera->camera.set_ortho(
    fan::vec2(0, window_size.x),
    fan::vec2(0, window_size.y)
  );

  // draw line from top left to bottom right
  fan::graphics::line_t line{{
    .src = fan::vec2(0, 0),
    .dst = fan::vec2(0, window_size.y),
    .color = fan::colors::white
  }};

  
  loco.loop([&] {

  });

  return 0;
}