#include fan_pch


int main() {
  loco_t loco;
  loco.default_camera->camera.set_ortho(
    fan::vec2(-1, 1),
    fan::vec2(-1, 1)
  );

  fan::graphics::circle_t circle{{
    .position = fan::vec2(0, 0),
    .radius = 0.5,
    .color = fan::colors::white,
    .blending = true
  }};

  loco.loop([&] {
    loco.get_fps();
  });

  return 0;
}