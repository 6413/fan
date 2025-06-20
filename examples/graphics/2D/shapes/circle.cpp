#include <fan/pch.h>




int main() {
  loco_t loco;
  //loco.camera_set_ortho(
  //  loco.orthographic_render_view.camera,
  //  fan::vec2(-1, 1),
  //  fan::vec2(-1, 1)
  //);

  fan::graphics::circle_t circle{ {
    .position = fan::vec2(400, 400),
    .radius = 500,
    .color = fan::colors::white,
    .blending = true
  } };

  loco.loop([&] {
    loco.get_fps();
  });

  return 0;
}