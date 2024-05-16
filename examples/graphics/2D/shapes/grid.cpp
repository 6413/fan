#include <fan/pch.h>

int main() {
  loco_t loco;

  fan::graphics::grid_t grid{ {
      .position = fan::vec3(400, 400, 0),
      .size = fan::vec2(400, 400),
      .grid_size = 32,
      .color = fan::colors::white
    } };

  loco.loop([&] {
    loco.get_fps();
  });

  return 0;
}