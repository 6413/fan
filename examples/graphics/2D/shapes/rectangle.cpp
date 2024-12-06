#include <fan/pch.h>

int main() {

  loco_t loco;
  loco_t::shape_t rect = fan::graphics::rectangle_t{ {
    .position = fan::vec3(fan::random::vec2(0, 600), 0),
    .size = 50,
    .color = fan::colors::red,
  } };

  loco.loop([&] {

  });
}