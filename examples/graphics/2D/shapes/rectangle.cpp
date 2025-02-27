#include <fan/pch.h>

//fan_track_allocations();

int main() {

  loco_t loco;
  loco_t::shape_t rect = fan::graphics::rectangle_t{ {
    .position = fan::vec3(fan::vec2(400, 400), 0),
    .size = 200,
    .color = fan::colors::red,
  } };

  loco.loop([&] {

  });
}