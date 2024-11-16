#include <fan/pch.h>

int main() {
  loco_t loco;

  loco_t::image_t image = loco.image_load("images/tire.webp");

  fan::graphics::sprite_t us{ {
    .position = fan::vec3(400, 400, 254),
    .size = 100,
    .rotation_point = fan::vec2(100, 0),
    .image = image,
  } };


  loco.loop([&] {
    
  });

  return 0;
}