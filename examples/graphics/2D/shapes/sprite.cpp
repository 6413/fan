#include <fan/pch.h>

int main() {
  loco_t loco{{
    .renderer=fan::graphics::engine_t::renderer_t::opengl
  }};
  
  loco_t::image_t image = loco.image_load("images/tire.webp");

  fan::graphics::sprite_t us{ {
    .position = fan::vec3(400, 400, 254),
    .size = 100,
    .image = image,
  } };

  fan::graphics::rectangle_t rect{ {
    .position = fan::vec3(200, 200, 0),
    .size = 35,
    .color = fan::colors::red
  }};

  loco.loop([&] {
 //   rect.set_position(fan::graphics::get_mouse_position());
  });

  return 0;
}