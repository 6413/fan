#include <fan/pch.h>

int main() {
  using namespace fan::graphics;

  engine_t engine;
  
  image_t image = image_load("images/tire.png");
  image_t image2 = image_load("images/tire_n.png");
  image_t image3 = image_load("images/duck.webp");

  engine.lighting.ambient = 0.7;

  sprite_t us{ {
    .position = fan::vec3(400, 400, 254),
    .size = 256,
    .image = image,
    .images = {image2}
  } };

  light_t light{{
    .position = fan::vec3(400, 400, 0),
    .size = 400,
    .color = fan::colors::white * 10,
  }};

  fan_window_loop{
    light.set_position(get_mouse_position());
  };

  return 0;
}