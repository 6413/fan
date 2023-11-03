#include fan_pch

int main() {
  loco_t loco;

  loco_t::image_t image;
  image.load("images/2.webp");

  fan::graphics::sprite_t sprite{{
    .position = 0,
    .size = 0.5,
    .image = &image
  }};

  loco.set_vsync(false);

  loco.loop([&] {
    sprite.set_position(loco.get_mouse_position());
    loco.get_fps();
  });

  return 0;
}