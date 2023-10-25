#include fan_pch

int main() {
  loco_t loco;

  fan::vec2 window_size = loco.get_window()->get_size();
  fan::graphics::default_camera->camera.set_ortho(
    fan::vec2(0, window_size.x),
    fan::vec2(0, window_size.y)
  );

  loco_t::image_t image;
  image.load("images/2.webp");

  fan::graphics::sprite_t sprite{{
    .position = fan::vec2(400, 400),
    .size = fan::vec2(400, 400),
    .image = &image
  }};

  loco.loop([&] {
    sprite.set_position(loco.get_mouse_position());
  });

  return 0;
}