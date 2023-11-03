#include fan_pch

int main() {
  loco_t loco;

  fan::vec2 window_size = loco.get_window()->get_size();
  loco.default_camera->camera.set_ortho(
    fan::vec2(0, window_size.x),
    fan::vec2(0, window_size.y)
  );

  loco_t::image_t image;
  image.load("images/2.webp");

  fan::graphics::sprite_t sprite{{
    .position = window_size / 2,
    .size = window_size,
    .image = &image
  }};

  loco.set_vsync(false);

  loco.loop([&] {
    loco.get_fps();
    //sprite.set_position(loco.get_mouse_position());
  });

  return 0;
}