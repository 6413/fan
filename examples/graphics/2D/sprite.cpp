#include fan_pch

int main() {
  loco_t loco;

  loco_t::image_t image;
  image.load("images/tire.webp");
  std::vector<loco_t::shape_t> sprites;
  for (int i = 0; i < 254; ++i) {
    sprites.push_back(fan::graphics::sprite_t{ {
    .position = fan::vec3(fan::random::vec2(0, 1600), 254 - i),
    .size = 50,
    .image = &image
    } });
  }
  loco.set_vsync(false);

  f32_t angle = 0;
  loco.loop([&] {

    for (int i = 0; i < 254; ++i) {
      sprites[i].set_angle(fan::vec3(0, 0, angle));
    }
   // sprite.set_position(loco.get_mouse_position());
   // sprite.set_angle(fan::vec3(angle, 0, 0));
    angle += gloco->delta_time;
    loco.get_fps();
  });

  return 0;
}