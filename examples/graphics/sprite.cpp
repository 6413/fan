#include fan_pch

int main() {
  loco_t loco;

  loco_t::image_t image;
  image.load("images/tire.webp");

  loco_t::shape_t sprite = fan_init_struct(
      loco_t::shapes_t::sprite_t::properties_t,
      .position = fan::vec3(50, 50, 0),
      .size = 50,
      .image = &image
    );

 // fan::print(sprite.get_position());
  loco_t::shape_t s2 = sprite;
  fan::print(s2.get_position());

  loco.set_vsync(false);

  loco.loop([&] {
    sprite.set_position(loco.get_mouse_position());
  //  fan::print(sprite.get_position());
    loco.get_fps();
  });

  return 0;
}