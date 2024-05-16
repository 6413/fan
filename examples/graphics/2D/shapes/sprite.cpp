#include <fan/pch.h>

int main() {
  loco_t loco;

  loco_t::image_t image;
  image = loco.image_load("images/tire.webp");
  std::vector<loco_t::shape_t> sprites;
  for (int i = 0; i < 254; ++i) {
    sprites.push_back(fan::graphics::sprite_t{ {
    .position = fan::vec3(fan::random::vec2(0, 1600), 254 - i),
    .size = 50,
    .rotation_point = fan::vec2(100, 0),
    .image = image,
    } });
  }

  fan::graphics::sprite_t us{ {
    .position = fan::vec3(400, 400, 254),
    .size = 100,
    .rotation_point = fan::vec2(100, 0),
    .image = image,
    } };

  loco_t::light_t::properties_t lp;
  lp.position = fan::vec3(400, 400, 0);
  lp.size = 100;
  lp.color = fan::colors::yellow * 10;
  // {
  loco_t::shape_t l0 = lp;

  loco.set_vsync(false);

  loco.lighting.ambient = 0.1;

  loco_t::texturepack_t texturepack;
  texturepack.open_compiled("texture_packs/TexturePack");

  loco_t::sprite_t::properties_t p;

  loco_t::texturepack_t::ti_t ti;
   if (texturepack.qti("gui/fuel_station/fuel_icon", &ti)) {
     return 1;
   }
   //p.load_tp(&ti);

   us.set_tp(&ti);

  f32_t angle = 0;
  loco.loop([&] {

    l0.set_position(loco.get_mouse_position());

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