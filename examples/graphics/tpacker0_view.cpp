#include fan_pch

int main() {
  loco_t loco;
  loco.default_camera->camera.set_ortho(
    fan::vec2(-1, 1),
    fan::vec2(-1, 1)
  );

  loco_t::texturepack_t texturepack;
  texturepack.open_compiled("1");

  loco_t::shapes_t::sprite_t::properties_t p;

  loco_t::texturepack_t::ti_t ti;
  if (texturepack.qti("tile1", &ti)) {
    return 1;
  }
  p.load_tp(&ti);
  //p.image = &texturepack.pixel_data_list[0].image;

  p.position = 0;
  p.size = 0.5;
  p.position = 0;
  loco_t::shape_t sprite = p;
  
  loco.loop([&] {

  });

  return 0;
}