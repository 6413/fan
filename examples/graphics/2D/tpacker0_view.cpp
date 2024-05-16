#include <fan/pch.h>

int main() {
  loco_t loco;
  loco.camera_set_ortho(
    loco.orthographic_camera.camera,
    fan::vec2(-1, 1),
    fan::vec2(-1, 1)
  );

  loco_t::texturepack_t texturepack;
  texturepack.open_compiled("texture_packs/TexturePack");

  loco_t::sprite_t::properties_t p;

  loco_t::texturepack_t::ti_t ti;
 /* if (texturepack.qti("gui/fuel_station/fuel_icon", &ti)) {
    return 1;
  }
  p.load_tp(&ti);*/
  p.image = texturepack.pixel_data_list[3].image;

  p.position = 0;
  p.size = 1;
  p.position = 0;
  p.blending = true;
  loco_t::shape_t sprite = p;
  
  loco.loop([&] {

  });

  return 0;
}