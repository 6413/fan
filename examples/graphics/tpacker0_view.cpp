#include fan_pch

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  void open() {
    loco.open_camera(
      &camera,
      ortho_x,
      ortho_y
    );
    viewport.open();
    viewport.set(0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
  loco_t loco;
};

int main() {

  pile_t pile;
  pile.open();

  loco_t::texturepack_t texturepack;
  texturepack.open_compiled("TexturePack");

  loco_t::shapes_t::sprite_t::properties_t p;

  loco_t::texturepack_t::ti_t ti;
  if (texturepack.qti("entity_ship", &ti)) {
    return 1;
  }
  p.load_tp(&ti);

  p.position = 0;
  p.camera = &pile.camera;
  p.viewport = &pile.viewport;
  p.size = 0.5;
  p.position = 0;
  loco_t::cid_nr_t cid;
  cid.init();
  pile.loco.sprite.push_back(cid, p);
  
  pile.loco.loop([&] {

   });

  return 0;
}