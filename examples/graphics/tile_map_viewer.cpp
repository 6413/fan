#include fan_pch

#include _FAN_PATH(graphics/gui/tile_map_editor/loader.h)

int main() {
  loco_t loco;
  loco_t::texturepack_t tp;
  tp.open_compiled("TexturePack");

  ftme_loader_t loader;
  loader.open(&tp);

  auto compiled_map = loader.compile("map0.ftme");

  ftme_loader_t::properties_t p;

  p.position = fan::vec3(400, 400, 0);
  //p.size = 0.5;

  auto map_id0_t = loader.add(&compiled_map, p);

  loco.loop([&] {

  });
}