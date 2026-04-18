#include <string>

import fan;

using namespace fan::graphics;

struct app_t : engine_t {
  app_t() {
    texture_pack.open_compiled("textures.ftp");
    id = renderer.open_map("map.json", {.offset= -64*32});
    //map = shapes_from_json("map.json");
  }

  void loop() {
    engine_t::loop([&]{
    //  renderer.update(id, {64, 64});
    });
  }
  interactive_camera_t ic;
  tilemap_renderer_t renderer;
  tilemap_renderer_t::id_t id;
};

int main() {
  app_t app;

  app.loop();
}