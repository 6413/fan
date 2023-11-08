#include fan_pch

#include _FAN_PATH(graphics/gui/tile_map_editor/editor.h)

int main() {
  loco_t loco;

  ftme_t fte;
  fte.open("TexturePack");

  loco.set_vsync(0);
  loco.get_window()->set_max_fps(165);


  loco.loop([&] {

    //loco.get_fps();
  });

  return 0;
}
