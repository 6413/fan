#include fan_pch

#include _FAN_PATH(graphics/gui/tilemap_editor/editor.h)

int main() {
  loco_t loco;

  fte_t fte;
  fte.open("texture_packs/tilemap.ftp");
  fte.fin("tilemaps/tilemap_demo.fte");

  loco.set_vsync(0);
  //loco.get_window()->set_max_fps(165);
  loco.loop([&] {

    //loco.get_fps();
  });

  return 0;
}
