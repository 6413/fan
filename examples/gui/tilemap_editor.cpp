#include fan_pch

#include _FAN_PATH(graphics/gui/tilemap_editor/editor.h)

struct a_t {
  int x;
  double y;
};

int main() {
  //
  loco_t loco;
  fte_t fte;//
  fte.file_name = "tilemaps/test.fte";
  fte.open("1");
  fte.fin("tilemaps/test.fte");
  //loco.set_vsync(0);
  //loco.window.set_max_fps(165);
  loco.loop([&] {
    loco.get_fps();
  });
  
  return 0;
}
