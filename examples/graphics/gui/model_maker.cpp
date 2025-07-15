#include <fan/types/types.h>
#include <fan/math/math.h>

#include <fan/imgui/imgui.h>

import fan;

#include <fan/graphics/gui/model_maker/maker.h>

int main(int argc, char** argv) {
  if (argc < 2) {//
    fan::throw_error("usage: TexturePackCompiled");
  }////
  //////////
  //////////
  fan::graphics::engine_t engine;////

  model_maker_t mm;////

  mm.open("examples/games/forest game/forest_tileset.ftp", L"");
 // mm.fin("normal_map_tests.json");

  engine.loop([&] {
    mm.render();
  });

  return 0;
}