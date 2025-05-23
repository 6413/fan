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

  mm.open("texture_packs/TexturePack", L"");

  engine.loop([&] {
    mm.render();
  });

  return 0;
}