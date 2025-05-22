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
  loco_t loco;////

  model_maker_t mm;////

  mm.open("texture_packs/TexturePack", L"");
  //if(argc == 3){
    //mm.load("ship.json");
  //}
  //else{
  //mm.load("fmm_controller.json");
  //}//
  ////
  //
  loco.set_vsync(false);
  //pile->loco.window.set_max_fps(165);
  //pile->loco.window.set_max_fps(5);

  

  loco.loop([&] {
    mm.render();
  });
  // pile->close();

  return 0;
}