#include <fan/pch.h>

#include <fan/graphics/gui/model_maker/maker.h>

int main(int argc, char** argv) {
  if (argc < 2) {//
    fan::throw_error("usage: TexturePackCompiled");
  }////
  //////////
  //////////
  loco_t loco;
  
  model_maker_t mm;////

  mm.open("texture_packs/TexturePack", L"examples/games/puzzle");
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
  
  });
  // pile->close();

  return 0;
}