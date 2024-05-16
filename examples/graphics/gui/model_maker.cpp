#include <fan/pch.h>


int main(int argc, char** argv) {
  if (argc < 2) {
    fan::throw_error("usage: TexturePackCompiled");
  }


  loco_t loco;
  //
  model_maker_t mm;
  mm.open(argv[1]);
  //if(argc == 3){
    //mm.load("entity_ship.fmm");
  //}
  //else{
  mm.load("test.json");//
  //}//
  ////
  //
  loco.set_vsync(false);
  //pile->loco.window.set_max_fps(165);
  //pile->loco.window.set_max_fps(5);
  loco.loop([&] {
    loco.get_fps();
  });


  // pile->close();

  return 0;
}