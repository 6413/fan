#include fan_pch

struct pile_t {

  pile_t() {

  }

  loco_t loco_var_name;
};

pile_t* pile = new pile_t;

#include _FAN_PATH(graphics/gui/model_maker/maker.h)

int main(int argc, char** argv) {
  if (argc < 2) {
    fan::throw_error("usage: TexturePackCompiled");
  }
  
  model_maker_t mm;
  mm.open(argv[1]);
  if(argc == 3){
    mm.load(argv[2]);
  }
  else{
    mm.load("model.fmm");
  }
  loco_t::sprite_t::properties_t p;

  
  pile->loco.set_vsync(false);
  //pile->loco.get_window()->set_max_fps(165);
  //pile->loco.get_window()->set_max_fps(5);
  pile->loco.loop([&] {
    pile->loco.get_fps();
  });


  // pile->close();

  return 0;
}