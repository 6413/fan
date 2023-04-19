#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define fan_windows_subsystem fan_windows_subsystem_windows

//#define loco_vulkan

#define loco_window
#define loco_context

#define loco_line
#define loco_button
#define loco_sprite
#define loco_menu_maker_button
#define loco_menu_maker_text_box
#define loco_var_name loco
#include _FAN_PATH(graphics/loco.h)

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