#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 1
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
  loco_t loco_var_name;
};

pile_t* pile;

#define stage_maker_var_name stage_maker
#define fgm_build_stage_maker
#include _FAN_PATH(graphics/gui/stage_maker/maker.h)
stage_maker_t* stage_maker_var_name;

int main(int argc, char** argv) {
  if (argc < 2) {
    fan::throw_error("usage: TexturePackCompiled");
  }

  pile = new pile_t;

  stage_maker_var_name = new stage_maker_t;

  stage_maker_var_name->open(argv[1]);

  pile->loco.set_vsync(false);
  pile->loco.get_window()->set_max_fps(165);
  //pile->loco.get_window()->set_max_fps(5);
  pile->loco.loop([&] {
    //pile->loco.get_fps();
    //fan::print(pile->loco.menu_maker.get_selected(pile->stage_maker.instances[pile_t::stage_maker_t::stage_t::state_instance].menu_id));
  });


 // pile->close();

  return 0;
}