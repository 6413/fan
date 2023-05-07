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
#define loco_var_name loco
#define loco_line
#define loco_button
#define loco_sprite
#define loco_dropdown
#include _FAN_PATH(graphics/loco.h)

struct pile_t {
  loco_t loco_var_name;
};

pile_t* pile = new pile_t;

#define fgm_build_stage_maker
#include _FAN_PATH(graphics/gui/stage_maker/maker.h)

  fan::masterpiece_t<int, double, int> m;




int main(int argc, char** argv) {

  int x = m.iterate_ret([]<typename T>(const auto& i, const T& e) -> int {
    fan::print(i);
    if (i.value == 1) {
      return 1;
    }
    return 0;
  });

  return x;

  if (argc < 2) {
    fan::throw_error("usage: TexturePackCompiled");
  }

  stage_maker_t stage_maker(argv[1]);

  pile->loco.set_vsync(false);
  //pile->loco.get_window()->set_max_fps(165);
  //pile->loco.get_window()->set_max_fps(5);
  pile->loco.loop([&] {
  //  pile->loco.get_fps();
    //fan::print(pile->loco.menu_maker.get_selected(pile->stage_maker.instances[pile_t::stage_maker_t::stage_t::state_instance].menu_id));
  });


 // pile->close();

  return 0;
}