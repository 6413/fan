#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 1
#include fan_pch

struct pile_t {
  loco_t loco_var_name;
};

pile_t* pile = new pile_t;

#define fgm_build_stage_maker
#include _FAN_PATH(graphics/gui/stage_maker/maker.h)

  fan::masterpiece_t<int, double, int> m;




int main(int argc, char** argv) {
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