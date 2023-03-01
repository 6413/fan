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

    /* loco.get_window()->add_resize_callback(this, [](fan::window_t* window, const fan::vec2i& size, void* userptr) {
       fan::vec2 window_size = window->get_size();
       fan::vec2 ratio = window_size / window_size.max();
       std::swap(ratio.x, ratio.y);
       pile_t* pile = (pile_t*)userptr;
       pile->camera.set_ortho(
         &loco,
         ortho_x * ratio.x,
         ortho_y * ratio.y
       );
     });*/
  }
  ~pile_t() {

  }

  loco_t loco_var_name;
};

pile_t* pile;

#include _FAN_PATH(graphics/gui/fgm/fgm.h)

//struct fork_t : fgm_t {
//
//};

int main(int argc, char** argv) {
  if (argc < 2) {
    fan::throw_error("usage: TexturePackCompiled");
  }

  pile = new pile_t;

  fgm_t fgm;
  fgm.open(argv[1]);
  fgm.load();

  pile->loco.set_vsync(false);
  pile->loco.get_window()->set_max_fps(165);
  //pile->loco.get_window()->set_max_fps(5);
  pile->loco.loop([&] {

    });


  // pile->close();

  return 0;
}