// rectangle text button using loco

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_window
#define loco_context

#define loco_button
#include _FAN_PATH(graphics/loco.h)

struct pile_t {

  void open() {
    loco.open(loco_t::properties_t());
    fan::graphics::open_matrices(
      loco.get_context(),
      &matrices,
      loco.get_window()->get_size(),
      fan::vec2(-1, 1),
      fan::vec2(-1, 1)
    );
    viewport[0].open(loco.get_context(), 0, loco.get_window()->get_size());
    viewport[1].open(loco.get_context(), 0, loco.get_window()->get_size());
  }

  loco_t loco;
  fan::opengl::matrices_t matrices;
  fan::opengl::viewport_t viewport[2];
};

int main() {

  pile_t pile;
  pile.open();

  loco_t::button_t::properties_t tp;
  tp.matrices = &pile.matrices;
  tp.viewport = &pile.viewport[0];
  tp.position = 0;
  tp.position.z = 50;
  tp.size = fan::vec2(0.3, 0.1);
  tp.text = "hello world";
  tp.mouse_move_cb = [] (const loco_t::mouse_move_data_t& mm_d) -> void {
    fan::print(mm_d.position, (int)mm_d.mouse_stage);
  };
  tp.mouse_button_cb = [](const loco_t::mouse_button_data_t& ii_d) -> void {
    if (ii_d.flag->ignore_move_focus_check == false) {
      if (ii_d.button == fan::mouse_left && ii_d.button_state == fan::key_state::press) {
        ii_d.flag->ignore_move_focus_check = true;
      }
    }
    else {
      if (ii_d.button == fan::mouse_left && ii_d.button_state == fan::key_state::release) {
        ii_d.flag->ignore_move_focus_check = false;
      }
    }
    fan::print(ii_d.button, (int)ii_d.button_state, (int)ii_d.mouse_stage);
  };
  fan_2d::graphics::gui::themes::gray gray_theme;
  gray_theme.open(pile.loco.get_context());
  tp.theme = &gray_theme;
  constexpr auto count = 10;
  fan::opengl::cid_t cids[count];
  fan::print(loco_bdbt_usage(&pile.loco.bdbt));
  pile.loco.button.push_back(&cids[0], tp);
  tp.position.x += 0.4;
  tp.position.z += 10;
  //pile.loco.button.push_back(&pile.loco, &cids[1], tp);
  //pile.loco.button.set_position(0, 0, 0);
  

 // pile.loco.button.set(&pile.loco, &cids[2], &loco_t::button_t::instance_t::position, 0);

  while(pile.loco.window_open(pile.loco.process_frame([]{}))) {

  }

  return 0;
}