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
    fan::vec2 window_size = loco.get_window()->get_size();
    fan::vec2 ratio = window_size / window_size.max();
    fan::graphics::open_matrices(
      loco.get_context(),
      &matrices,
      loco.get_window()->get_size(),
      fan::vec2(-1, 1),
      fan::vec2(-1, 1),
      ratio
    );
    loco.get_window()->add_resize_callback(this, [](fan::window_t* window, const fan::vec2i& size, void* userptr) {
      fan::vec2 window_size = window->get_size();
      fan::vec2 ratio = window_size / window_size.max();
      //std::swap(ratio.x, ratio.y);
      pile_t* pile = (pile_t*)userptr;
      pile->matrices.set_ortho(
        fan::vec2(-1, 1),
        fan::vec2(-1, 1),
        ratio
      );
      pile->viewport[0].set_viewport(pile->loco.get_context(), 0, pile->loco.get_window()->get_size(), pile->loco.get_window()->get_size());
     });

    loco.get_window()->add_keys_callback(0, [](fan::window_t* w, uint16_t key, fan::key_state key_state, void* userptr) {
      if (key_state == fan::key_state::release) {
        return;
      }
      fan::vec2 window_size = w->get_size();
      switch (key) {
      case fan::key_up: {
        w->set_size(window_size + fan::vec2(0, -100));
        break;
      }
      case fan::key_down: {
        w->set_size(window_size + fan::vec2(0, 100));
        break;
      }
      case fan::key_left: {
        w->set_size(window_size + fan::vec2(-100, 0));
        break;
      }
      case fan::key_right: {
        w->set_size(window_size + fan::vec2(100, 0));
        break;
      }
      }
      fan::print(w->get_size());
    });

    fan::vec2 position = 0;
    fan::vec2 size = loco.get_window()->get_size();
    //position.y -= 200;
    //position.y += size.y / 2;
    //size.y /= 2;
    viewport[0].open(loco.get_context());
    viewport[0].set_viewport(loco.get_context(), position, size, loco.get_window()->get_size());
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
 // tp.position = 400;
  tp.position = 0;
  //tp.position.y = 0;
 // tp.position.z = 50;
  tp.size = fan::vec2(0.3, 0.1) 
    //* 300
    ;
  tp.text = "hello world";
  //tp.font_size = 32;
  tp.mouse_move_cb = [] (const loco_t::mouse_move_data_t& mm_d) -> void {
    fan::print(mm_d.position, (int)mm_d.mouse_stage);
  };
  tp.mouse_button_cb = [](const loco_t::mouse_button_data_t& ii_d) -> void {

  };
  fan_2d::graphics::gui::themes::gray gray_theme;
  gray_theme.open(pile.loco.get_context());
  tp.theme = &gray_theme;
  constexpr auto count = 10;
  fan::opengl::cid_t cids[count];
  fan::print(loco_bdbt_usage(&pile.loco.bdbt));
  pile.loco.button.push_back(&cids[0], tp);

  while(pile.loco.window_open(pile.loco.process_frame([]{}))) {

  }

  return 0;
}