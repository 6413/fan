// rectangle text button using loco

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/gui/gui.h)

using rectangle_text_button_t = fan_2d::graphics::gui::rectangle_text_button_t;
using letter_t = rectangle_text_button_t::letter_t;

#define loco_window
#define loco_context
// for testing
#define loco_rectangle
#define loco_sprite

#define loco_letter
#define loco_button
#include _FAN_PATH(graphics/loco.h)

struct pile_t {

  void open() {
    loco.open(loco_t::properties_t());
    matrices = fan::graphics::open_matrices(
      loco.get_window()->get_size(),
      fan::vec2(-1, 1),
      fan::vec2(-1, 1)
    );
  }

  loco_t loco;
  fan::opengl::matrices_t matrices;
};

int main() {

  pile_t pile;
  pile.open();

  loco_t::rectangle_text_button_t::properties_t tp;
  tp.position = 0;
  tp.size = fan::vec2(0.3, 0.1);
  tp.text = "hello world";
  tp.mouse_move_cb = [] (const loco_t::mouse_move_data_t& mm_d) -> uint8_t {
    fan::print((int)mm_d.mouse_stage, mm_d.depth);
    return 0;
  };
  tp.mouse_input_cb = [](const loco_t::mouse_input_data_t& ii_d) -> uint8_t {

    fan::print(ii_d.key, (int)ii_d.key_state, (int)ii_d.mouse_stage, ii_d.depth);
    return 0;
  };
  uint32_t ids[2];
  ids[0] = pile.loco.button.push_back(0, tp);
  tp.theme = fan_2d::graphics::gui::themes::gray();
  tp.position.x += 0.1;
  tp.position.z += 0.2;
  tp.text = "hw2";
  ids[1] = pile.loco.button.push_back(1, tp);

  while(pile.loco.window_open(pile.loco.process_frame())) {

  }

  return 0;
}