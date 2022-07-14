// rectangle text button using loco

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/gui/gui.h)


using rectangle_t = fan_2d::graphics::rectangle_t;
using rectangle_text_button_t = fan_2d::graphics::gui::rectangle_text_button_t;
using letter_t = rectangle_text_button_t::letter_t;

struct pile_t {
  fan::window_t window;
};

struct draw_pile_types_t {
  fan::opengl::matrices_t matrices;
  rectangle_text_button_t rectangle_text_button;
  letter_t letter;
  rectangle_t rectangle;
};

#define loco_rectangle
#define loco_sprite
#define loco_letter
#define loco_rectangle_text_button
#include _FAN_PATH(graphics/loco.h)

int main() {

  loco_t loco;

  pile_t pile;
  pile.window.open();

  loco.context.init();
  loco.context.bind_to_window(&pile.window);
  loco.context.set_viewport(0, pile.window.get_size());

  fan::opengl::matrices_t matrices;
  matrices.open();

  fan::vec2 window_size = pile.window.get_size();
  fan::vec2 ratio = window_size / window_size.max();
  std::swap(ratio.x, ratio.y);
  matrices.set_ortho(&loco.context, fan::vec2(-1, 1), fan::vec2(-1, 1));

  loco_t::properties_t lp;
  lp.matrices = &matrices;
  loco.open(lp);

  rectangle_text_button_t::properties_t tp;
  tp.position = 0;
  tp.size = fan::vec2(0.3, 0.1);
  tp.text = "hello world";
  fan::opengl::cid_t cid;
  loco.push_back(&cid, tp);
  loco.rectangle_text_button.enable_draw(&loco.context);

  loco.letter.enable_draw(&loco.context);

  pile.window.add_keys_callback(&loco, [](fan::window_t* window, uint16_t key, fan::key_state key_state, void* user_ptr) {
    loco_t& loco = *(loco_t*)user_ptr;
    fan::vec2 window_size = window->get_size();
    loco.feed_mouse_input(&loco.context, key, key_state, fan::cast<f32_t>(window->get_mouse_position()) / window_size * 2 - 1, 0); // depth
  });

  pile.window.add_mouse_move_callback(&loco, [](fan::window_t* window, const fan::vec2i& mouse_position, void* user_ptr) {
    loco_t& loco = *(loco_t*)user_ptr;
    fan::vec2 window_size = window->get_size();
    loco.feed_mouse_move(&loco.context, fan::cast<f32_t>(mouse_position) / window_size * 2 - 1, 0); // depth
  });

  while(1) {

    uint32_t window_event = pile.window.handle_events();
    if(window_event & fan::window_t::events::close){
      pile.window.close();
      break;
    }

    loco.context.process();
    loco.context.render(&pile.window);
  }

  return 0;
}