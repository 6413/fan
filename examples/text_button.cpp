// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/gui/gui.h)

struct pile_t;

using rectangle_t = fan_2d::graphics::rectangle_t<pile_t*, uint32_t>;
using rectangle_text_button_t = fan_2d::graphics::gui::rectangle_text_button_t<pile_t*, uint32_t>;
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

using loco_t = fan_2d::graphics::loco_t<draw_pile_types_t>;

void letter_cb(letter_t* l, uint32_t src, uint32_t dst, rectangle_text_button_t::text_renderer_t::letter_data_t* lp) {
  
}

void text_button_cb(rectangle_text_button_t::box_t::rectangle_t* l, uint32_t src, uint32_t dst, uint32_t *p) {
  
}

void be_cb(fan_2d::graphics::gui::be_t*, uint32_t src, uint32_t dst) {

}

int main() {

  loco_t loco;

  pile_t pile;
  pile.window.open();
  pile.window.add_resize_callback(&loco, [](fan::window_t*, const fan::vec2i& size, void* userptr) {
    loco_t* pile = (loco_t*)userptr;

    pile->context.set_viewport(0, size);

    fan::vec2 window_size = size;
    fan::vec2 ratio = window_size / window_size.max();
    std::swap(ratio.x, ratio.y);
    pile->matrices.set_ortho(&pile->context, fan::vec2(-1, 1) * ratio.x, fan::vec2(-1, 1) * ratio.y);
  });

  loco.context.init();
  loco.context.bind_to_window(&pile.window);
  loco.context.set_viewport(0, pile.window.get_size());

  loco.matrices.open();

  fan_2d::graphics::font_t font;
  font.open(&loco.context, "fonts/bitter");

  loco_t::properties_t lp;
  lp.font = &font;
  loco.open(lp);

  loco.letter.open(&loco.context, &font, letter_cb, &pile);
  loco.letter.bind_matrices(&loco.context, &loco.matrices);

  loco.rectangle_text_button.open(&loco.context, text_button_cb, &pile);
  loco.rectangle_text_button.bind_matrices(&loco.context, &loco.matrices);

  fan_2d::graphics::gui::be_t be;
  be.open(be_cb);
  be.bind_to_window(&pile.window);

  rectangle_text_button_t::properties_t tp;
  tp.theme.button.outline_thickness = 0.005;
  tp.position = 0;
  tp.size = fan::vec2(0.4, 0.1);
  tp.text = "HeLoWoRlD_";
  loco.rectangle_text_button.push_back(&loco.context, &be, &loco.letter, tp);
  loco.rectangle_text_button.enable_draw(&loco.context);

  loco.letter.enable_draw(&loco.context);

  fan::vec2 window_size = pile.window.get_size();
  fan::vec2 ratio = window_size / window_size.max();
  std::swap(ratio.x, ratio.y);
  loco.matrices.set_ortho(&loco.context, fan::vec2(-1, 1) * ratio.x, fan::vec2(-1, 1) * ratio.y);

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