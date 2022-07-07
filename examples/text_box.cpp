// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/gui/gui.h)


using id_holder_t = bll_t<uint32_t>;

struct pile_t;

using text_box_t = fan_2d::graphics::gui::rectangle_text_box_t<pile_t*, uint32_t>;
using letter_t = text_box_t::letter_t;

struct pile_t {
  fan::opengl::matrices_t matrices;
  fan::window_t window;
  fan::opengl::context_t context;
  id_holder_t ids;
  text_box_t text_box;
  letter_t letters;
};

void letter_cb(letter_t* l, uint32_t src, uint32_t dst, text_box_t::text_renderer_t::letter_data_t* lp) {
  //l->user_global_data->ids[*p] = dst;
}

void text_box_cb(text_box_t::box_t::rectangle_t* l, uint32_t src, uint32_t dst, uint32_t *p) {
  l->get_user_global_data()->ids[*p] = dst;
}

int main() {

  pile_t pile;

  pile.window.open();

  pile.context.init();
  pile.context.bind_to_window(&pile.window);
  pile.context.set_viewport(0, pile.window.get_size());

  pile.window.add_resize_callback(&pile, [](fan::window_t*, const fan::vec2i& size, void* userptr) {
    pile_t* pile = (pile_t*)userptr;

    pile->context.set_viewport(0, size);

    fan::vec2 window_size = pile->window.get_size();
    fan::vec2 ratio = window_size / window_size.max();
    std::swap(ratio.x, ratio.y);
    pile->matrices.set_ortho(&pile->context, fan::vec2(-1, 1) * ratio.x, fan::vec2(-1, 1) * ratio.y);
  });

  pile.matrices.open();

  pile.ids.open();

  fan_2d::graphics::font_t font;
  font.open(&pile.context, "fonts/bitter");

  pile.letters.open(&pile.context, &font, letter_cb, &pile);
  pile.letters.bind_matrices(&pile.context, &pile.matrices);

  pile.text_box.open(&pile.context, text_box_cb, &pile);
  pile.text_box.bind_matrices(&pile.context, &pile.matrices);
  text_box_t::properties_t tp;
  tp.theme.button.outline_thickness = 0.005;
  tp.position = 0;
  tp.size = fan::vec2(0.4, 0.1);
  tp.text = "HeLoWoRlD_";
  pile.text_box.push_back(&pile.context, &pile.letters, tp);
  pile.text_box.enable_draw(&pile.context);

  pile.letters.enable_draw(&pile.context);

  fan::vec2 window_size = pile.window.get_size();
  fan::vec2 ratio = window_size / window_size.max();
  std::swap(ratio.x, ratio.y);
  pile.matrices.set_ortho(&pile.context, fan::vec2(-1, 1) * ratio.x, fan::vec2(-1, 1) * ratio.y);

  while(1) {

    uint32_t window_event = pile.window.handle_events();
    if(window_event & fan::window_t::events::close){
      pile.window.close();
      break;
    }

    pile.context.process();
    pile.context.render(&pile.window);
  }

  return 0;
}