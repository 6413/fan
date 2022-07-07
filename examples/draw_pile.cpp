// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/gui/gui.h)

struct pile_t;

// filler                         
using rectangle_t = fan_2d::graphics::rectangle_t<pile_t*, uint32_t>;
using sprite_t = fan_2d::graphics::sprite_t<pile_t*, uint32_t>;
using text_box_t = fan_2d::graphics::gui::rectangle_text_box_sized_t<pile_t*, uint32_t>;
using letter_t = text_box_t::text_renderer_t::letter_t;

struct pile_t {
  fan::opengl::matrices_t matrices;
  fan::window_t window;
  fan::opengl::context_t context;
  text_box_t text_box;
  letter_t letters;
};

struct _t {
  rectangle_t rectangle;
  sprite_t sprite;
};

using draw_pile_t = fan_2d::graphics::draw_pile_t<_t>;

int main() {

  draw_pile_t draw_pile;

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

  rectangle_t::properties_t rp;
  sprite_t::properties_t sp;
  
  draw_pile.push_back(&pile.context, sp);

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