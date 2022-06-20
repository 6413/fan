// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define fan_debug fan_debug_none

#include _FAN_PATH(graphics/graphics.h)

struct pile_t {
  fan::opengl::matrices_t matrices;
  fan::window_t window;
  fan::opengl::context_t context;
  fan_2d::graphics::text_renderer_t tr;
};

//void cb(letter_t* l, uint32_t src, uint32_t dst, void *p) {
//  fan::print(src, dst);
//}

int main() {

  pile_t pile;

  pile.window.open();

  pile.context.init();
  pile.context.bind_to_window(&pile.window);
  pile.context.set_viewport(0, pile.window.get_size());
  pile.window.add_resize_callback(&pile, [](fan::window_t*, const fan::vec2i& size, void* userptr) {
    pile_t* pile = (pile_t*)userptr;

    pile->context.set_viewport(0, size);
    fan::vec2 ratio = fan::cast<f32_t>(size) / size.max();
    pile->matrices.set_ortho(&pile->context, fan::vec2(0, 1) * ratio.x, fan::vec2(0, 1) * ratio.y);
  });

  pile.matrices.open();

  fan_2d::graphics::font_t font;
  font.open(&pile.context, "fonts/bitter");
  int x;
  pile.tr.open(&pile.context, &font);
  pile.tr.bind_matrices(&pile.context, &pile.matrices);

  constexpr auto c = 1000;

  uint32_t ids[c];

  fan_2d::graphics::text_renderer_t::properties_t p;
  
  p.font_size = 0.05;
  for (uint32_t i = 0; i < c; i++) {
    p.position = fan::random::vec2(0.1, 0.9);
    p.text = fan::random::string(5);
    ids[i] = pile.tr.push_back(&pile.context, p);
  }
  

  pile.tr.enable_draw(&pile.context);

  pile.matrices.set_ortho(&pile.context, fan::vec2(0, 1), fan::vec2(0, 1));

  pile.context.set_vsync(&pile.window, 0);

  while(1) {

    pile.tr.erase(&pile.context, ids[0]);
    p.position = fan::random::vec2(0.1, 0.9);
    p.text = fan::random::string(5);
    ids[0] = pile.tr.push_back(&pile.context, p);

    pile.window.get_fps();

    // letter.set_position(&pile.context, 0, fan::cast<f32_t>(pile.window.get_mouse_position()) / pile.window.get_size());

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