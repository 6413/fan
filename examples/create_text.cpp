// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define fan_debug fan_debug_none

#include _FAN_PATH(graphics/gui/gui.h)
#include _FAN_PATH(graphics/opengl/2D/objects/text_renderer_string.h)

struct pile_t {
  fan::opengl::matrices_t matrices;
  fan::window_t window;
  fan::opengl::context_t context;
};

int main() {

  pile_t pile;

  pile.window.open();

  pile.context.init();
  pile.context.bind_to_window(&pile.window);
  pile.context.set_viewport(0, pile.window.get_size());
  pile.window.add_resize_callback(&pile, [](fan::window_t*, const fan::vec2i& size, void* userptr) {
    pile_t* pile = (pile_t*)userptr;

    pile->context.set_viewport(0, size);
  });

  pile.matrices.open();


  fan_2d::graphics::letter_t letter;

  fan_2d::graphics::font_t font;
  font.open(&pile.context, "fonts/bitter");
  letter.open(&pile.context, &font);

  fan_2d::graphics::letter_t::properties_t p;

  for (uint32_t i = 0; i < 5; i++) {
    p.position = fan::vec2(fan::random::value_f32(-0.7, 0.7), fan::random::value_f32(-0.7, 0.7));
    p.color = fan::random::color();
    p.font_size = 0.1;
    std::string str = fan::random::string(1);
    std::wstring w(str.begin(), str.end());
    p.letter_id = font.decode_letter(w[0]);

    letter.push_back(&pile.context, p);
  }
  letter.enable_draw(&pile.context);

  fan::vec2 window_size = pile.window.get_size();

  pile.matrices.set_ortho(&pile.context, fan::vec2(0, window_size.x), fan::vec2(0, window_size.y));

  pile.context.set_vsync(&pile.window, 0);

  while(1) {

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