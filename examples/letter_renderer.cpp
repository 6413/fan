// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define fan_debug fan_debug_high

#include _FAN_PATH(graphics/graphics.h)

constexpr uint32_t count = 1000;

struct pile_t {
  fan::opengl::matrices_t matrices;
  fan::window_t window;
  fan::opengl::context_t context;
  fan::opengl::cid_t cids[count];
};

using letter_t = fan_2d::graphics::letter_t;

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
  });

  pile.matrices.open();


  letter_t letter;

  fan_2d::graphics::font_t font;
  font.open(&pile.context, "fonts/bitter");
  letter.open(&pile.context, &font);
  letter.bind_matrices(&pile.context, &pile.matrices);

  letter_t::properties_t p;

  for (uint32_t i = 0; i < count; i++) {
    p.position = fan::vec2(fan::random::value_f32(0.1, .9), fan::random::value_f32(.1, .9));
    p.color = fan::color(1, 0, f32_t(i) / count, 1);
    p.font_size = 0.1;
    std::string str = fan::random::string(1);
    std::wstring w(str.begin(), str.end());
    p.letter_id = font.decode_letter(w[0]);

    letter.push_back(&pile.context, &pile.cids[i], p);
  }

  letter.enable_draw(&pile.context);

  fan::color color = letter.get(&pile.context, &pile.cids[0], &letter_t::instance_t::color);
  fan::print(color);
  letter.set(&pile.context, &pile.cids[0], &letter_t::instance_t::color, fan::color(0, 0, 1));
  color = letter.get(&pile.context, &pile.cids[0], &letter_t::instance_t::color);
  fan::print(color);

  fan::vec2 window_size = pile.window.get_size();

  pile.matrices.set_ortho(&pile.context, fan::vec2(0, 1), fan::vec2(0, 1));

  pile.context.set_vsync(&pile.window, 0);
  //pile.window.set_max_fps(2);

  static uint32_t i = 0;
  while(1) {
    letter.erase(&pile.context, &pile.cids[i]);
    //p.position = fan::vec2(fan::random::value_f32(0.1, .9), fan::random::value_f32(.1, .9));
    //p.color = fan::color(0, 0, 1, 1);
    //p.font_size = 0.1;
    //std::string str = fan::random::string(1);
    //std::wstring w(str.begin(), str.end());
    //p.letter_id = font.decode_letter(w[0]);

    //letter.push_back(&pile.context, p);

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