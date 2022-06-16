// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/gui/gui.h)

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


  fan_2d::graphics::gui::rectangle_text_button_sized_t rtbs;
  fan_2d::graphics::gui::rectangle_text_button_sized_t::properties_t p;

  rtbs.open(&pile.window, &pile.context);
  rtbs.bind_matrices(&pile.context, &pile.matrices);
  rtbs.enable_draw(&pile.window, &pile.context);
  p.position = pile.window.get_size() / 2;
  p.size = 100;

  rtbs.push_back(&pile.window, &pile.context, p);

  fan::vec2 window_size = pile.window.get_size();

  pile.matrices.set_ortho(&pile.context, fan::vec2(0, window_size.x), fan::vec2(0, window_size.y));

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