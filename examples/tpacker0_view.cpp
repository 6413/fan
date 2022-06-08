// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/graphics.h)


#include _FAN_PATH(tp/tp.h)

#define gui_demo

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
#ifdef gui_demo
    pile->matrices.set_ortho(&pile->context, fan::vec2(0, size.x), fan::vec2(0, size.y));
#endif
    });

  pile.matrices.open();

  fan_2d::graphics::sprite_t r;
  r.open(&pile.context);
  r.bind_matrices(&pile.context, &pile.matrices);
  r.enable_draw(&pile.context);

  fan_2d::graphics::sprite_t::properties_t p;
#ifdef gui_demo
#else
  p.position = 0;
  p.size = 0.5;
#endif

  fan::tp::texture_packe0 tpacker_view;
  tpacker_view.open();
  tpacker_view.load("TexturePack");
  tpacker_view.process();

  p.image.load(&pile.context, tpacker_view.get_pixel_data(1));
  p.size = p.image.size / 2;
  p.position = p.size;
  r.push_back(&pile.context, p);

  fan::vec2 window_size = pile.window.get_size();
#ifdef gui_demo
  pile.matrices.set_ortho(&pile.context, fan::vec2(0, window_size.x), fan::vec2(0, window_size.y));
#else
  pile.matrices.set_ortho(&pile.context, fan::vec2(-1, 1), fan::vec2(1, -1));
#endif

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