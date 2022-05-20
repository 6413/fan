# fan

**Example window creation:**
```
// Creates window and opengl context

#include <fan/graphics/graphics.h>

int main() {

  fan::window_t window;
  window.open();

  fan::opengl::context_t context;
  context.init();
  context.bind_to_window(&window);
  context.set_viewport(0, window.get_size());
  window.add_resize_callback(&context, [](fan::window_t*, const fan::vec2i& size, void* userptr) {
    ((fan::opengl::context_t*)userptr)->set_viewport(0, size);
  });

  while(1) {

    uint32_t window_event = window.handle_events();
    if(window_event & fan::window_t::events::close){
      window.close();
      break;
    }

    context.process();
    context.render(&window);
  }

  return 0;
}
```

**Example rectangle rendering:**
```
// Creates window, opengl context and renders a rectangle

#include <fan/graphics/graphics.h>

//#define gui_demo

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
    #ifndef gui_demo
      pile->matrices.set_ortho(&pile->context, fan::vec2(0, size.x), fan::vec2(0, size.y));
    #endif
  });

  pile.matrices.open();


  fan_2d::graphics::rectangle_t r;
  r.open(&pile.context);
  r.m_shader.bind_matrices(&pile.context, &pile.matrices);
  r.enable_draw(&pile.context);

  fan_2d::graphics::rectangle_t::properties_t p;
  #ifdef gui_demo
    p.position = 0;
    p.size = 0.5;
  #else
    p.position = pile.window.get_size() / 2;
    p.size = 100;
  #endif
  p.color = fan::colors::red;
  r.push_back(&pile.context, p);

  fan::vec2 window_size = pile.window.get_size();
    #ifdef gui_demo
    pile.matrices.set_ortho(&pile.context, fan::vec2(-1, 1), fan::vec2(1, -1));
  #else
    pile.matrices.set_ortho(&pile.context, fan::vec2(0, window_size.x), fan::vec2(0, window_size.y));
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
```

