# fan

**Example code:**

// Code creates window and context with it
#include <fan/graphics/graphics.h>

int main() {

  fan::window_t w;
  w.open();

  fan::opengl::context_t c;
  c.init();
  c.bind_to_window(&w);
  c.set_viewport(0, w.get_size());

  while(1) {

    uint32_t window_event = w.handle_events();
    if(window_event & fan::window_t::events::close){
      w.close();
      break;
    }

    c.process();
    c.render(&w);
  }

  return 0;
}
