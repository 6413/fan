# fan

**Example code:**
```
// Creates window and opengl context

#include <fan/graphics/graphics.h>

int main() {

  fan::window_t w;
  w.open();

  fan::opengl::context_t c;
  c.init();
  c.bind_to_window(&w);
  c.set_viewport(0, w.get_size());
  w.add_resize_callback(&c, [](fan::window_t*, const fan::vec2i& size, void* userptr) {
    ((fan::opengl::context_t*)userptr)->set_viewport(0, size);
  });

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
```
