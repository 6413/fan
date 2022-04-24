# fan

**Example code:**
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
