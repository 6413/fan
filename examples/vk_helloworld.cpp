// Creates window, opengl context

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0

#include <fan/types/types.h>
#include <fan/window/window.h>

#include <fan/graphics/vulkan/vk_core.h>

int main() {

  fan::window_t window;
  window.open();

  fan::vulkan::context_t context;
  context.open();
  context.bind_to_window(&window);

  context.set_vsync(&window, false);

  while (1) {
    uint32_t window_event = window.handle_events();
    if (window_event & fan::window_t::events::close) {
      window.close();
      break;
    }

    window.get_fps();
    context.render(&window, []{});
  }

  context.close();

  return 0;
}