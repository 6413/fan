// Example of opening gui maker

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)
#define fan_debug 0

#include _FAN_PATH(graphics/graphics.h)
#include _FAN_PATH(graphics/gui/fgm/fgm.h)

int main(int argc, char** argv) {

  if (argc < 2) {
    fan::throw_error("invalid amount of arguments. Usage:*.exe texturepack");
  }

  fan_2d::graphics::gui::fgm::pile_t pile;

  pile.open(argc, argv);
  //pile.load_file("123");
  pile.context.set_vsync(&pile.window, 0);

  while (1) {

    pile.window.get_fps();

    uint32_t window_event = pile.window.handle_events();
    if (window_event & fan::window_t::events::close) {
      pile.window.close();
      break;
    }

    pile.context.process();
    pile.context.render(&pile.window);
  }

  return 0;
}
