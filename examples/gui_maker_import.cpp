// Example of opening gui maker


#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)
#define fan_debug fan_debug_low

#include _FAN_PATH(graphics/graphics.h)
#include _FAN_PATH(graphics/gui/fgm/import.h)

int main(int argc, char** argv) {

  if (argc < 2) {
    fan::throw_error("invalid amount of arguments. Usage:*.exe texturepack");
  }

  fan::window_t window;
  window.open();
  fan::opengl::context_t context;

  context.init();
  context.bind_to_window(&window);
  context.set_viewport(0, window.get_size());

  fan::opengl::texturepack tp;
  tp.open(&context, argv[1]);

  fan_2d::graphics::gui::fgm::load_t load;
  load.open(&window, &context);
  load.load(&window, &context, "123", &tp);
  load.enable_draw(&window, &context);

  load.set_on_input([](const std::string& id, fan_2d::graphics::gui::be_t* be, uint32_t index, uint16_t key, fan::key_state key_state
    ,fan_2d::graphics::gui::mouse_stage mouse_stage) {
      if (mouse_stage != fan_2d::graphics::gui::mouse_stage::inside) {
        return;
      }
      if (key != fan::mouse_left) {
        return;
      }
      if (key_state != fan::key_state::release) {
        return;
      }

      if (id == "drill") {
        fan::print("drill");
      }
      if (id == "dirt") {
        fan::print("dirt");
      }
  });

  context.set_vsync(&window, 0);

  while (1) {

    uint32_t window_event = window.handle_events();
    if (window_event & fan::window_t::events::close) {
      window.close();
      break;
    }

    context.process();
    context.render(&window);
  }

  return 0;
}

