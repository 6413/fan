module;

#include <fan/utility.h>

export module fan.graphics.input_subsystem;

export import fan.window;
export import fan.window.input_action;

export namespace fan::graphics {
  struct input_subsystem_t {
    void init(fan::window_t& window);
    void destroy();

    fan::window::input_action_t input_action;

    fan::window_t::buttons_handle_t  buttons_handle;
    fan::window_t::keys_handle_t     keys_handle;
    fan::window_t::mouse_move_handle_t mouse_move_handle;
    fan::window_t::text_callback_handle_t text_callback_handle;
  };
}