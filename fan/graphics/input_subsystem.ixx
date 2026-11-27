module;

export module fan.graphics.input_subsystem;

#if defined (FAN_WINDOW)

import std;

export import fan.window;
export import fan.window.input_action;

export namespace fan::graphics {
  struct input_subsystem_t {
    void init(fan::window_t& window);
    void destroy();

    // movement, combat and debug actions: added by init, a program that isn't a game removes them
    void add_game_actions();
    void remove_game_actions();
    static constexpr const char* game_actions[] = {
      "Move Forward", "Move Back", "Move Left", "Move Right", "Move Up",
      "Light Attack", "Block Attack",
      "Toggle Debug Physics", "Toggle Light Buffer", "Recompile Shaders",
    };

    fan::window::input_action_t input_action;

    fan::window_t::buttons_handle_t  buttons_handle;
    fan::window_t::keys_handle_t     keys_handle;
    fan::window_t::mouse_move_handle_t mouse_move_handle;
    fan::window_t::text_callback_handle_t text_callback_handle;
  };
}

#endif