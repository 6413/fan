module;

#if defined (FAN_WINDOW)

// some random gcc bugs
#include <fan/graphics/shape_macros.h>

#endif

module fan.graphics.input_subsystem;

#if defined (FAN_WINDOW)

import std;

import fan.window.input;

import fan.graphics.shapes;
import fan.graphics.common_context;

namespace fan::graphics {
  void input_subsystem_t::init(fan::window_t& window) {
    input_action.window = &window;

#if defined(FAN_GUI)
    input_action.insert_or_assign(
      {fan::key_escape, fan::gamepad_start},
      fan::actions::toggle_settings,
      fan::actions::groups::system
    );
    input_action.insert_or_assign(
      fan::key_f3,
      fan::actions::toggle_console,
      fan::actions::groups::system
    );
#endif

    input_action.insert_or_assign(
      {fan::key_a, fan::gamepad_left_thumb},
      fan::actions::move_left,
      fan::actions::groups::movement
    );
    input_action.insert_or_assign(
      {fan::key_d, fan::gamepad_left_thumb},
      fan::actions::move_right,
      fan::actions::groups::movement
    );
    input_action.insert_or_assign(fan::key_w, fan::actions::move_forward, fan::actions::groups::movement);
    input_action.insert_or_assign(fan::key_s, fan::actions::move_back,    fan::actions::groups::movement);
    input_action.insert_or_assign(
      {fan::key_space, fan::key_w, fan::gamepad_a},
      fan::actions::move_up,
      fan::actions::groups::movement
    );
    input_action.insert_or_assign(
      {fan::mouse_left, fan::gamepad_right_bumper},
      fan::actions::light_attack,
      fan::actions::groups::combat
    );
    input_action.insert_or_assign(
      {fan::mouse_right, fan::gamepad_left_bumper},
      fan::actions::block_attack,
      fan::actions::groups::combat
    );
    input_action.insert_or_assign_combo(
      {fan::key_left_control, fan::key_4},
      fan::actions::toggle_debug_light_buffer,
      fan::actions::groups::debug
    );
#if defined(FAN_PHYSICS_2D)
    input_action.insert_or_assign_combo(
      {fan::key_left_control, fan::key_5},
      fan::actions::toggle_debug_physics,
      fan::actions::groups::debug
    );
#endif
    input_action.insert_or_assign_combo(
      {fan::key_left_control, fan::key_left_shift, fan::key_r},
      fan::actions::recompile_shaders,
      fan::actions::groups::debug
    );

#if defined(FAN_2D)
    buttons_handle = window.add_buttons_callback([](const fan::window_t::buttons_data_t& d) {
      fan::graphics::g_shapes->vfi.feed_mouse_button(d.button, d.state);
    });
#endif

    keys_handle = window.add_keys_callback([&window, windowed = true](const fan::window_t::keys_data_t& d) mutable {
#if defined(FAN_2D)
      fan::graphics::g_shapes->vfi.feed_keyboard(d.key, d.state);
#endif
      if (d.key == fan::key_enter && d.state == fan::keyboard_state::press &&
          window.key_pressed(fan::key_left_alt)) {
        windowed = !windowed;
        d.window->set_display_mode(windowed
          ? fan::window_t::mode::windowed
          : fan::window_t::mode::borderless);
      }
    });

#if defined(FAN_2D)
    mouse_move_handle = window.add_mouse_move_callback([](const fan::window_t::mouse_move_data_t& d) {
      fan::graphics::g_shapes->vfi.feed_mouse_move(d.position);
    });
    text_callback_handle = window.add_text_callback([](const fan::window_t::text_data_t& d) {
      fan::graphics::g_shapes->vfi.feed_text(d.character);
    });
#endif
  }

  void input_subsystem_t::destroy() {
    // handles are RAII — destruct automatically
  }
}

#endif