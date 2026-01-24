module;

#if defined(FAN_GUI)

#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <functional>

#endif

export module fan.graphics.gui.keybinds_menu;

#if defined(FAN_GUI)

import fan.window.input_action;
import fan.graphics.gui.base;
import fan.types.color;
import fan.types.json;

export namespace fan::graphics::gui {
  struct keybind_menu_t {
    using key_code_t = fan::key_code_t;
    using combo_t = fan::window::input_action_t::combo_t;

    static constexpr const char* label_action = "Action";
    static constexpr const char* device_name_keyboard = "Keyboard";
    static constexpr const char* device_name_mouse = "Mouse";
    static constexpr const char* device_name_gamepad = "Gamepad";
    static constexpr const char* label_press_keys = "Press keys...";
    static constexpr const char* label_other = "Other";
    static constexpr const char* label_reset_defaults = "Reset Defaults";
    static constexpr const char* label_save = "Save";

    static constexpr const char* label_keybind_info = "Keybind Info";
    static constexpr const char* label_click_to_rebind = "Click any keybind to rebind it.";
    static constexpr const char* label_right_click_remove = "Right-click to remove a binding.";
    static constexpr const char* label_esc_to_cancel = "Press ESC to cancel rebinding.";

    static constexpr int wnd_flags = gui::window_flags_no_move | gui::window_flags_no_collapse | gui::window_flags_no_resize | gui::window_flags_no_title_bar;
    static constexpr fan::color title_color = fan::color::from_rgba(0x948c80ff) * 1.5f;

    struct device_bindings_t {
      std::vector<combo_t> keyboard;
      std::vector<combo_t> mouse;
      std::vector<combo_t> gamepad;
    };

    struct key_capture_state_t {
      bool active = false;
      bool capturing = false;
      combo_t current_combo;
    };

    void categorize_combo_into_bindings(device_bindings_t& bindings, const combo_t& combo);
    static fan::device_type_e get_combo_device_type(const combo_t& combo);
    void update_device_binding(device_bindings_t& bindings, fan::device_type_e device, const combo_t& combo);

    void begin_capture();
    void finish_capture(const std::string& action_name, fan::device_type_e device);
    void cancel_capture();
    bool is_capturing() const;

    void render_input_button(
      const std::string& action_name,
      int listening_index,
      fan::device_type_e device,
      const combo_t& combo
    );

    f32_t calc_button_width(const std::string& label);
    void render_action_row(int base_listening_index, const std::string& action_name);

    static void menu_keybinds_left(
      keybind_menu_t* menu,
      const fan::vec2& next_window_position,
      const fan::vec2& next_window_size
    );

    static void menu_keybinds_right(
      keybind_menu_t* menu,
      const fan::vec2& next_window_position,
      const fan::vec2& next_window_size
    );

    void mark_dirty();
    void update();
    bool should_suppress_input() const;

    void sync_from_input_action();
    void update_input_action(const std::string& action_name);

    void add_bindings_to_json(
      fan::json& binds_arr,
      std::set<std::string>& unique_combos,
      const std::vector<combo_t>& combos
    ) const;

    void load_from_settings_json(const fan::json& j);
    void save_to_settings_json(fan::json& j) const;
    void refresh_input_actions();

    std::unordered_map<std::string, device_bindings_t> device_bindings;
    key_capture_state_t key_cap_state;
    std::vector<bool> listening_states;
    std::function<void()> reset_keybinds_cb = [] {};
    int suppress_input_frame = 0;
  };
}

#endif