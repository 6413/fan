module;

#include <fan/utility.h>

module fan.graphics.gui.keybinds_menu;

import fan.graphics.loco;
import fan.graphics.gui.settings_menu;

namespace fan::graphics::gui {
  void keybind_menu_t::categorize_combo_into_bindings(device_bindings_t& bindings, const combo_t& combo) {
    if (combo.empty()) return;

    int device = get_combo_device_type(combo);
    switch (device) {
    case fan::device_keyboard: bindings.keyboard.push_back(combo); break;
    case fan::device_mouse: bindings.mouse.push_back(combo); break;
    case fan::device_gamepad: bindings.gamepad.push_back(combo); break;
    }
  }

  fan::device_type_e keybind_menu_t::get_combo_device_type(const combo_t& combo) {
    for (int key : combo) {
      if (fan::is_keyboard_key(key)) return fan::device_keyboard;
      if (fan::is_mouse_button(key)) return fan::device_mouse;
      if (fan::is_gamepad_button(key)) return fan::device_gamepad;
    }
    return fan::device_keyboard;
  }

  void keybind_menu_t::update_device_binding(device_bindings_t& bindings, fan::device_type_e device, const combo_t& combo) {
    std::vector<combo_t>* target = nullptr;
    switch (device) {
    case fan::device_keyboard: target = &bindings.keyboard; break;
    case fan::device_mouse: target = &bindings.mouse; break;
    case fan::device_gamepad: target = &bindings.gamepad; break;
    }
    if (target) {
      if (target->empty()) target->push_back(combo);
      else (*target)[0] = combo;
    }
  }

  void keybind_menu_t::begin_capture() {
    key_cap_state.active = true;
    key_cap_state.capturing = true;
    key_cap_state.current_combo.clear();
  }

  void keybind_menu_t::finish_capture(const std::string& action_name, fan::device_type_e device) {
    key_cap_state.active = false;
    key_cap_state.capturing = false;

    if (key_cap_state.current_combo.empty()) return;

    fan::device_type_e combo_device = get_combo_device_type(key_cap_state.current_combo);
    if (combo_device != device) return;

    auto it = device_bindings.find(action_name);
    if (it == device_bindings.end()) return;

    update_device_binding(it->second, device, key_cap_state.current_combo);
    update_input_action(action_name);
    mark_dirty();
    suppress_input_frame = 2;
  }

  void keybind_menu_t::cancel_capture() {
    key_cap_state = {};
  }

  bool keybind_menu_t::is_capturing() const {
    return key_cap_state.active;
  }

  void keybind_menu_t::render_input_button(
    const std::string& action_name,
    int listening_index,
    fan::device_type_e device,
    const combo_t& combo
  ) {
    static constexpr f32_t key_button_height = 26.f;

    bool is_listening = listening_states[listening_index];
    std::string label = is_listening ? label_press_keys : gloco()->input_action.combo_to_string(combo);

    fan::vec2 cell_pad = gui::get_style().CellPadding;
    fan::vec2 cursor = gui::get_cursor_screen_pos();
    f32_t avail_w = gui::get_content_region_avail().x;

    fan::vec2 pmin = cursor - cell_pad;
    fan::vec2 pmax = cursor + fan::vec2(avail_w, key_button_height) + cell_pad;

    gui::rect_t full_rect(
      fan::vec2(pmin.x, pmin.y),
      fan::vec2(pmax.x, pmax.y)
    );

    auto* window = gui::get_current_window();
    auto id = window->GetID(("##" + action_name + "_" + std::to_string(listening_index)).c_str());
    gui::item_add(full_rect, id);

    bool hovered = false;
    bool held = false;
    bool pressed = gui::button_behavior(full_rect, id, &hovered, &held);

    gui::dummy(fan::vec2(avail_w, key_button_height));

    if (hovered) {
      auto* dl = gui::get_window_draw_list();
      dl->AddRect(
        full_rect.Min,
        full_rect.Max,
        fan::color(0.f, 1.f, 1.f, 1.f).get_gui_color(),
        0.f,
        1.5f
      );
    }

    fan::vec2 text_size = gui::calc_text_size(label);

    fan::vec2 text_pos(
      cursor.x + (avail_w - text_size.x) * 0.5f,
      cursor.y + (key_button_height - text_size.y) * 0.5f
    );

    text_pos.x = std::floor(text_pos.x);
    text_pos.y = std::floor(text_pos.y);

    gui::set_cursor_screen_pos(text_pos);
    gui::text(label.c_str());

    if (!is_listening && pressed) {
      std::fill(listening_states.begin(), listening_states.end(), false);
      listening_states[listening_index] = true;
      key_cap_state = {};
    }

    if (hovered && fan::window::is_mouse_clicked(fan::mouse_right)) {
      auto it = device_bindings.find(action_name);
      if (it != device_bindings.end()) {
        update_device_binding(it->second, device, combo_t {});
        update_input_action(action_name);
        mark_dirty();
      }
    }

    if (is_listening) {
      if (device != fan::device_keyboard &&
        fan::window::is_key_pressed(fan::key_escape) &&
        key_cap_state.current_combo.empty()) {
        cancel_capture();
        listening_states[listening_index] = false;
        return;
      }

      if (!key_cap_state.active) {
        begin_capture();
        suppress_input_frame = 1;
        return;
      }

      if (suppress_input_frame > 0) {
        return;
      }

      combo_t current = gloco()->input_action.get_current_combo(device);
      if (!current.empty()) {
        if (key_cap_state.current_combo.empty() ||
          current.size() > key_cap_state.current_combo.size()) {
          key_cap_state.current_combo = current;
        }
        if (device == fan::device_mouse) {
          finish_capture(action_name, device);
          listening_states[listening_index] = false;
        }
        return;
      }

      if ((device == fan::device_keyboard || device == fan::device_gamepad) &&
        !key_cap_state.current_combo.empty()) {
        finish_capture(action_name, device);
        listening_states[listening_index] = false;
      }
    }
  }

  f32_t keybind_menu_t::calc_button_width(const std::string& label) {
    f32_t text_width = gui::calc_text_size(label).x;
    return std::max(100.f, text_width + 20.f);
  }

  void keybind_menu_t::render_action_row(int base_listening_index, const std::string& action_name) {
    auto it = device_bindings.find(action_name);
    if (it == device_bindings.end()) return;

    auto& bindings = it->second;

    static constexpr f32_t key_button_height = 26.f;

    gui::table_next_row(gui::table_row_flags_none, gui::get_style().CellPadding.y * 2.f);
    gui::table_next_column();

    f32_t text_height = gui::get_text_line_height();
    f32_t y_offset = (key_button_height - text_height) * 0.5f;

    gui::set_cursor_pos_y(gui::get_cursor_pos_y() + y_offset);
    gui::text(action_name.c_str());

    combo_t keyboard_combo = bindings.keyboard.empty() ? combo_t {} : bindings.keyboard[0];
    combo_t mouse_combo = bindings.mouse.empty() ? combo_t {} : bindings.mouse[0];
    combo_t gamepad_combo = bindings.gamepad.empty() ? combo_t {} : bindings.gamepad[0];

    gui::table_next_column();
    render_input_button(action_name, base_listening_index, fan::device_keyboard, keyboard_combo);

    gui::table_next_column();
    render_input_button(action_name, base_listening_index + 1, fan::device_mouse, mouse_combo);

    gui::table_next_column();
    render_input_button(action_name, base_listening_index + 2, fan::device_gamepad, gamepad_combo);
  }

  void keybind_menu_t::menu_keybinds_left(
    keybind_menu_t* menu,
    const fan::vec2& next_window_position,
    const fan::vec2& next_window_size
  ) {
    gui::push_font(gui::get_font(24));
    gui::set_next_window_pos(next_window_position);
    gui::set_next_window_size(next_window_size);
    gui::set_next_window_bg_alpha(0.99);
    gui::begin("##Keybinds Left", nullptr, wnd_flags);

    if (menu->device_bindings.empty()) {
      menu->refresh_input_actions();
    }

    int num_actions = (int)menu->device_bindings.size();
    int required_size = num_actions * 3;

    if (menu->listening_states.size() != required_size) {
      menu->listening_states.clear();
      menu->listening_states.resize(required_size, false);
    }

    gui::begin_child(
      "##keybinds_scroll_region",
      fan::vec2(0, -70),
      false,
      gui::window_flags_horizontal_scrollbar
    );

    std::map<std::string, std::vector<std::string>> grouped;
    auto& ia = gloco()->input_action;

    for (auto& [action_name, _] : menu->device_bindings) {
      std::string group;
      auto it = ia.action_groups.find(action_name);
      if (it != ia.action_groups.end()) {
        group = it->second;
      }
      grouped[group].push_back(action_name);
    }

    std::vector<std::string> group_order;
    group_order.reserve(grouped.size());
    for (auto& [group_name, actions] : grouped) {
      group_order.push_back(group_name);
      std::sort(actions.begin(), actions.end());
    }

    std::sort(
      group_order.begin(),
      group_order.end(),
      [](const std::string& a, const std::string& b) {
      if (a.empty()) return false;
      if (b.empty()) return true;
      return a < b;
    }
    );

    constexpr f32_t device_col_width = 200.f;

    int base_index = 0;

    for (size_t group_idx = 0; group_idx < group_order.size(); ++group_idx) {
      const auto& group_name = group_order[group_idx];
      const auto& actions = grouped[group_name];

      if (group_idx > 0) {
        gui::dummy(fan::vec2(0.f, 20.f));
      }

      gui::push_font(gui::get_font(28, true));
      std::string display_name = group_name.empty() ? label_other : group_name;
      gui::text(title_color, display_name.c_str());

      if (group_idx == 0) {
        f32_t cell_width = gui::table_get_cell_width(device_col_width);
        f32_t action_width = gui::get_content_region_avail().x - cell_width * 3.f;

        const char* labels[] = {device_name_keyboard, device_name_mouse, device_name_gamepad};

        for (int i = 0; i < 3; i++) {
          gui::same_line();
          f32_t tw = gui::get_text_size(labels[i]).x;
          gui::set_cursor_pos_x(action_width + cell_width * (i + 0.5f) - tw * 0.5f);
          gui::text(title_color, labels[i]);
        }
      }

      gui::pop_font();

      gui::dummy(fan::vec2(0.f, 8.f));

      std::string table_id = "keybinds_table_" + std::to_string(group_idx);

      gui::push_style_var(gui::style_var_cell_padding, fan::vec2(16.f, 12.f));
      gui::push_style_var(gui::style_var_item_spacing, fan::vec2(8.f, 4.f));
      gui::push_style_color(
        gui::col_table_row_bg,
        fan::color(0.0f, 0.0f, 0.0f, 0.00f)
      );
      gui::push_style_color(
        gui::col_table_row_bg_alt,
        fan::color(1.0f, 1.0f, 1.0f, 0.06f)
      );

      if (gui::begin_table(
        table_id.c_str(),
        4,
        gui::table_flags_row_bg |
        gui::table_flags_borders_inner_h |
        gui::table_flags_borders_outer_h |
        gui::table_flags_borders_outer_v |
        gui::table_flags_borders_inner_v |
        gui::table_flags_no_clip
      )) {
        gui::table_setup_column(label_action, gui::table_column_flags_width_stretch);
        gui::table_setup_column(device_name_keyboard, gui::table_column_flags_width_fixed, device_col_width);
        gui::table_setup_column(device_name_mouse, gui::table_column_flags_width_fixed, device_col_width);
        gui::table_setup_column(device_name_gamepad, gui::table_column_flags_width_fixed, device_col_width);

        for (const auto& action_name : actions) {
          menu->render_action_row(base_index, action_name);
          base_index += 3;
        }

        gui::end_table();
      }

      gui::pop_style_color(2);
      gui::pop_style_var(2);
    }

    gui::end_child();

    gui::dummy(fan::vec2(0.f, 5.f));
    gui::push_style_var(gui::style_var_frame_padding, fan::vec2(15.f, 10.f));

    if (gui::button(label_reset_defaults, fan::vec2(190.f, 40.f))) {
      menu->reset_keybinds_cb();
      menu->mark_dirty();
    }

    gui::same_line();
    gui::dummy(fan::vec2(10.f, 0.f));
    gui::same_line();

    if (gui::button(label_save, fan::vec2(100.f, 40.f))) {
      OFFSETLESS(menu, fan::graphics::gui::settings_menu_t, keybind_menu)->config.save();
    }

    gui::pop_style_var();
    gui::end();
    gui::pop_font();
  }

  void keybind_menu_t::menu_keybinds_right(keybind_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
    gui::set_next_window_pos(next_window_position);
    gui::set_next_window_size(next_window_size);
    gui::set_next_window_bg_alpha(0.99);
    gui::begin("##Keybinds Right", nullptr, wnd_flags);
    gui::push_font(gui::get_font(32, true));
    gui::text(title_color, label_keybind_info);
    gui::pop_font();
    gui::new_line();
    gui::text_wrapped(label_click_to_rebind);
    gui::text_wrapped(label_right_click_remove);
    gui::text_wrapped(label_esc_to_cancel);
    gui::end();
  }

  void keybind_menu_t::mark_dirty() {
    OFFSETLESS(this, settings_menu_t, keybind_menu)->mark_dirty();
  }

  void keybind_menu_t::update() {
    if (suppress_input_frame > 0) {
      suppress_input_frame--;
    }
  }

  bool keybind_menu_t::should_suppress_input() const {
    return suppress_input_frame > 0;
  }

  void keybind_menu_t::sync_from_input_action() {
    auto& actions = gloco()->input_action.input_actions;

    for (auto& [action_name, action_data] : actions) {
      if (device_bindings.find(action_name) != device_bindings.end()) continue;

      device_bindings_t bindings;
      for (const auto& chord : action_data.keybinds) {
        if (chord.empty() || chord[0].empty()) continue;
        categorize_combo_into_bindings(bindings, chord[0]);
      }
      device_bindings[action_name] = bindings;
    }
  }

  void keybind_menu_t::update_input_action(const std::string& action_name) {
    auto& actions = gloco()->input_action.input_actions;
    auto it = device_bindings.find(action_name);
    if (it == device_bindings.end()) return;

    auto& bindings = it->second;
    auto& action_data = actions[action_name];
    action_data.keybinds.clear();

    for (const auto& combo : bindings.keyboard) {
      fan::window::input_action_t::chord_t chord;
      chord.push_back(combo);
      action_data.keybinds.push_back(chord);
    }
    for (const auto& combo : bindings.mouse) {
      fan::window::input_action_t::chord_t chord;
      chord.push_back(combo);
      action_data.keybinds.push_back(chord);
    }
    for (const auto& combo : bindings.gamepad) {
      fan::window::input_action_t::chord_t chord;
      chord.push_back(combo);
      action_data.keybinds.push_back(chord);
    }
  }

  void keybind_menu_t::add_bindings_to_json(
    fan::json& binds_arr,
    std::set<std::string>& unique_combos,
    const std::vector<combo_t>& combos
  ) const {
    for (const combo_t& combo : combos) {
      std::string combo_str = gloco()->input_action.combo_to_string(combo);
      if (!combo_str.empty() && !fan::iequals(combo_str, "none")) {
        if (unique_combos.insert(combo_str).second) {
          binds_arr.push_back(combo_str);
        }
      }
    }
  }

  void keybind_menu_t::load_from_settings_json(const fan::json& j) {
    if (!j.contains("keybinds")) return;
    const auto& kb = j["keybinds"];
    if (!kb.is_object()) return;

    for (auto it = kb.begin(); it != kb.end(); ++it) {
      std::string action_name = it.key();
      const fan::json& binds_arr = it.value();
      if (!binds_arr.is_array()) continue;

      device_bindings_t bindings;
      for (const auto& combo_str : binds_arr) {
        if (!combo_str.is_string()) continue;
        combo_t combo = gloco()->input_action.combo_from_string((std::string)combo_str);
        categorize_combo_into_bindings(bindings, combo);
      }

      device_bindings[action_name] = bindings;
      update_input_action(action_name);
    }
  }

  void keybind_menu_t::save_to_settings_json(fan::json& j) const {
    fan::json keybinds_json = fan::json::object();

    for (const auto& [action_name, bindings] : device_bindings) {
      std::set<std::string> unique_combos;
      fan::json binds_arr = fan::json::array();

      add_bindings_to_json(binds_arr, unique_combos, bindings.keyboard);
      add_bindings_to_json(binds_arr, unique_combos, bindings.mouse);
      add_bindings_to_json(binds_arr, unique_combos, bindings.gamepad);

      if (!binds_arr.empty()) {
        keybinds_json[action_name] = binds_arr;
      }
    }

    j["keybinds"] = keybinds_json;
  }

  void keybind_menu_t::refresh_input_actions() {
    sync_from_input_action();
    listening_states.clear();
    int num_actions = (int)device_bindings.size();
    listening_states.resize(num_actions * 3, false);
  }
}