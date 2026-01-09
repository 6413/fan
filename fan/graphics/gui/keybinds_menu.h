struct keybind_config_t {
  using key_code_t = int;
  using combo_t = std::vector<key_code_t>;

  static std::string to_lower(const std::string& s) {
    std::string r;
    r.reserve(s.size());
    for (unsigned char c : s) {
      r.push_back((char)std::tolower(c));
    }
    return r;
  }
  static std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) ++b;
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
  }
  static bool iequals(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i]))
        return false;
    }
    return true;
  }
  static bool is_modifier(int key) {
    return
      key == fan::key_left_control ||
      key == fan::key_right_control ||
      key == fan::key_left_shift ||
      key == fan::key_right_shift ||
      key == fan::key_left_alt ||
      key == fan::key_right_alt ||
      key == fan::key_left_super ||
      key == fan::key_right_super;
  }
  static void sort_combo(combo_t& combo) {
    std::sort(combo.begin(), combo.end(), [](int a, int b) {
      bool ma = is_modifier(a), mb = is_modifier(b);
      if (ma != mb) return ma > mb;
      return a < b;
    });
    combo.erase(std::unique(combo.begin(), combo.end()), combo.end());
  }
  static bool is_keyboard_key(int key) {
    return key >= fan::key_first && key <= fan::key_last;
  }
  static bool is_mouse_button(int key) {
    return key >= fan::mouse_first && key <= fan::mouse_last;
  }
  static bool is_gamepad_button(int key) {
    return key >= fan::gamepad_first && key <= fan::gamepad_last;
  }
  static int key_name_to_code(const std::string& name) {
    std::string trimmed = trim(name);
    if (trimmed.empty()) return -1;
    std::string lowered = to_lower(trimmed);

    if (iequals(lowered, "none") || iequals(lowered, "unknown")) {
      return -1;
    }

    for (int key = fan::mouse_first; key <= fan::mouse_last; ++key) {
      const char* mn = fan::get_mouse_name(key);
      if (!mn) continue;
      if (iequals(lowered, to_lower(mn))) {
        return key;
      }
    }

    for (int key = fan::gamepad_a; key <= fan::gamepad_last; ++key) {
      const char* gn = fan::get_key_name(key);
      if (!gn) continue;
      if (iequals(lowered, to_lower(gn))) {
        return key;
      }
    }

    for (int key = fan::key_first; key <= fan::key_last; ++key) {
      const char* kn = fan::get_key_name(key);
      if (!kn) continue;
      if (iequals(lowered, to_lower(kn))) {
        return key;
      }
    }

    return -1;
  }
  static std::string key_code_to_name(int key) {
    if (key >= fan::gamepad_a && key <= fan::gamepad_last) {
      const char* gn = fan::get_key_name(key);
      if (gn && std::strcmp(gn, "Unknown") != 0) {
        return std::string(gn);
      }
      return "Unknown";
    }

    if (key >= fan::mouse_first && key <= fan::mouse_last) {
      const char* mn = fan::get_mouse_name(key);
      if (mn && std::strcmp(mn, "Unknown") != 0) {
        return std::string(mn);
      }
    }

    const char* kn = fan::get_key_name(key);
    if (!kn) return "Unknown";
    if (std::strcmp(kn, "Unknown") == 0) return "Unknown";
    return std::string(kn);
  }
  static combo_t combo_from_string(const std::string& combo_str) {
    combo_t combo;
    std::string trimmed = trim(combo_str);
    if (trimmed.empty()) return combo;

    size_t start = 0;
    while (start <= trimmed.size()) {
      size_t end = trimmed.find('+', start);
      std::string token = (end == std::string::npos)
        ? trimmed.substr(start)
        : trimmed.substr(start, end - start);
      start = (end == std::string::npos) ? trimmed.size() + 1 : end + 1;

      token = trim(token);
      if (token.empty()) continue;

      int key = key_name_to_code(token);
      if (key != -1) combo.push_back(key);
    }

    sort_combo(combo);
    return combo;
  }
  static std::string combo_to_string(const combo_t& combo) {
    if (combo.empty()) return "None";

    combo_t sorted = combo;
    sort_combo(sorted);

    std::string s;
    bool first = true;
    for (int key : sorted) {
      std::string name = key_code_to_name(key);
      if (iequals(name, "unknown")) continue;
      if (!first) s += " + ";
      s += name;
      first = false;
    }
    return s.empty() ? "None" : s;
  }
};

struct keybind_menu_t {
  using key_code_t = keybind_config_t::key_code_t;
  using combo_t = keybind_config_t::combo_t;

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

  enum device_type_e {
    device_keyboard,
    device_mouse,
    device_gamepad
  };

  static constexpr int first_key = fan::key_first;
  static constexpr int last_key = fan::input_last;

  static bool is_gamepad_button_down(int key) {
    int jid = 0;
    if (!glfwJoystickPresent(jid)) return false;

    int count;
    const unsigned char* buttons = glfwGetJoystickButtons(jid, &count);
    if (!buttons) return false;

    int idx = key - fan::gamepad_a;
    if (idx < 0 || idx >= count) return false;

    return buttons[idx] == GLFW_PRESS;
  }
  static bool is_gamepad_axis_active(int key) {
    fan::vec2 axis = gloco()->window.get_gamepad_axis(key);

    if (key == fan::gamepad_l2 || key == fan::gamepad_r2) {
      return axis.x > gloco()->window.gamepad_axis_deadzone;
    }

    return axis.length() > gloco()->window.gamepad_axis_deadzone;
  }
  static combo_t get_current_combo() {
    combo_t combo;
    combo.reserve(8);

    for (int key = first_key; key <= last_key; ++key) {
      if (fan::window::is_key_down(key)) {
        combo.push_back(key);
      }
    }

    for (int btn = fan::mouse_first; btn <= fan::mouse_last; ++btn) {
      if (fan::window::is_key_down(btn)) {
        combo.push_back(btn);
      }
    }

    for (int btn = fan::gamepad_a; btn <= fan::gamepad_last; ++btn) {
      if (is_gamepad_button_down(btn) || is_gamepad_axis_active(btn)) {
        combo.push_back(btn);
      }
    }

    keybind_config_t::sort_combo(combo);
    return combo;
  }
  void begin_capture() {
    key_cap_state.active = true;
    key_cap_state.capturing = true;
    key_cap_state.current_combo.clear();
  }
  void finish_capture(const std::string& action_name, device_type_e device) {
    key_cap_state.active = false;
    key_cap_state.capturing = false;

    if (key_cap_state.current_combo.empty()) return;

    bool has_keyboard = false;
    bool has_mouse = false;
    bool has_gamepad = false;

    for (int key : key_cap_state.current_combo) {
      if (keybind_config_t::is_keyboard_key(key)) has_keyboard = true;
      if (keybind_config_t::is_mouse_button(key)) has_mouse = true;
      if (keybind_config_t::is_gamepad_button(key)) has_gamepad = true;
    }

    if ((device == device_keyboard && !has_keyboard) ||
      (device == device_mouse && !has_mouse) ||
      (device == device_gamepad && !has_gamepad)) {
      return;
    }

    auto it = device_bindings.find(action_name);
    if (it == device_bindings.end()) return;

    auto& bindings = it->second;
    switch (device) {
    case device_keyboard:
      if (bindings.keyboard.empty()) bindings.keyboard.push_back(key_cap_state.current_combo);
      else bindings.keyboard[0] = key_cap_state.current_combo;
      break;
    case device_mouse:
      if (bindings.mouse.empty()) bindings.mouse.push_back(key_cap_state.current_combo);
      else bindings.mouse[0] = key_cap_state.current_combo;
      break;
    case device_gamepad:
      if (bindings.gamepad.empty()) bindings.gamepad.push_back(key_cap_state.current_combo);
      else bindings.gamepad[0] = key_cap_state.current_combo;
      break;
    }

    update_input_action(action_name);
    mark_dirty();
    suppress_input_frame = 2;
  }
  void cancel_capture() {
    key_cap_state = {};
  }
  bool is_capturing() const {
    return key_cap_state.active;
  }
  static std::string combo_to_string(const combo_t& combo) {
    return keybind_config_t::combo_to_string(combo);
  }
  static combo_t get_current_combo(device_type_e device) {
    combo_t combo;
    combo.reserve(8);

    if (device == device_keyboard) {
      for (int key = first_key; key <= last_key; ++key) {
        if (fan::window::is_key_down(key)) {
          combo.push_back(key);
        }
      }
    }

    if (device == device_mouse) {
      for (int btn = fan::mouse_first; btn <= fan::mouse_last; ++btn) {
        if (fan::window::is_key_down(btn)) {
          combo.push_back(btn);
        }
      }
    }

    if (device == device_gamepad) {
      for (int btn = fan::gamepad_a; btn <= fan::gamepad_last; ++btn) {
        if (is_gamepad_button_down(btn) || is_gamepad_axis_active(btn)) {
          combo.push_back(btn);
        }
      }
    }

    keybind_config_t::sort_combo(combo);
    return combo;
  }

  void render_input_button(const std::string& action_name,
    int listening_index,
    device_type_e device,
    const combo_t& combo) {
    static constexpr f32_t key_button_height = 36.f;

    bool is_listening = listening_states[listening_index];

    std::string label = is_listening ? "Press keys..." : combo_to_string(combo);

    fan::vec2 text_size = gui::calc_text_size(label);

    constexpr f32_t horizontal_padding = 10.f;
    f32_t vertical_padding = (key_button_height - text_size.y) * 0.5f;

    f32_t button_width = text_size.x + 2 * horizontal_padding;

    gui::push_style_var(gui::style_var_frame_padding,
      fan::vec2(horizontal_padding, vertical_padding));
    gui::push_style_var(gui::style_var_frame_rounding, 6.f);

    if (is_listening) {
      gui::push_style_color(gui::col_button,
        fan::color(0.85f, 0.55f, 0.25f, 1.0f));
    }

    bool pressed = gui::button(
      (label + "##" + action_name + "_" + std::to_string(listening_index)).c_str(),
      fan::vec2(button_width, key_button_height)
    );

    if (is_listening) {
      gui::pop_style_color();
    }

    if (is_listening) {
      if (device != device_keyboard &&
        fan::window::is_key_pressed(fan::key_escape) &&
        key_cap_state.current_combo.empty()) {
        cancel_capture();
        listening_states[listening_index] = false;
        gui::pop_style_var(2);
        return;
      }
      if (!key_cap_state.active) {
        begin_capture();
        suppress_input_frame = 1;
        gui::pop_style_var(2);
        return;
      }
      if (suppress_input_frame > 0) {
        gui::pop_style_var(2);
        return;
      }
      combo_t current = get_current_combo(device);

      if (!current.empty()) {
        if (key_cap_state.current_combo.empty() ||
          current.size() > key_cap_state.current_combo.size()) {
          key_cap_state.current_combo = current;
        }

        if (device == device_mouse) {
          finish_capture(action_name, device);
          listening_states[listening_index] = false;
        }

        gui::pop_style_var(2);
        return;
      }
      if ((device == device_keyboard || device == device_gamepad) &&
        !key_cap_state.current_combo.empty()) {
        finish_capture(action_name, device);
        listening_states[listening_index] = false;
      }
    }
    else {
      if (pressed) {
        std::fill(listening_states.begin(), listening_states.end(), false);
        listening_states[listening_index] = true;
        key_cap_state = {};
      }

      if (gui::is_item_clicked(fan::mouse_right)) {
        auto it = device_bindings.find(action_name);
        if (it != device_bindings.end()) {
          auto& bindings = it->second;

          switch (device) {
          case device_keyboard:
            if (!bindings.keyboard.empty()) bindings.keyboard[0].clear();
            break;
          case device_mouse:
            if (!bindings.mouse.empty()) bindings.mouse[0].clear();
            break;
          case device_gamepad:
            if (!bindings.gamepad.empty()) bindings.gamepad[0].clear();
            break;
          }

          update_input_action(action_name);
          mark_dirty();
        }
      }
    }


    gui::pop_style_var(2);
  }

  f32_t calc_button_width(const std::string& label) {
    f32_t text_width = gui::calc_text_size(label).x;
    return std::max(100.f, text_width + 20.f);
  }

  void render_action_row(int base_listening_index, const std::string& action_name) {
    auto it = device_bindings.find(action_name);
    if (it == device_bindings.end()) return;

    auto& bindings = it->second;

    gui::table_next_row();
    gui::table_next_column();

    gui::set_cursor_pos_x(gui::get_cursor_pos_x() + 25.f);
    gui::set_cursor_pos_y(gui::get_cursor_pos_y() + 9.f);
    gui::text(action_name.c_str());

    combo_t keyboard_combo = bindings.keyboard.empty() ? combo_t {} : bindings.keyboard[0];
    combo_t mouse_combo = bindings.mouse.empty() ? combo_t {} : bindings.mouse[0];
    combo_t gamepad_combo = bindings.gamepad.empty() ? combo_t {} : bindings.gamepad[0];

    gui::table_next_column();
    render_input_button(action_name, base_listening_index, device_keyboard, keyboard_combo);

    gui::table_next_column();
    render_input_button(action_name, base_listening_index + 1, device_mouse, mouse_combo);

    gui::table_next_column();
    render_input_button(action_name, base_listening_index + 2, device_gamepad, gamepad_combo);
  }

  static void menu_keybinds_left(
    keybind_menu_t* menu,
    const fan::vec2& next_window_position,
    const fan::vec2& next_window_size
  ) {
    gui::push_font(gui::get_font(24));
    gui::set_next_window_pos(next_window_position);
    gui::set_next_window_size(next_window_size);
    gui::set_next_window_bg_alpha(0.99);
    gui::begin("##Keybinds Left", nullptr, wnd_flags);
    gui::text(title_color, "KEYBINDS");

    int num_actions = (int)menu->device_bindings.size();
    int required_size = num_actions * 3;

    if (menu->listening_states.size() != required_size) {
      menu->listening_states.clear();
      menu->listening_states.resize(required_size, false);
    }

    f32_t max_keyboard_width = 0;
    f32_t max_mouse_width = 0;

    for (const auto& [action_name, bindings] : menu->device_bindings) {
      combo_t kb = bindings.keyboard.empty() ? combo_t {} : bindings.keyboard[0];
      combo_t ms = bindings.mouse.empty() ? combo_t {} : bindings.mouse[0];

      std::string kb_label = keybind_menu_t::combo_to_string(kb);
      std::string ms_label = keybind_menu_t::combo_to_string(ms);

      f32_t kb_width = std::max(100.f, gui::calc_text_size(kb_label).x + 20.f);
      f32_t ms_width = std::max(100.f, gui::calc_text_size(ms_label).x + 20.f);

      max_keyboard_width = std::max(max_keyboard_width, kb_width);
      max_mouse_width = std::max(max_mouse_width, ms_width);
    }

    constexpr f32_t spacing = 20.f;

    gui::push_style_color(gui::col_table_header_bg, fan::colors::transparent);

    gui::begin_child("##keybinds_scroll_region", fan::vec2(0, -70), false,
      gui::window_flags_horizontal_scrollbar |
      gui::window_flags_always_horizontal_scrollbar
    );

    gui::push_style_var(gui::style_var_cell_padding, fan::vec2(16.f, 12.f));
    gui::push_style_var(gui::style_var_item_spacing, fan::vec2(8.f, 4.f));
    //gui::push_style_color(gui::col_table_row_bg_alt, fan::color(0.f, 0.f, 0.f, 0.05f));
    gui::push_style_color(gui::col_table_row_bg,     fan::color(0.0f, 0.0f, 0.0f, 0.00f));
    gui::push_style_color(gui::col_table_row_bg_alt, fan::color(1.0f, 1.0f, 1.0f, 0.06f));

    gui::begin_table("keybinds_table", 4,
      gui::table_flags_row_bg |
      gui::table_flags_borders_inner_h |
      gui::table_flags_scroll_x |
      gui::table_flags_no_clip
    );

    f32_t max_action_width = 0.f;

    gui::push_font(gui::get_font(20));
    for (const auto& [action_name, bindings] : menu->device_bindings) {
      f32_t w = gui::calc_text_size(action_name).x;
      max_action_width = std::max(max_action_width, w);
    }
    gui::pop_font();

    constexpr f32_t action_padding = 50.f;
    f32_t action_column_width = max_action_width + action_padding;

    gui::table_setup_column("Control", gui::table_column_flags_width_fixed, action_column_width);
    gui::table_setup_column("Keyboard", gui::table_column_flags_width_fixed, max_keyboard_width);
    gui::table_setup_column("Mouse", gui::table_column_flags_width_fixed, max_mouse_width + spacing);
    gui::table_setup_column("Gamepad", gui::table_column_flags_width_fixed, 200.f);

    gui::table_next_row(gui::table_row_flags_headers, 40.f);

    gui::table_next_column();
    gui::push_font(gui::get_font(28, true));
    gui::set_cursor_pos_x(gui::get_cursor_pos_x() + 12.f);
    gui::text("Control");
    gui::pop_font();

    gui::table_next_column();
    gui::push_font(gui::get_font(28, true));
    gui::text("Keyboard");
    gui::pop_font();

    gui::table_next_column();
    gui::push_font(gui::get_font(28, true));
    gui::text("Mouse");
    gui::pop_font();

    gui::table_next_column();
    gui::push_font(gui::get_font(28, true));
    gui::text("Gamepad");
    gui::pop_font();

    {
      auto* dl = gui::get_window_draw_list();

      fan::vec2 win_pos = gui::get_window_pos();
      fan::vec2 win_size = gui::get_window_size();

      f32_t x = win_pos.x + gui::get_window_content_region_min().x + action_column_width;

      f32_t y_min = win_pos.y;
      f32_t y_max = win_pos.y + win_size.y;

      dl->AddLine(
        {x, y_min},
        {x, y_max},
        fan::color::rgb(255, 255, 255, 40).get_gui_color(),
        1.f
      );
    }

    std::map<std::string, std::vector<std::string>> grouped;

    auto& ia = gloco()->input_action;

    for (auto& [action_name, _] : ia.input_actions) {
      std::string group = "";
      auto it = ia.action_groups.find(action_name);
      if (it != ia.action_groups.end()) {
        group = it->second;
      }
      grouped[group].push_back(action_name);
    }

    int base_index = 0;

    std::vector<std::string> group_order;
    group_order.reserve(grouped.size());

    for (auto& [group_name, _] : grouped) {
      group_order.push_back(group_name);
    }

    std::sort(group_order.begin(), group_order.end(), [](const std::string& a, const std::string& b) {
      if (a.empty()) return false;
      if (b.empty()) return true;
      return a < b;
    });

    for (const auto& group_name : group_order) {
      auto& actions = grouped[group_name];

      if (&group_name != &grouped.begin()->first) {
        gui::table_next_row();
        gui::table_next_column();
        gui::dummy(fan::vec2(0, 12.f));
      }
      gui::table_next_row();
      gui::table_next_column();

      auto* dl = gui::get_window_draw_list();
      fan::vec2 cursor = gui::get_cursor_screen_pos();

      gui::push_font(gui::get_font(28, true));
      std::string display_name = group_name.empty() ? "Other" : group_name;
      fan::vec2 text_size = gui::calc_text_size(display_name);

      fan::vec2 table_min = gui::get_cursor_screen_pos();
      table_min.x -= 12.f;
      table_min.y -= 4.f;

      fan::vec2 content_region = gui::get_content_region_avail();
      fan::vec2 table_max = fan::vec2(
        table_min.x + content_region.x + 24.f,
        table_min.y + text_size.y + 8.f
      );

      gui::set_cursor_pos_x(gui::get_cursor_pos_x() + 12.f);
      gui::text(title_color, display_name.c_str());
      gui::pop_font();

      gui::table_next_column();
      gui::table_next_column();
      gui::table_next_column();

      for (const auto& action_name : actions) {
        menu->render_action_row(base_index, action_name);
        base_index += 3;
      }
    }

    gui::end_table();
    gui::pop_style_color(2);
    gui::pop_style_var(2);
    gui::end_child();
    gui::pop_style_color();

    gui::dummy(fan::vec2(0, 5.f));
    gui::push_style_var(gui::style_var_frame_padding, fan::vec2(15.f, 10.f));
    if (gui::button("Reset Defaults", fan::vec2(190, 40))) {
      menu->reset_keybinds_cb();
      menu->mark_dirty();
    }
    gui::same_line();
    gui::dummy(fan::vec2(10.f, 0));
    gui::same_line();
    if (gui::button("Save", fan::vec2(100, 40))) {
      OFFSETLESS(menu, settings_menu_t, keybind_menu)->config.save();
    }
    gui::pop_style_var();

    gui::end();
    gui::pop_font();
  }


  static void menu_keybinds_right(keybind_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
    gui::set_next_window_pos(next_window_position);
    gui::set_next_window_size(next_window_size);
    gui::set_next_window_bg_alpha(0.99);
    gui::begin("##Keybinds Right", nullptr, wnd_flags);
    gui::push_font(gui::get_font(32, true));
    gui::text(title_color, "Keybind Info");
    gui::pop_font();
    gui::new_line();
    gui::text_wrapped("Click on any keybind to rebind it.");
    gui::text_wrapped("Press ESC to cancel rebinding.");
    gui::end();
  }
  void mark_dirty() {
    OFFSETLESS(this, settings_menu_t, keybind_menu)->mark_dirty();
  }
  void update() {
    if (suppress_input_frame > 0) {
      suppress_input_frame--;
    }
  }
  bool should_suppress_input() const {
    return suppress_input_frame > 0;
  }
  void sync_from_input_action() {
    auto& actions = gloco()->input_action.input_actions;

    for (auto& [action_name, action_data] : actions) {
      if (device_bindings.find(action_name) != device_bindings.end()) {
        continue;
      }

      device_bindings_t bindings;

      for (const auto& chord : action_data.keybinds) {
        if (chord.empty() || chord[0].empty()) continue;

        const auto& combo = chord[0];
        bool has_keyboard = false;
        bool has_mouse = false;
        bool has_gamepad = false;

        for (int key : combo) {
          if (keybind_config_t::is_keyboard_key(key)) has_keyboard = true;
          if (keybind_config_t::is_mouse_button(key)) has_mouse = true;
          if (keybind_config_t::is_gamepad_button(key)) has_gamepad = true;
        }

        if (has_keyboard) {
          bindings.keyboard.push_back(combo);
        }
        else if (has_mouse) {
          bindings.mouse.push_back(combo);
        }
        else if (has_gamepad) {
          bindings.gamepad.push_back(combo);
        }
      }

      device_bindings[action_name] = bindings;
    }
  }
  void update_input_action(const std::string& action_name) {
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
  void load_from_settings_json(const fan::json& j) {
    if (!j.contains("keybinds")) return;
    const auto& kb = j["keybinds"];
    if (!kb.is_object()) return;

    device_bindings.clear();

    for (auto it = kb.begin(); it != kb.end(); ++it) {
      std::string action_name = it.key();
      const fan::json& binds_arr = it.value();

      if (!binds_arr.is_array()) continue;

      device_bindings_t bindings;

      for (const auto& combo_str : binds_arr) {
        if (!combo_str.is_string()) continue;
        combo_t combo = keybind_config_t::combo_from_string((std::string)combo_str);
        if (combo.empty()) continue;

        bool has_keyboard = false;
        bool has_mouse = false;
        bool has_gamepad = false;

        for (int key : combo) {
          if (keybind_config_t::is_keyboard_key(key)) has_keyboard = true;
          if (keybind_config_t::is_mouse_button(key)) has_mouse = true;
          if (keybind_config_t::is_gamepad_button(key)) has_gamepad = true;
        }

        if (has_keyboard) {
          bindings.keyboard.push_back(combo);
        }
        else if (has_mouse) {
          bindings.mouse.push_back(combo);
        }
        else if (has_gamepad) {
          bindings.gamepad.push_back(combo);
        }
      }

      device_bindings[action_name] = bindings;
      update_input_action(action_name);
    }
  }
  void save_to_settings_json(fan::json& j) const {
    fan::json keybinds_json = fan::json::object();

    for (const auto& [action_name, bindings] : device_bindings) {
      std::set<std::string> unique_combos;
      fan::json binds_arr = fan::json::array();

      for (const combo_t& combo : bindings.keyboard) {
        std::string combo_str = keybind_config_t::combo_to_string(combo);
        if (!combo_str.empty() && !keybind_config_t::iequals(combo_str, "none")) {
          if (unique_combos.insert(combo_str).second) {
            binds_arr.push_back(combo_str);
          }
        }
      }

      for (const combo_t& combo : bindings.mouse) {
        std::string combo_str = keybind_config_t::combo_to_string(combo);
        if (!combo_str.empty() && !keybind_config_t::iequals(combo_str, "none")) {
          if (unique_combos.insert(combo_str).second) {
            binds_arr.push_back(combo_str);
          }
        }
      }

      for (const combo_t& combo : bindings.gamepad) {
        std::string combo_str = keybind_config_t::combo_to_string(combo);
        if (!combo_str.empty() && !keybind_config_t::iequals(combo_str, "none")) {
          if (unique_combos.insert(combo_str).second) {
            binds_arr.push_back(combo_str);
          }
        }
      }

      if (!binds_arr.empty()) {
        keybinds_json[action_name] = binds_arr;
      }
    }

    j["keybinds"] = keybinds_json;
  }

  void refresh_input_actions() {
    sync_from_input_action();
    listening_states.clear();
    int num_actions = (int)device_bindings.size();
    listening_states.resize(num_actions * 3, false);
  }

  static constexpr int wnd_flags = gui::window_flags_no_move | gui::window_flags_no_collapse | gui::window_flags_no_resize | gui::window_flags_no_title_bar;
  static constexpr fan::color title_color = fan::color::from_rgba(0x948c80ff) * 1.5f;

  std::unordered_map<std::string, device_bindings_t> device_bindings;
  key_capture_state_t key_cap_state;
  std::vector<bool> listening_states;
  std::function<void()> reset_keybinds_cb = [] {};
  int suppress_input_frame = 0;
};

struct keybind_settings_bridge_t {
  static void menu_left(settings_menu_t* settings_menu, const fan::vec2& pos, const fan::vec2& size) {
    keybind_menu_t::menu_keybinds_left(&OFFSETLESS(settings_menu, settings_menu_t, keybind_menu)->keybind_menu, pos, size);
  }
  static void menu_right(settings_menu_t* settings_menu, const fan::vec2& pos, const fan::vec2& size) {
    keybind_menu_t::menu_keybinds_right(&OFFSETLESS(settings_menu, settings_menu_t, keybind_menu)->keybind_menu, pos, size);
  }
};