module;

#include <fan/utility.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <cstring>
#include <functional>
#include <vector>
#include <algorithm>

export module fan.window.input_action;

export import fan.window.input;
export import fan.window;

export namespace fan::window {
  struct input_action_t {
    enum {
      none = -1,
      release = (int)fan::keyboard_state::release,
      press = (int)fan::keyboard_state::press,
      repeat = (int)fan::keyboard_state::repeat,
      press_or_repeat
    };

    using combo_t = std::vector<key_code_t>; // simultaneous keys
    using chord_t = std::vector<combo_t>; // sequence of combos

    struct action_data_t {
      std::vector<chord_t> keybinds; // multiple keybinds per action
    };

    std::unordered_map<std::string, action_data_t> input_actions;
    std::unordered_map<std::string, std::string> action_groups;

    fan::window_t* window = nullptr;

    // ---- internal helpers ----

    int eval_combo(const combo_t& combo) {
      if (combo.empty()) {
        return none;
      }

      int min_state = repeat;
      for (int key : combo) {
        int s = window->key_state(key);
        if (s != press && s != repeat) {
          return none;
        }
        if (s < min_state) {
          min_state = s;
        }
      }
      return min_state;
    }

    int eval_chord(const chord_t& chord) {
      if (chord.empty()) {
        return none;
      }

      return eval_combo(chord[0]);
    }

    void add(const int* keys, std::size_t count, const std::string& action_name) {
      if (count == 0) {
        return;
      }

      action_data_t& action_data = input_actions[action_name];

      for (std::size_t i = 0; i < count; ++i) {
        int key = keys[i];
        chord_t chord;
        chord.push_back(combo_t{ key });
        action_data.keybinds.push_back(chord);
      }
    }

    void add(int key, const std::string& action_name) {
      add(&key, 1, action_name);
    }

    void add(std::initializer_list<int> keys, const std::string& action_name) {
      add(keys.begin(), keys.size(), action_name);
    }

    void edit(int key, const std::string& action_name) {
      action_data_t& action_data = input_actions[action_name];

      chord_t chord;
      chord.push_back(combo_t{ key });

      action_data.keybinds.clear();
      action_data.keybinds.push_back(chord);
    }

    void add_keycombo(std::initializer_list<int> keys, const std::string& action_name) {
      std::vector<int> v(keys.begin(), keys.end());
      add_keycombo(v, action_name);
    }

    void add_keycombo(const std::vector<int>& keys, const std::string& action_name) {
      if (keys.empty()) {
        return;
      }

      combo_t combo = keys;
      std::sort(combo.begin(), combo.end());
      combo.erase(std::unique(combo.begin(), combo.end()), combo.end());

      chord_t chord;
      chord.push_back(combo);

      input_actions[action_name].keybinds.push_back(chord);
    }

    bool is_active(const std::string& action_name, int pstate = input_action_t::press) {
      auto found = input_actions.find(action_name);
      if (found == input_actions.end()) {
        return pstate == input_action_t::none;
      }

      action_data_t& action_data = found->second;

      int best_state = none;

      for (const chord_t& chord : action_data.keybinds) {
        int s = eval_chord(chord);
        if (s > best_state) {
          best_state = s;
        }
      }

      switch (pstate) {
      case input_action_t::press:
        return best_state == input_action_t::press;
      case input_action_t::repeat:
        return best_state == input_action_t::repeat;
      case input_action_t::press_or_repeat:
        return best_state == input_action_t::press || best_state == input_action_t::repeat;
      case input_action_t::none:
        return best_state == input_action_t::none;
      default:
        break;
      }
      return false;
    }

    bool is_clicked(const std::string& action_name) {
      return is_active(action_name, input_action_t::press);
    }

    bool is_down(const std::string& action_name) {
      return is_active(action_name, input_action_t::press_or_repeat);
    }

    bool exists(const std::string& action_name) {
      return input_actions.find(action_name) != input_actions.end();
    }

    void insert_or_assign(std::initializer_list<int> keys, const std::string& action_name, const std::string& group_name = "") {
      for (int key : keys) {
        insert_or_assign(key, action_name, group_name);
      }
    }

    void insert_or_assign(int key, const std::string& action_name, const std::string& group_name = "") {
      if (!group_name.empty()) {
        action_groups[action_name] = group_name;
      }

      combo_t combo {key};
      chord_t target;
      target.push_back(combo);

      action_data_t& action_data = input_actions[action_name];

      for (const chord_t& chord : action_data.keybinds) {
        if (chord.size() == 1 && chord[0] == combo) {
          return;
        }
      }

      action_data.keybinds.push_back(target);
    }

    void insert_or_assign_combo(std::initializer_list<int> keys,
      const std::string& action_name,
      const std::string& group_name = "") {
      if (!group_name.empty()) {
        action_groups[action_name] = group_name;
      }

      combo_t combo(keys.begin(), keys.end());
      std::sort(combo.begin(), combo.end());
      combo.erase(std::unique(combo.begin(), combo.end()), combo.end());

      chord_t target;
      target.push_back(combo);

      action_data_t& action_data = input_actions[action_name];

      for (const chord_t& chord : action_data.keybinds) {
        if (chord.size() == 1 && chord[0] == combo) {
          return;
        }
      }

      action_data.keybinds.push_back(target);
    }

    void add_empty_keybind(const std::string& action_name) {
      action_data_t& action_data = input_actions[action_name];
      chord_t chord;
      action_data.keybinds.push_back(chord);
    }

    void remove_keybind(const std::string& action_name, int index) {
      auto it = input_actions.find(action_name);
      if (it == input_actions.end()) {
        return;
      }

      auto& keybinds = it->second.keybinds;
      if (index < 0 || index >= (int)keybinds.size()) {
        return;
      }

      keybinds.erase(keybinds.begin() + index);
    }

    void remove(const std::string& action_name) {
      input_actions.erase(action_name);
    }

    std::vector<int> get_all_keys(const std::string& action_name) const {
      std::vector<int> result;

      auto it = input_actions.find(action_name);
      if (it == input_actions.end()) {
        return result;
      }

      const auto& keybinds = it->second.keybinds;

      for (const auto& chord : keybinds) {
        if (chord.empty()) {
          continue;
        }

        const auto& combo = chord[0];

        for (int key : combo) {
          result.push_back(key);
        }
      }

      return result;
    }

    int get_first_gamepad_key(const std::string& action_name) const {
      auto it = input_actions.find(action_name);
      if (it == input_actions.end()) {
        return -1;
      }

      const auto& keybinds = it->second.keybinds;

      for (const auto& chord : keybinds) {
        if (chord.empty()) {
          continue;
        }

        const auto& combo = chord[0];

        for (int key : combo) {
          if (key >= fan::gamepad_a && key <= fan::gamepad_last) {
            return key;
          }
        }
      }

      return -1;
    }

    static fan::device_type_e get_combo_device_type(const combo_t& combo) {
      for (int key : combo) {
        if (fan::is_keyboard_key(key)) return fan::device_keyboard;
        if (fan::is_mouse_button(key)) return fan::device_mouse;
        if (fan::is_gamepad_button(key)) return fan::device_gamepad;
      }
      return fan::device_keyboard;
    }
    combo_t get_current_combo(fan::device_type_e device) {
      combo_t combo;
      combo.reserve(8);

      if (device == fan::device_keyboard) {
        for (int key = fan::key_first; key <= fan::input_last; ++key) {
          if (window->is_key_down(key)) {
            combo.push_back(key);
          }
        }
      }

      if (device == fan::device_mouse) {
        for (int btn = fan::mouse_first; btn <= fan::mouse_last; ++btn) {
          if (window->is_key_down(btn)) {
            combo.push_back(btn);
          }
        }
      }

      if (device == fan::device_gamepad) {
        for (int btn = fan::gamepad_a; btn <= fan::gamepad_last; ++btn) {
          if (window->is_gamepad_button_down(btn) || 
              window->is_gamepad_axis_active(btn)) {
            combo.push_back(btn);
          }
        }
      }

      std::sort(combo.begin(), combo.end());
      combo.erase(std::unique(combo.begin(), combo.end()), combo.end());
      
      return combo;
    }
    using combo_t = std::vector<key_code_t>;
    void sort_combo(combo_t& combo) {
      std::sort(combo.begin(), combo.end(), [](int a, int b) {
        bool ma = fan::is_modifier(a), mb = fan::is_modifier(b);
        if (ma != mb) return ma > mb;
        return a < b;
      });
      combo.erase(std::unique(combo.begin(), combo.end()), combo.end());
    }

    combo_t combo_from_string(const std::string& combo_str) {
      combo_t combo;
      std::string trimmed = fan::trim(combo_str);
      if (trimmed.empty()) return combo;

      size_t start = 0;
      while (start <= trimmed.size()) {
        size_t end = trimmed.find('+', start);
        std::string token = (end == std::string::npos)
          ? trimmed.substr(start)
          : trimmed.substr(start, end - start);
        start = (end == std::string::npos) ? trimmed.size() + 1 : end + 1;

        token = fan::trim(token);
        if (token.empty()) continue;

        int key = fan::key_name_to_code(token);
        if (key != -1) combo.push_back(key);
      }

      sort_combo(combo);
      return combo;
    }

    std::string combo_to_string(const combo_t& combo) {
      if (combo.empty()) return "None";

      combo_t sorted = combo;
      sort_combo(sorted);

      std::string s;
      bool first = true;
      for (int key : sorted) {
        std::string name = fan::key_code_to_name(key);
        if (fan::iequals(name, "unknown")) continue;
        if (!first) s += " + ";
        s += name;
        first = false;
      }
      return s.empty() ? "None" : s;
    }
  };

  inline constexpr auto device_keyboard = fan::device_keyboard;
  inline constexpr auto device_mouse = fan::device_mouse;
  inline constexpr auto device_gamepad = fan::device_gamepad;
}