module;

#include <fan/utility.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <cstring>
#include <functional>

export module fan.window.input_action;

export import fan.window.input;

export namespace fan::window {
  struct input_action_t {
    enum {
      none = -1,
      release = (int)fan::keyboard_state::release,
      press = (int)fan::keyboard_state::press,
      repeat = (int)fan::keyboard_state::repeat,
      press_or_repeat
    };

    struct action_data_t {
      static constexpr int max_keys_per_action = 5;
      int keys[max_keys_per_action]{};
      uint8_t count = 0;
      static constexpr int max_keys_combos = 5;
      int key_combos[max_keys_combos]{};
      uint8_t combo_count = 0;
    };

    void add(const int* keys, std::size_t count, const std::string& action_name) {
      action_data_t action_data;
      action_data.count = (uint8_t)count;
      std::memcpy(action_data.keys, keys, sizeof(int) * count);
      input_actions[action_name] = action_data;
    }
    void add(int key, const std::string& action_name) {
      add(&key, 1, action_name);
    }
    void add(std::initializer_list<int> keys, const std::string& action_name) {
      add(keys.begin(), keys.size(), action_name);
    }

    void edit(int key, const std::string& action_name) {
      auto found = input_actions.find(action_name);
      if (found == input_actions.end()) {
        fan::throw_error_impl("trying to modify non existing action");
      }
      std::memset(found->second.keys, 0, sizeof(found->second.keys));
      found->second.keys[0] = key;
      found->second.count = 1;
      found->second.combo_count = 0;
    }

    void add_keycombo(std::initializer_list<int> keys, const std::string& action_name) {
      action_data_t action_data;
      action_data.combo_count = (uint8_t)keys.size();
      std::memcpy(action_data.key_combos, keys.begin(), sizeof(int) * action_data.combo_count);
      input_actions[action_name] = action_data;
    }

    bool is_active(const std::string& action_name, int pstate = input_action_t::press) {
      auto found = input_actions.find(action_name);
      if (found != input_actions.end()) {
        action_data_t& action_data = found->second;

        if (action_data.combo_count) {
          int state = none;
          for (int i = 0; i < action_data.combo_count; ++i) {
            int s = is_active_func(action_data.key_combos[i]);
            if (s == none) {
              return none == input_action_t::press;
            }
            if (state == input_action_t::press && s == input_action_t::repeat) {
              state = 1;
            }
            if (state == input_action_t::press_or_repeat) {
              if (state == input_action_t::press && s == input_action_t::repeat) {
              }
            }
            else {
              state = s;
            }
          }
          if (pstate == input_action_t::press_or_repeat) {
            return state == input_action_t::press ||
              state == input_action_t::repeat;
          }
          return state == pstate;
        }
        else if (action_data.count) {
          int state = none;
          for (int i = 0; i < action_data.count; ++i) {
            int s = is_active_func(action_data.keys[i]);
            if (s != none) {
              state = s;
            }
          }
          if (pstate == input_action_t::press_or_repeat) {
            return state == input_action_t::press ||
              state == input_action_t::repeat;
          }
          return state == pstate;
        }
      }
      return none == pstate;
    }
    std::function<int(int key)> is_active_func;

    bool is_action_clicked(const std::string& action_name) {
      return is_active(action_name);
    }
    bool is_action_down(const std::string& action_name) {
      return is_active(action_name, press_or_repeat);
    }
    bool exists(const std::string& action_name) {
      return input_actions.find(action_name) != input_actions.end();
    }
    void insert_or_assign(int key, const std::string& action_name) {
      action_data_t action_data;
      action_data.count = (uint8_t)1;
      std::memcpy(action_data.keys, &key, sizeof(int) * 1);
      input_actions.insert_or_assign(action_name, action_data);
    }
    void remove(const std::string& action_name) {
      input_actions.erase(action_name);
    }

    std::unordered_map<std::string, action_data_t> input_actions;
  };
}