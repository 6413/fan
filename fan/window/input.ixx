module;

#if defined (FAN_WINDOW)

#include <fan/utility.h>

#if defined(fan_compiler_gcc)
  // fixes collision with GLFW3 headers while doing import std;
	#ifndef _GCC_MAX_ALIGN_T
		#define _GCC_MAX_ALIGN_T
	#endif
#endif

#ifdef fan_platform_windows
  #define NOMINMAX
  #define WIN32_LEAN_AND_MEAN
  #include <Windows.h>
  #pragma comment(lib, "user32.lib")
#endif
#include <vulkan/vulkan.h>
#if defined(fan_platform_windows)
  #define GLFW_EXPOSE_NATIVE_WIN32
  #define GLFW_EXPOSE_NATIVE_WGL
  #define GLFW_NATIVE_INCLUDE_NONE
#endif
#ifndef GLFW_INCLUDE_NONE
  #define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#endif

export module fan.window.input;

#if defined (FAN_WINDOW)

import std;

export namespace fan {
  using key_code_t = int;

  struct keyboard_state {
    enum {
      release = GLFW_RELEASE,
      press = GLFW_PRESS,
      repeat = GLFW_REPEAT
    };
  };
  using keyboard_state_t = decltype(keyboard_state::release);

  struct mouse_state{
    enum {
      release = GLFW_RELEASE,
      press = GLFW_PRESS,
      repeat = GLFW_REPEAT
    };
  };

  enum input {
    input_first = GLFW_KEY_SPACE,
    key_first = GLFW_KEY_SPACE,
    key_space = GLFW_KEY_SPACE,
    key_0 = GLFW_KEY_0,
    key_1 = GLFW_KEY_1,
    key_2 = GLFW_KEY_2,
    key_3 = GLFW_KEY_3,
    key_4 = GLFW_KEY_4,
    key_5 = GLFW_KEY_5,
    key_6 = GLFW_KEY_6,
    key_7 = GLFW_KEY_7,
    key_8 = GLFW_KEY_8,
    key_9 = GLFW_KEY_9,
    key_apostrophe = GLFW_KEY_APOSTROPHE,
    key_period = GLFW_KEY_PERIOD,
    key_comma = GLFW_KEY_COMMA,
    key_plus = GLFW_KEY_EQUAL,
    key_minus = GLFW_KEY_MINUS,
    key_slash = GLFW_KEY_SLASH,
    key_semicolon = GLFW_KEY_SEMICOLON,
    key_a = GLFW_KEY_A,
    key_b = GLFW_KEY_B,
    key_c = GLFW_KEY_C,
    key_d = GLFW_KEY_D,
    key_e = GLFW_KEY_E,
    key_f = GLFW_KEY_F,
    key_g = GLFW_KEY_G,
    key_h = GLFW_KEY_H,
    key_i = GLFW_KEY_I,
    key_j = GLFW_KEY_J,
    key_k = GLFW_KEY_K,
    key_l = GLFW_KEY_L,
    key_m = GLFW_KEY_M,
    key_n = GLFW_KEY_N,
    key_o = GLFW_KEY_O,
    key_p = GLFW_KEY_P,
    key_q = GLFW_KEY_Q,
    key_r = GLFW_KEY_R,
    key_s = GLFW_KEY_S,
    key_t = GLFW_KEY_T,
    key_u = GLFW_KEY_U,
    key_v = GLFW_KEY_V,
    key_w = GLFW_KEY_W,
    key_x = GLFW_KEY_X,
    key_y = GLFW_KEY_Y,
    key_z = GLFW_KEY_Z,
    key_left_bracket = GLFW_KEY_LEFT_BRACKET,
    key_backslash = GLFW_KEY_BACKSLASH,
    key_right_bracket = GLFW_KEY_RIGHT_BRACKET,
    key_grave_accent = GLFW_KEY_GRAVE_ACCENT,
    key_less_than = GLFW_KEY_WORLD_1,
    key_greater_than = GLFW_KEY_WORLD_2,
    key_escape = GLFW_KEY_ESCAPE,
    key_enter = GLFW_KEY_ENTER,
    key_tab = GLFW_KEY_TAB,
    key_backspace = GLFW_KEY_BACKSPACE,
    key_insert = GLFW_KEY_INSERT,
    key_delete = GLFW_KEY_DELETE,
    key_right = GLFW_KEY_RIGHT,
    key_left = GLFW_KEY_LEFT,
    key_down = GLFW_KEY_DOWN,
    key_up = GLFW_KEY_UP,
    key_page_up = GLFW_KEY_PAGE_UP,
    key_page_down = GLFW_KEY_PAGE_DOWN,
    key_home = GLFW_KEY_HOME,
    key_end = GLFW_KEY_END,
    key_caps_lock = GLFW_KEY_CAPS_LOCK,
    key_scroll_lock = GLFW_KEY_SCROLL_LOCK,
    key_num_lock = GLFW_KEY_NUM_LOCK,
    key_print_screen = GLFW_KEY_PRINT_SCREEN,
    key_break = GLFW_KEY_PAUSE,
    key_f1 = GLFW_KEY_F1,
    key_f2 = GLFW_KEY_F2,
    key_f3 = GLFW_KEY_F3,
    key_f4 = GLFW_KEY_F4,
    key_f5 = GLFW_KEY_F5,
    key_f6 = GLFW_KEY_F6,
    key_f7 = GLFW_KEY_F7,
    key_f8 = GLFW_KEY_F8,
    key_f9 = GLFW_KEY_F9,
    key_f10 = GLFW_KEY_F10,
    key_f11 = GLFW_KEY_F11,
    key_f12 = GLFW_KEY_F12,
    key_f13 = GLFW_KEY_F13,
    key_f14 = GLFW_KEY_F14,
    key_f15 = GLFW_KEY_F15,
    key_f16 = GLFW_KEY_F16,
    key_f17 = GLFW_KEY_F17,
    key_f18 = GLFW_KEY_F18,
    key_f19 = GLFW_KEY_F19,
    key_f20 = GLFW_KEY_F20,
    key_f21 = GLFW_KEY_F21,
    key_f22 = GLFW_KEY_F22,
    key_f23 = GLFW_KEY_F23,
    key_f24 = GLFW_KEY_F24,
    key_numpad_0 = GLFW_KEY_KP_0,
    key_numpad_1 = GLFW_KEY_KP_1,
    key_numpad_2 = GLFW_KEY_KP_2,
    key_numpad_3 = GLFW_KEY_KP_3,
    key_numpad_4 = GLFW_KEY_KP_4,
    key_numpad_5 = GLFW_KEY_KP_5,
    key_numpad_6 = GLFW_KEY_KP_6,
    key_numpad_7 = GLFW_KEY_KP_7,
    key_numpad_8 = GLFW_KEY_KP_8,
    key_numpad_9 = GLFW_KEY_KP_9,
    key_numpad_decimal = GLFW_KEY_KP_DECIMAL,
    key_numpad_divide = GLFW_KEY_KP_DIVIDE,
    key_numpad_multiply = GLFW_KEY_KP_MULTIPLY,
    key_numpad_subtract = GLFW_KEY_KP_SUBTRACT,
    key_numpad_add = GLFW_KEY_KP_ADD,
    key_numpad_enter = GLFW_KEY_KP_ENTER,
    key_numpad_equal = GLFW_KEY_KP_EQUAL,
    key_shift = GLFW_KEY_LEFT_SHIFT,
    key_control = GLFW_KEY_LEFT_CONTROL,
    key_alt = GLFW_KEY_LEFT_ALT,
    key_left_shift = GLFW_KEY_LEFT_SHIFT,
    key_left_control = GLFW_KEY_LEFT_CONTROL,
    key_left_alt = GLFW_KEY_LEFT_ALT,
    key_left_super = GLFW_KEY_LEFT_SUPER,
    key_right_shift = GLFW_KEY_RIGHT_SHIFT,
    key_right_control = GLFW_KEY_RIGHT_CONTROL,
    key_right_alt = GLFW_KEY_RIGHT_ALT,
    key_right_super = GLFW_KEY_RIGHT_SUPER,
    key_menu = GLFW_KEY_MENU,
    key_last = key_menu,
    mouse_first = GLFW_MOUSE_BUTTON_LEFT,
    mouse_left = GLFW_MOUSE_BUTTON_LEFT,
    mouse_right = GLFW_MOUSE_BUTTON_RIGHT,
    mouse_middle = GLFW_MOUSE_BUTTON_MIDDLE,
    mouse_scroll_up = GLFW_MOUSE_BUTTON_LAST + 1,
    mouse_scroll_down = GLFW_MOUSE_BUTTON_LAST + 2,
    mouse_last = mouse_scroll_down,
    gamepad_first = GLFW_KEY_LAST + 1,
    gamepad_a = gamepad_first,
    gamepad_b,
    gamepad_x,
    gamepad_y,
    gamepad_left_bumper,
    gamepad_right_bumper,
    gamepad_l1 = gamepad_left_bumper,
    gamepad_r1 = gamepad_right_bumper,
    gamepad_back,
    gamepad_start,
    gamepad_left_thumb,
    gamepad_right_thumb,
    gamepad_up,
    gamepad_right,
    gamepad_down,
    gamepad_left,
    gamepad_guide,
    gamepad_cross = gamepad_a,
    gamepad_circle = gamepad_b,
    gamepad_square = gamepad_x,
    gamepad_triangle = gamepad_y,
    gamepad_l2 = 364,
    gamepad_r2,
    gamepad_last = gamepad_r2,
    key_invalid = GLFW_KEY_UNKNOWN,
    input_last = gamepad_last
  };
  
  int input_enum_to_array_index(int key);

  int array_index_to_enum_input(int index);

  namespace special_lparam {
    constexpr auto lshift_lparam_down = 0x2a0001;
    constexpr auto rshift_lparam_down = 0x360001;
    constexpr auto lctrl_lparam_down = 0x1d0001;
    constexpr auto rctrl_lparam_down = 0x11d0001;
    
    constexpr auto lshift_lparam_up = 0xC02A0001;
    constexpr auto rshift_lparam_up = 0xC0360001;
    constexpr auto lctrl_lparam_up = 0xC01D0001;
    constexpr auto rctrl_lparam_up = 0xC11D0001;
  }

  const char* get_key_name(int key);

  constexpr const char* get_mouse_name(int button) {
    if (button == -1) return "None";
    
    switch (button) {
      case fan::mouse_left: return "Left Click";
      case fan::mouse_right: return "Right Click";
      case fan::mouse_middle: return "Middle Click";
      default: return "Unknown";
    }
  }

  const char* get_controller_button_name(int button) {
    if (button == -1) return "None";

    thread_local char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "Button %d", button);
    return buffer;
  }

  const char* get_controller_axis_name(int axis) {
    if (axis == -1) return "None";
    
    switch (axis) {
      case 0: return "Left X";
      case 1: return "Left Y";
      case 2: return "Right X";
      case 3: return "Right Y";
      case 4: return "Left Trigger";
      case 5: return "Right Trigger";
      default: {
        static char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "Axis %d", axis);
        return buffer;
      }
    }
  }
  enum device_type_e {
    device_keyboard,
    device_mouse,
    device_gamepad
  };

  constexpr device_type_e get_key_device_type(int key) {
    if (key >= key_first && key <= key_last) return device_keyboard;
    if (key >= mouse_first && key <= mouse_last) return device_mouse;
    if (key >= gamepad_first && key <= gamepad_last) return device_gamepad;
    return device_keyboard; // default
  }
  constexpr bool is_keyboard_key(int key) {
    return key >= key_first && key <= key_last;
  }
  constexpr bool is_mouse_button(int key) {
    return key >= mouse_first && key <= mouse_last;
  }
  constexpr bool is_gamepad_button(int key) {
    return key >= gamepad_first && key <= gamepad_last;
  }
  
  std::string to_lower(const std::string& s) {
    std::string r;
    r.reserve(s.size());
    for (unsigned char c : s) {
      r.push_back((char)std::tolower(c));
    }
    return r;
  }
  std::string trim_ws(const std::string& s) {
    std::size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) ++b;
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
  }
  bool iequals(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
      if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i]))
        return false;
    }
    return true;
  }
  bool is_modifier(int key) {
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
  int key_name_to_code(const std::string& name);

  std::string key_code_to_name(int key);
  std::string join_keys(const std::vector<int>& keys, std::string_view sep) {
    std::string r;
    r.reserve(keys.size() * 6);
    for (std::size_t i = 0; i < keys.size(); i++) {
      r += fan::get_key_name(keys[i]);
      if (i + 1 < keys.size()) r += sep;
    }
    return r;
  }
  constexpr bool is_any_of(char c, std::string_view set) {
    return set.find(c) != std::string_view::npos;
  }
}

#endif