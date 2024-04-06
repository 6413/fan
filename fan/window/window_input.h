#pragma once

#include <unordered_map>

#ifdef fan_platform_windows

	#define WIN32_LEAN_AND_MEAN

	#include <Windows.h>

	#pragma comment(lib, "user32.lib")

	#undef min
	#undef max

#endif

namespace fan {

  enum class keyboard_state {
    release = GLFW_RELEASE,
    press = GLFW_PRESS,
    repeat = GLFW_REPEAT
  };

  enum class mouse_state {
    release = GLFW_RELEASE,
    press = GLFW_PRESS
  };
  
  enum input {
    first = GLFW_KEY_UNKNOWN, // start with unknown

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
    key_exclamation_mark = GLFW_KEY_APOSTROPHE,
    key_apostrophe = GLFW_KEY_APOSTROPHE,
    key_period = GLFW_KEY_PERIOD,
    key_comma = GLFW_KEY_COMMA,
    key_plus = GLFW_KEY_EQUAL,
    key_minus = GLFW_KEY_MINUS,
    key_colon = GLFW_KEY_SEMICOLON,
    key_slash = GLFW_KEY_SLASH,
    key_semicolon = GLFW_KEY_SEMICOLON,
    //key_quote = GLFW_KEY_GRAVE_ACCENT,
    key_angle = GLFW_KEY_WORLD_1,
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
    key_control_print_screen = GLFW_KEY_PRINT_SCREEN,
    key_break = GLFW_KEY_PAUSE,
    key_control_break = GLFW_KEY_PAUSE,
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
    key_tilde = GLFW_KEY_GRAVE_ACCENT,

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
    //key_numpad_separator = GLFW_KEY_KP_SEPARATOR,

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

    mouse_left = GLFW_MOUSE_BUTTON_LEFT,
    mouse_right = GLFW_MOUSE_BUTTON_RIGHT,
    mouse_middle = GLFW_MOUSE_BUTTON_MIDDLE,

    mouse_scroll_up = GLFW_MOUSE_BUTTON_LAST + 1,
    mouse_scroll_down = GLFW_MOUSE_BUTTON_LAST + 2,

    key_invalid = GLFW_KEY_UNKNOWN,

    last = GLFW_KEY_LAST
  };
  
	namespace special_lparam {
		static constexpr auto lshift_lparam_down = 0x2a0001;
		static constexpr auto rshift_lparam_down = 0x360001;
		static constexpr auto lctrl_lparam_down = 0x1d0001;
		static constexpr auto rctrl_lparam_down = 0x11d0001;

		static constexpr auto lshift_lparam_up = 0xC02A0001;
		static constexpr auto rshift_lparam_up = 0xC0360001;
		static constexpr auto lctrl_lparam_up = 0xC01D0001;
		static constexpr auto rctrl_lparam_up = 0xC11D0001;
	}


	namespace window_input {

    #include _FAN_PATH(window/window_input_common.h)

	}
}