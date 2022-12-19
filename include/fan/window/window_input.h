#pragma once

#include _FAN_PATH(types/types.h)

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
		release,
		press,
		repeat
	};
	enum class button_state {
		release,
		press
	};
  
	enum input {
		first = 0x2A,
		key_space,
		key_0,
		key_1,
		key_2,
		key_3,
		key_4,
		key_5,
		key_6,
		key_7,
		key_8,
		key_9,
		key_exclamation_mark,
		key_apostrophe,
		key_period,
		key_comma,
		key_plus,
		key_minus,
		key_colon,
		key_slash,
		key_semicolon,
		key_quote,
		key_angle,
		key_a,
		key_b,
		key_c,
		key_d,
		key_e,
		key_f,
		key_g,
		key_h,
		key_i,
		key_j,
		key_k,
		key_l,
		key_m,
		key_n,
		key_o,
		key_p,
		key_q,
		key_r,
		key_s,
		key_t,
		key_u,
		key_v,
		key_w,
		key_x,
		key_y,
		key_z,
		key_left_bracket,
		key_backslash,
		key_right_bracket,
		key_grave_accent,

		key_escape,
		key_enter,
		key_tab,
		key_backspace,
		key_insert,
		key_delete,
		key_right,
		key_left,
		key_down,
		key_up,
		key_page_up,
		key_page_down,
		key_home,
		key_end,
		key_caps_lock,
		key_scroll_lock,
		key_num_lock,
		key_print_screen,
		key_pause,
		key_f1,
		key_f2,
		key_f3,
		key_f4,
		key_f5,
		key_f6,
		key_f7,
		key_f8,
		key_f9,
		key_f10,
		key_f11,
		key_f12,
		key_f13,
		key_f14,
		key_f15,
		key_f16,
		key_f17,
		key_f18,
		key_f19,
		key_f20,
		key_f21,
		key_f22,
		key_f23,
		key_f24,
		key_tilde,

		key_numpad_0,
		key_numpad_1,
		key_numpad_2,
		key_numpad_3,
		key_numpad_4,
		key_numpad_5,
		key_numpad_6,
		key_numpad_7,
		key_numpad_8,
		key_numpad_9,
		key_numpad_decimal,
		key_numpad_divide,
		key_numpad_multiply,
		key_numpad_substract,
		key_numpad_add,
		key_numpad_enter,
		key_numpad_equal,
		key_numpad_separator,

		key_shift,
		key_control,
		key_alt,
		key_left_shift,
		key_left_control,
		key_left_alt,
		key_left_super,
		key_right_shift,
		key_right_control,
		key_right_alt,
		key_right_super,
		key_menu,

		button_left,
		button_right,
		button_middle,

		mouse_scroll_up,
		mouse_scroll_down,

		key_invalid,

		last

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