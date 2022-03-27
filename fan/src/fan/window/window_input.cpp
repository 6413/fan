#include <fan/window/window_input.h>

uint16_t fan::window_input::convert_keys_to_fan(uint16_t key) {
	switch (key) {
		//shifts and ctrls require lparam check for which side the button is pressed this is usually checked before converting
		#ifdef fan_platform_windows
		case 0x01: { return fan::input::mouse_left; }
		case 0x02: { return fan::input::mouse_right; }
		case 0x04: { return fan::input::mouse_middle; }

		case 0x08: { return fan::input::key_backspace; }
		case 0x09: { return fan::input::key_tab; }
		case 0x0D: { return fan::input::key_enter; }
		case 0x10: { return fan::input::key_shift; }
		case 0x11: { return fan::input::key_control; }
		case 0x12: { return fan::input::key_menu; }
		case 0x13: { return fan::input::key_pause; }
		case 0x14: { return fan::input::key_caps_lock; }
		case 0x1B: { return fan::input::key_escape; }
		case 0x20: { return fan::input::key_space; }
		case 0x21: { return fan::input::key_page_up; }
		case 0x22: { return fan::input::key_page_down; }
		case 0x23: { return fan::input::key_end; }
		case 0x24: { return fan::input::key_home; }
		case 0x25: { return fan::input::key_left; }
		case 0x26: { return fan::input::key_up; }
		case 0x27: { return fan::input::key_right; }
		case 0x28: { return fan::input::key_down; }
		case 0x2C: { return fan::input::key_print_screen; }
		case 0x2D: { return fan::input::key_insert; }
		case 0x2E: { return fan::input::key_delete; }

		case 0x30: { return fan::input::key_0; }
		case 0x31: { return fan::input::key_1; }
		case 0x32: { return fan::input::key_2; }
		case 0x33: { return fan::input::key_3; }
		case 0x34: { return fan::input::key_4; }
		case 0x35: { return fan::input::key_5; }
		case 0x36: { return fan::input::key_6; }
		case 0x37: { return fan::input::key_7; }
		case 0x38: { return fan::input::key_8; }
		case 0x39: { return fan::input::key_9; }

		case 0x41: { return fan::input::key_a; }
		case 0x42: { return fan::input::key_b; }
		case 0x43: { return fan::input::key_c; }
		case 0x44: { return fan::input::key_d; }
		case 0x45: { return fan::input::key_e; }
		case 0x46: { return fan::input::key_f; }
		case 0x47: { return fan::input::key_g; }
		case 0x48: { return fan::input::key_h; }
		case 0x49: { return fan::input::key_i; }
		case 0x4A: { return fan::input::key_j; }
		case 0x4B: { return fan::input::key_k; }
		case 0x4C: { return fan::input::key_l; }
		case 0x4D: { return fan::input::key_m; }
		case 0x4E: { return fan::input::key_n; }
		case 0x4F: { return fan::input::key_o; }
		case 0x50: { return fan::input::key_p; }
		case 0x51: { return fan::input::key_q; }
		case 0x52: { return fan::input::key_r; }
		case 0x53: { return fan::input::key_s; }
		case 0x54: { return fan::input::key_t; }
		case 0x55: { return fan::input::key_u; }
		case 0x56: { return fan::input::key_v; }
		case 0x57: { return fan::input::key_w; }
		case 0x58: { return fan::input::key_x; }
		case 0x59: { return fan::input::key_y; }
		case 0x5A: { return fan::input::key_z; }

		case 0x60: { return fan::input::key_numpad_0; }
		case 0x61: { return fan::input::key_numpad_1; }
		case 0x62: { return fan::input::key_numpad_2; }
		case 0x63: { return fan::input::key_numpad_3; }
		case 0x64: { return fan::input::key_numpad_4; }
		case 0x65: { return fan::input::key_numpad_5; }
		case 0x66: { return fan::input::key_numpad_6; }
		case 0x67: { return fan::input::key_numpad_7; }
		case 0x68: { return fan::input::key_numpad_8; }
		case 0x69: { return fan::input::key_numpad_9; }

		case 0x6A: { return fan::input::key_numpad_multiply; }
		case 0x6B: { return fan::input::key_numpad_add; }
		case 0x6C: { return fan::input::key_numpad_separator; }
		case 0x6D: { return fan::input::key_numpad_substract; }
		case 0x6E: { return fan::input::key_numpad_decimal; }
		case 0x6F: { return fan::input::key_numpad_divide; }

		case 0x70: { return fan::input::key_f1; }
		case 0x71: { return fan::input::key_f2; }
		case 0x72: { return fan::input::key_f3; }
		case 0x73: { return fan::input::key_f4; }
		case 0x74: { return fan::input::key_f5; }
		case 0x75: { return fan::input::key_f6; }
		case 0x76: { return fan::input::key_f7; }
		case 0x77: { return fan::input::key_f8; }
		case 0x78: { return fan::input::key_f9; }
		case 0x79: { return fan::input::key_f10; }
		case 0x7A: { return fan::input::key_f11; }
		case 0x7B: { return fan::input::key_f12; }
		case 0x7C: { return fan::input::key_f13; }
		case 0x7D: { return fan::input::key_f14; }
		case 0x7E: { return fan::input::key_f15; } // possibly tilde
		case 0x7F: { return fan::input::key_f16; }
		case 0x80: { return fan::input::key_f17; }
		case 0x81: { return fan::input::key_f18; }
		case 0x82: { return fan::input::key_f19; }
		case 0x83: { return fan::input::key_f20; }
		case 0x84: { return fan::input::key_f21; }
		case 0x85: { return fan::input::key_f22; }
		case 0x86: { return fan::input::key_f23; }
		case 0x87: { return fan::input::key_f24; }

		case 0x90: { return fan::input::key_num_lock; }
		case 0x91: { return fan::input::key_scroll_lock; }
		case 0xA0: { return fan::input::key_left_shift; }
		case 0xA1: { return fan::input::key_right_shift; }
		case 0xA2: { return fan::input::key_left_control; }
		case 0xA3: { return fan::input::key_right_control; }

		case 0xBB: { return fan::input::key_plus; }
		case 0xBC: { return fan::input::key_comma; }
		case 0xBD: { return fan::input::key_minus; }
		case 0xBE: { return fan::input::key_period; }
		case 0xBF: { return fan::input::key_apostrophe; }

		#elif defined(fan_platform_unix)

		case 0x01: { return fan::input::mouse_left; }
		case 0x02: { return fan::input::mouse_middle; }
		case 0x03: { return fan::input::mouse_right; }
		case 0x04: { return fan::input::mouse_scroll_up; }
		case 0x05: { return fan::input::mouse_scroll_down; }

		case 0x09: { return fan::input::key_escape; }
		case 0x43: { return fan::input::key_f1; }
		case 0x44: { return fan::input::key_f2; }
		case 0x45: { return fan::input::key_f3; }
		case 0x46: { return fan::input::key_f4; }
		case 0x47: { return fan::input::key_f5; }
		case 0x48: { return fan::input::key_f6; }
		case 0x49: { return fan::input::key_f7; }
		case 0x4A: { return fan::input::key_f8; }
		case 0x4B: { return fan::input::key_f9; }
		case 0x4C: { return fan::input::key_f10; }
		case 0x5F: { return fan::input::key_f11; }
		case 0x60: { return fan::input::key_f12; }
				 // more f keys?
				 //case print: ?
		case 0x4E: { return fan::input::key_scroll_lock; }
		case 0x7F: { return fan::input::key_pause; }
		case 0x31: { return fan::input::key_tilde; }
		case 0x13: { return fan::input::key_0; }
		case 0x0A: { return fan::input::key_1; }
		case 0x0B: { return fan::input::key_2; }
		case 0x0C: { return fan::input::key_3; }
		case 0x0D: { return fan::input::key_4; }
		case 0x0E: { return fan::input::key_5; }
		case 0x0F: { return fan::input::key_6; }
		case 0x10: { return fan::input::key_7; }
		case 0x11: { return fan::input::key_8; }
		case 0x12: { return fan::input::key_9; }
		case 0x14: { return fan::input::key_exclamation_mark; }
		case 0x15: { return fan::input::key_apostrophe; }
		case 0x16: { return fan::input::key_backspace; }
		case 0x76: { return fan::input::key_insert; }
		case 0x6E: { return fan::input::key_home; }
		case 0x70: { return fan::input::key_page_up; }
		case 0x17: { return fan::input::key_tab; }
		case 0x18: { return fan::input::key_q; }
		case 0x19: { return fan::input::key_w; }
		case 0x1A: { return fan::input::key_e; }
		case 0x1B: { return fan::input::key_r; }
		case 0x1C: { return fan::input::key_t; }
		case 0x1D: { return fan::input::key_y; }
		case 0x1E: { return fan::input::key_u; }
		case 0x1F: { return fan::input::key_i; }
		case 0x20: { return fan::input::key_o; }
		case 0x21: { return fan::input::key_p; }
				 //case 0x22: { return fan::input::key_å; } ?
				 //case 0x23: { return fan::input::key_^; }
		case 0x24: { return fan::input::key_enter; }
		case 0x77: { return fan::input::key_delete; }
		case 0x73: { return fan::input::key_end; }
		case 0x75: { return fan::input::key_page_down; }

		case 0x42: { return fan::input::key_caps_lock; }
		case 0x26: { return fan::input::key_a; }
		case 0x27: { return fan::input::key_s; }
		case 0x28: { return fan::input::key_d; }
		case 0x29: { return fan::input::key_f; }
		case 0x2A: { return fan::input::key_g; }
		case 0x2B: { return fan::input::key_h; }
		case 0x2C: { return fan::input::key_j; }
		case 0x2D: { return fan::input::key_k; }
		case 0x2E: { return fan::input::key_l; }
		case 0x2F: { return fan::input::key_semicolon; }
		case 0x33: { return fan::input::key_quote; }

		case 0x32: { return fan::input::key_left_shift; }
		case 0x5E: { return fan::input::key_angle; }
		case 0x34: { return fan::input::key_z; }
		case 0x35: { return fan::input::key_x; }
		case 0x36: { return fan::input::key_c; }
		case 0x37: { return fan::input::key_v; }
		case 0x38: { return fan::input::key_b; }
		case 0x39: { return fan::input::key_n; }
		case 0x3A: { return fan::input::key_m; }
		case 0x3B: { return fan::input::key_comma; }
		case 0x3C: { return fan::input::key_colon; }
		case 0x3D: { return fan::input::key_slash; }
		case 0x3E: { return fan::input::key_right_shift; }

		case 0x25: { return fan::input::key_left_control; }
				 //case key_left_super
		case 0x40: { return fan::input::key_left_alt; }
		case 0x41: { return fan::input::key_space; }
		case 0x6C: { return fan::input::key_right_alt; }
		case 0x86: { return fan::input::key_right_super; }
		case 0x69: { return fan::input::key_right_control; }

		case 0x71: { return fan::input::key_left; }
		case 0x6F: { return fan::input::key_up; }
		case 0x72: { return fan::input::key_right; }
		case 0x74: { return fan::input::key_down; }

		case 0x5A: { return fan::input::key_numpad_0; }
		case 0x57: { return fan::input::key_numpad_1; }
		case 0x58: { return fan::input::key_numpad_2; }
		case 0x59: { return fan::input::key_numpad_3; }
		case 0x53: { return fan::input::key_numpad_4; }
		case 0x54: { return fan::input::key_numpad_5; }
		case 0x55: { return fan::input::key_numpad_6; }
		case 0x4F: { return fan::input::key_numpad_7; }
		case 0x50: { return fan::input::key_numpad_8; }
		case 0x51: { return fan::input::key_numpad_9; }

		case 0x3F: { return fan::input::key_numpad_multiply; }
		case 0x56: { return fan::input::key_numpad_add; }
		case 0x52: { return fan::input::key_numpad_substract; }
		case 0x5B: { return fan::input::key_numpad_decimal; }
		case 0x6A: { return fan::input::key_numpad_divide; }

		#else

		static_assert("not implemented os");

		#endif

		default:   { return fan::input::key_invalid; }

	}
}

uint16_t fan::window_input::convert_fan_to_keys(uint16_t key) {
	switch (key) {
		//shifts and ctrls require lparam check for which side the button is pressed this is usually checked before converting
#ifdef fan_platform_windows
	case fan::input::mouse_left: { return 0x01; }
	case fan::input::mouse_right: { return 0x02; }
	case fan::input::mouse_middle: { return 0x04; }

	case fan::input::key_backspace: { return 0x08; }
	case fan::input::key_tab: { return 0x09; }
	case fan::input::key_enter: { return 0x0D; }
	case fan::input::key_shift: { return 0x10; }
	case fan::input::key_control: { return 0x11; }
	case fan::input::key_menu: { return 0x12; }
	case fan::input::key_pause: { return 0x13; }
	case fan::input::key_caps_lock: { return 0x14; }
	case fan::input::key_escape: { return 0x1B; }
	case fan::input::key_space: { return 0x20; }
	case fan::input::key_page_up: { return 0x21; }
	case fan::input::key_page_down: { return 0x22; }
	case fan::input::key_end: { return 0x23; }
	case fan::input::key_home: { return 0x24; }
	case fan::input::key_left: { return 0x25; }
	case fan::input::key_up: { return 0x26; }
	case fan::input::key_right: { return 0x27; }
	case fan::input::key_down: { return 0x28; }
	case fan::input::key_print_screen: { return 0x2C; }
	case fan::input::key_insert: { return 0x2D; }
	case fan::input::key_delete: { return 0x2E; }

	case fan::input::key_0: { return 0x30; }
	case fan::input::key_1: { return 0x31; }
	case fan::input::key_2: { return 0x32; }
	case fan::input::key_3: { return 0x33; }
	case fan::input::key_4: { return 0x34; }
	case fan::input::key_5: { return 0x35; }
	case fan::input::key_6: { return 0x36; }
	case fan::input::key_7: { return 0x37; }
	case fan::input::key_8: { return 0x38; }
	case fan::input::key_9: { return 0x39; }

	case fan::input::key_a: { return 0x41; }
	case fan::input::key_b: { return 0x42; }
	case fan::input::key_c: { return 0x43; }
	case fan::input::key_d: { return 0x44; }
	case fan::input::key_e: { return 0x45; }
	case fan::input::key_f: { return 0x46; }
	case fan::input::key_g: { return 0x47; }
	case fan::input::key_h: { return 0x48; }
	case fan::input::key_i: { return 0x49; }
	case fan::input::key_j: { return 0x4A; }
	case fan::input::key_k: { return 0x4B; }
	case fan::input::key_l: { return 0x4C; }
	case fan::input::key_m: { return 0x4D; }
	case fan::input::key_n: { return 0x4E; }
	case fan::input::key_o: { return 0x4F; }
	case fan::input::key_p: { return 0x50; }
	case fan::input::key_q: { return 0x51; }
	case fan::input::key_r: { return 0x52; }
	case fan::input::key_s: { return 0x53; }
	case fan::input::key_t: { return 0x54; }
	case fan::input::key_u: { return 0x55; }
	case fan::input::key_v: { return 0x56; }
	case fan::input::key_w: { return 0x57; }
	case fan::input::key_x: { return 0x58; }
	case fan::input::key_y: { return 0x59; }
	case fan::input::key_z: { return 0x5A; }

	case fan::input::key_numpad_0: { return 0x60; }
	case fan::input::key_numpad_1: { return 0x61; }
	case fan::input::key_numpad_2: { return 0x62; }
	case fan::input::key_numpad_3: { return 0x63; }
	case fan::input::key_numpad_4: { return 0x64; }
	case fan::input::key_numpad_5: { return 0x65; }
	case fan::input::key_numpad_6: { return 0x66; }
	case fan::input::key_numpad_7: { return 0x67; }
	case fan::input::key_numpad_8: { return 0x68; }
	case fan::input::key_numpad_9: { return 0x69; }

	case fan::input::key_numpad_multiply: { return 0x6A; }
	case fan::input::key_numpad_add: { return 0x6B; }
	case fan::input::key_numpad_separator: { return 0x6C; }
	case fan::input::key_numpad_substract: { return 0x6D; }
	case fan::input::key_numpad_decimal: { return 0x6E; }
	case fan::input::key_numpad_divide: { return 0x6F; }

	case fan::input::key_f1: { return 0x70; }
	case fan::input::key_f2: { return 0x71; }
	case fan::input::key_f3: { return 0x72; }
	case fan::input::key_f4: { return 0x73; }
	case fan::input::key_f5: { return 0x74; }
	case fan::input::key_f6: { return 0x75; }
	case fan::input::key_f7: { return 0x76; }
	case fan::input::key_f8: { return 0x77; }
	case fan::input::key_f9: { return 0x78; }
	case fan::input::key_f10: { return 0x79; }
	case fan::input::key_f11: { return 0x7A; }
	case fan::input::key_f12: { return 0x7B; }
	case fan::input::key_f13: { return 0x7C; }
	case fan::input::key_f14: { return 0x7D; }
	case fan::input::key_f15: { return 0x7E; } // possibly tilde
	case fan::input::key_f16: { return 0x7F; }
	case fan::input::key_f17: { return 0x80; }
	case fan::input::key_f18: { return 0x81; }
	case fan::input::key_f19: { return 0x82; }
	case fan::input::key_f20: { return 0x83; }
	case fan::input::key_f21: { return 0x84; }
	case fan::input::key_f22: { return 0x85; }
	case fan::input::key_f23: { return 0x86; }
	case fan::input::key_f24: { return 0x87; }

	case fan::input::key_num_lock: { return 0x90; }
	case fan::input::key_scroll_lock: { return 0x91; }
	case fan::input::key_left_shift: { return 0xA0; }
	case fan::input::key_right_shift: { return 0xA1; }
	case fan::input::key_left_control: { return 0xA2; }
	case fan::input::key_right_control: { return 0xA3; }

	case fan::input::key_plus: { return 0xBB; }
	case fan::input::key_comma: { return 0xBC; }
	case fan::input::key_minus: { return 0xBD; }
	case fan::input::key_period: { return 0xBE; }
	case fan::input::key_apostrophe: { return 0xBF; }

#elif defined(fan_platform_unix) // todo

#else

		static_assert("not implemented os");

#endif

	default: { return fan::input::key_invalid; }

	}
}

void fan::window_input::get_keys(std::unordered_map<uint16_t, bool>& keys, uint16_t key, bool state) {
	switch (key) {
		case fan::input::key_left_shift:
		case fan::input::key_right_shift:
		{
			keys[fan::input::key_shift] = state;
			break;
		}
		case fan::input::key_left_control:
		case fan::input::key_right_control:
		{
			keys[fan::input::key_control] = state;
			break;
		}
		case fan::input::key_left_alt:
		case fan::input::key_right_alt:
		{
			keys[fan::input::key_alt] = state;
			break;
		}
	}
	keys[key] = state;
}

// with utf using winapi at least delete and '.' is colliding
uint16_t fan::window_input::convert_utfkeys_to_fan(uint16_t key) {
	switch (key) {
		//shifts and ctrls require lparam check for which side the button is pressed this is usually checked before converting
#ifdef fan_platform_windows
		case 0x01: { return fan::input::mouse_left; }
		case 0x02: { return fan::input::mouse_right; }
		case 0x04: { return fan::input::mouse_middle; }

		case 0x08: { return fan::input::key_backspace; }
		case 0x09: { return fan::input::key_tab; }
		case 0x0D: { return fan::input::key_enter; }
		case 0x10: { return fan::input::key_shift; }
		case 0x11: { return fan::input::key_control; }
		case 0x12: { return fan::input::key_menu; }
		case 0x13: { return fan::input::key_pause; }
		case 0x14: { return fan::input::key_caps_lock; }
		case 0x1B: { return fan::input::key_escape; }
		case 0x20: { return fan::input::key_space; }
		case 0x21: { return fan::input::key_page_up; }
		case 0x22: { return fan::input::key_page_down; }
		case 0x23: { return fan::input::key_end; }
		case 0x24: { return fan::input::key_home; }
		case 0x25: { return fan::input::key_left; }
		case 0x26: { return fan::input::key_up; }
		case 0x27: { return fan::input::key_right; }
		case 0x28: { return fan::input::key_down; }
		case 0x2C: { return fan::input::key_print_screen; }
		case 0x2D: { return fan::input::key_insert; }
		case 0x2E: { return fan::input::key_period; }

		case 0x30: { return fan::input::key_0; }
		case 0x31: { return fan::input::key_1; }
		case 0x32: { return fan::input::key_2; }
		case 0x33: { return fan::input::key_3; }
		case 0x34: { return fan::input::key_4; }
		case 0x35: { return fan::input::key_5; }
		case 0x36: { return fan::input::key_6; }
		case 0x37: { return fan::input::key_7; }
		case 0x38: { return fan::input::key_8; }
		case 0x39: { return fan::input::key_9; }

		case 0x41: { return fan::input::key_a; }
		case 0x42: { return fan::input::key_b; }
		case 0x43: { return fan::input::key_c; }
		case 0x44: { return fan::input::key_d; }
		case 0x45: { return fan::input::key_e; }
		case 0x46: { return fan::input::key_f; }
		case 0x47: { return fan::input::key_g; }
		case 0x48: { return fan::input::key_h; }
		case 0x49: { return fan::input::key_i; }
		case 0x4A: { return fan::input::key_j; }
		case 0x4B: { return fan::input::key_k; }
		case 0x4C: { return fan::input::key_l; }
		case 0x4D: { return fan::input::key_m; }
		case 0x4E: { return fan::input::key_n; }
		case 0x4F: { return fan::input::key_o; }
		case 0x50: { return fan::input::key_p; }
		case 0x51: { return fan::input::key_q; }
		case 0x52: { return fan::input::key_r; }
		case 0x53: { return fan::input::key_s; }
		case 0x54: { return fan::input::key_t; }
		case 0x55: { return fan::input::key_u; }
		case 0x56: { return fan::input::key_v; }
		case 0x57: { return fan::input::key_w; }
		case 0x58: { return fan::input::key_x; }
		case 0x59: { return fan::input::key_y; }
		case 0x5A: { return fan::input::key_z; }

		case 0x60: { return fan::input::key_numpad_0; }
		case 0x61: { return fan::input::key_numpad_1; }
		case 0x62: { return fan::input::key_numpad_2; }
		case 0x63: { return fan::input::key_numpad_3; }
		case 0x64: { return fan::input::key_numpad_4; }
		case 0x65: { return fan::input::key_numpad_5; }
		case 0x66: { return fan::input::key_numpad_6; }
		case 0x67: { return fan::input::key_numpad_7; }
		case 0x68: { return fan::input::key_numpad_8; }
		case 0x69: { return fan::input::key_numpad_9; }

		case 0x6A: { return fan::input::key_numpad_multiply; }
		case 0x6B: { return fan::input::key_numpad_add; }
		case 0x6C: { return fan::input::key_numpad_separator; }
		case 0x6D: { return fan::input::key_numpad_substract; }
		case 0x6E: { return fan::input::key_numpad_decimal; }
		case 0x6F: { return fan::input::key_numpad_divide; }

		case 0x70: { return fan::input::key_f1; }
		case 0x71: { return fan::input::key_f2; }
		case 0x72: { return fan::input::key_f3; }
		case 0x73: { return fan::input::key_f4; }
		case 0x74: { return fan::input::key_f5; }
		case 0x75: { return fan::input::key_f6; }
		case 0x76: { return fan::input::key_f7; }
		case 0x77: { return fan::input::key_f8; }
		case 0x78: { return fan::input::key_f9; }
		case 0x79: { return fan::input::key_f10; }
		case 0x7A: { return fan::input::key_f11; }
		case 0x7B: { return fan::input::key_f12; }
		case 0x7C: { return fan::input::key_f13; }
		case 0x7D: { return fan::input::key_f14; }
		case 0x7E: { return fan::input::key_f15; } // possibly tilde
		case 0x7F: { return fan::input::key_f16; }
		case 0x80: { return fan::input::key_f17; }
		case 0x81: { return fan::input::key_f18; }
		case 0x82: { return fan::input::key_f19; }
		case 0x83: { return fan::input::key_f20; }
		case 0x84: { return fan::input::key_f21; }
		case 0x85: { return fan::input::key_f22; }
		case 0x86: { return fan::input::key_f23; }
		case 0x87: { return fan::input::key_f24; }

		case 0x90: { return fan::input::key_num_lock; }
		case 0x91: { return fan::input::key_scroll_lock; }
		case 0xA0: { return fan::input::key_left_shift; }
		case 0xA1: { return fan::input::key_right_shift; }
		case 0xA2: { return fan::input::key_left_control; }
		case 0xA3: { return fan::input::key_right_control; }

		case 0xBB: { return fan::input::key_plus; }
		case 0xBC: { return fan::input::key_comma; }

#elif defined(fan_platform_unix)

		case 0x01: { return fan::input::mouse_left; }
		case 0x02: { return fan::input::mouse_middle; }
		case 0x03: { return fan::input::mouse_right; }
		case 0x04: { return fan::input::mouse_scroll_up; }
		case 0x05: { return fan::input::mouse_scroll_down; }

		case 0x09: { return fan::input::key_escape; }
		case 0x43: { return fan::input::key_f1; }
		case 0x44: { return fan::input::key_f2; }
		case 0x45: { return fan::input::key_f3; }
		case 0x46: { return fan::input::key_f4; }
		case 0x47: { return fan::input::key_f5; }
		case 0x48: { return fan::input::key_f6; }
		case 0x49: { return fan::input::key_f7; }
		case 0x4A: { return fan::input::key_f8; }
		case 0x4B: { return fan::input::key_f9; }
		case 0x4C: { return fan::input::key_f10; }
		case 0x5F: { return fan::input::key_f11; }
		case 0x60: { return fan::input::key_f12; }
				 // more f keys?
				 //case print: ?
		case 0x4E: { return fan::input::key_scroll_lock; }
		case 0x7F: { return fan::input::key_pause; }
		case 0x31: { return fan::input::key_tilde; }
		case 0x13: { return fan::input::key_0; }
		case 0x0A: { return fan::input::key_1; }
		case 0x0B: { return fan::input::key_2; }
		case 0x0C: { return fan::input::key_3; }
		case 0x0D: { return fan::input::key_4; }
		case 0x0E: { return fan::input::key_5; }
		case 0x0F: { return fan::input::key_6; }
		case 0x10: { return fan::input::key_7; }
		case 0x11: { return fan::input::key_8; }
		case 0x12: { return fan::input::key_9; }
		case 0x14: { return fan::input::key_exclamation_mark; }
		case 0x15: { return fan::input::key_apostrophe; }
		case 0x16: { return fan::input::key_backspace; }
		case 0x76: { return fan::input::key_insert; }
		case 0x6E: { return fan::input::key_home; }
		case 0x70: { return fan::input::key_page_up; }
		case 0x17: { return fan::input::key_tab; }
		case 0x18: { return fan::input::key_q; }
		case 0x19: { return fan::input::key_w; }
		case 0x1A: { return fan::input::key_e; }
		case 0x1B: { return fan::input::key_r; }
		case 0x1C: { return fan::input::key_t; }
		case 0x1D: { return fan::input::key_y; }
		case 0x1E: { return fan::input::key_u; }
		case 0x1F: { return fan::input::key_i; }
		case 0x20: { return fan::input::key_o; }
		case 0x21: { return fan::input::key_p; }
				 //case 0x22: { return fan::input::key_å; } ?
				 //case 0x23: { return fan::input::key_^; }
		case 0x24: { return fan::input::key_enter; }
		case 0x77: { return fan::input::key_delete; }
		case 0x73: { return fan::input::key_end; }
		case 0x75: { return fan::input::key_page_down; }

		case 0x42: { return fan::input::key_caps_lock; }
		case 0x26: { return fan::input::key_a; }
		case 0x27: { return fan::input::key_s; }
		case 0x28: { return fan::input::key_d; }
		case 0x29: { return fan::input::key_f; }
		case 0x2A: { return fan::input::key_g; }
		case 0x2B: { return fan::input::key_h; }
		case 0x2C: { return fan::input::key_j; }
		case 0x2D: { return fan::input::key_k; }
		case 0x2E: { return fan::input::key_l; }
		case 0x2F: { return fan::input::key_semicolon; }
		case 0x33: { return fan::input::key_quote; }

		case 0x32: { return fan::input::key_left_shift; }
		case 0x5E: { return fan::input::key_angle; }
		case 0x34: { return fan::input::key_z; }
		case 0x35: { return fan::input::key_x; }
		case 0x36: { return fan::input::key_c; }
		case 0x37: { return fan::input::key_v; }
		case 0x38: { return fan::input::key_b; }
		case 0x39: { return fan::input::key_n; }
		case 0x3A: { return fan::input::key_m; }
		case 0x3B: { return fan::input::key_comma; }
		case 0x3C: { return fan::input::key_colon; }
		case 0x3D: { return fan::input::key_slash; }
		case 0x3E: { return fan::input::key_right_shift; }

		case 0x25: { return fan::input::key_left_control; }
				 //case key_left_super
		case 0x40: { return fan::input::key_left_alt; }
		case 0x41: { return fan::input::key_space; }
		case 0x6C: { return fan::input::key_right_alt; }
		case 0x86: { return fan::input::key_right_super; }
		case 0x69: { return fan::input::key_right_control; }

		case 0x71: { return fan::input::key_left; }
		case 0x6F: { return fan::input::key_up; }
		case 0x72: { return fan::input::key_right; }
		case 0x74: { return fan::input::key_down; }

		case 0x5A: { return fan::input::key_numpad_0; }
		case 0x57: { return fan::input::key_numpad_1; }
		case 0x58: { return fan::input::key_numpad_2; }
		case 0x59: { return fan::input::key_numpad_3; }
		case 0x53: { return fan::input::key_numpad_4; }
		case 0x54: { return fan::input::key_numpad_5; }
		case 0x55: { return fan::input::key_numpad_6; }
		case 0x4F: { return fan::input::key_numpad_7; }
		case 0x50: { return fan::input::key_numpad_8; }
		case 0x51: { return fan::input::key_numpad_9; }

		case 0x3F: { return fan::input::key_numpad_multiply; }
		case 0x56: { return fan::input::key_numpad_add; }
		case 0x52: { return fan::input::key_numpad_substract; }
		case 0x5B: { return fan::input::key_numpad_decimal; }
		case 0x6A: { return fan::input::key_numpad_divide; }

#else

		static_assert("not implemented os");

#endif

		default:   { return fan::input::key_invalid; }

	}
}