static uint16_t convert_scancode_to_fan(uint16_t key) {
	switch (key) {
		//shifts and ctrls require lparam check for which side the button is pressed this is usually checked before converting
		#ifdef fan_platform_windows
		case 0x01: { return fan::input::key_escape; }
		case 0x02: { return fan::input::key_1; }
		case 0x03: { return fan::input::key_2; }
		case 0x04: { return fan::input::key_3; }
		case 0x05: { return fan::input::key_4; }
		case 0x06: { return fan::input::key_5; }
		case 0x07: { return fan::input::key_6; }
		case 0x08: { return fan::input::key_7; }
		case 0x09: { return fan::input::key_8; }
		case 0x0A: { return fan::input::key_9; }
		case 0x0B: { return fan::input::key_0; }
		case 0x0C: { return fan::input::key_minus; }
		case 0x0D: { return fan::input::key_plus; }
		case 0x0E: { return fan::input::key_backspace; }
		case 0x0F: { return fan::input::key_tab; }
		case 0x10: { return fan::input::key_q; }
		case 0x11: { return fan::input::key_w; }
		case 0x12: { return fan::input::key_e; }
		case 0x13: { return fan::input::key_r; }
		case 0x14: { return fan::input::key_t; }
		case 0x15: { return fan::input::key_y; }
		case 0x16: { return fan::input::key_u; }
		case 0x17: { return fan::input::key_i; }
		case 0x18: { return fan::input::key_o; }
    case 0x19: { return fan::input::key_p; }
    case 0x1A: { return fan::input::key_left_bracket; }
    case 0x1B: { return fan::input::key_right_bracket; }
    case 0x1C: { return fan::input::key_enter; }
    case 0x1D: { return fan::input::key_left_control; }
    case 0x1E: { return fan::input::key_a; }
    case 0x1F: { return fan::input::key_s; }
    case 0x20: { return fan::input::key_d; }
		case 0x21: { return fan::input::key_f; }
		case 0x22: { return fan::input::key_g; }
		case 0x23: { return fan::input::key_h; }
		case 0x24: { return fan::input::key_j; }
		case 0x25: { return fan::input::key_k; }
		case 0x26: { return fan::input::key_l; }
		case 0x27: { return fan::input::key_semicolon; }
		case 0x28: { return fan::input::key_quote; }
		case 0x29: { return fan::input::key_tilde; }
		case 0x2A: { return fan::input::key_left_shift; }
		case 0x2B: { return fan::input::key_backslash; }
		case 0x2C: { return fan::input::key_z; }
		case 0x2D: { return fan::input::key_x; }
		case 0x2E: { return fan::input::key_c; }
		case 0x2F: { return fan::input::key_v; }
		case 0x30: { return fan::input::key_b; }
		case 0x31: { return fan::input::key_n; }
		case 0x32: { return fan::input::key_m; }
		case 0x33: { return fan::input::key_less_than; }
		case 0x34: { return fan::input::key_greater_than; }
		case 0x35: { return fan::input::key_slash; }
		case 0x36: { return fan::input::key_right_shift; }
		case 0x37: { return fan::input::key_numpad_multiply; }
    case 0x38: { return fan::input::key_left_alt; }
    case 0x39: { return fan::input::key_space; }
    case 0x3A: { return fan::input::key_caps_lock; }
    case 0x3B: { return fan::input::key_f1; }
    case 0x3C: { return fan::input::key_f2; }
    case 0x3D: { return fan::input::key_f3; }
    case 0x3E: { return fan::input::key_f4; }
    case 0x3F: { return fan::input::key_f5; }
    case 0x40: { return fan::input::key_f6; }
		case 0x41: { return fan::input::key_f7; }
		case 0x42: { return fan::input::key_f8; }
		case 0x43: { return fan::input::key_f9; }
		case 0x44: { return fan::input::key_f10; }
		case 0x45: { return fan::input::key_num_lock; }
		case 0x46: { return fan::input::key_scroll_lock; }
		case 0x47: { return fan::input::key_numpad_7; }
		case 0x48: { return fan::input::key_numpad_8; }
		case 0x49: { return fan::input::key_numpad_9; }
		case 0x4A: { return fan::input::key_numpad_substract; }
		case 0x4B: { return fan::input::key_numpad_4; }
		case 0x4C: { return fan::input::key_numpad_5; }
		case 0x4D: { return fan::input::key_numpad_6; }
		case 0x4E: { return fan::input::key_numpad_add; }
		case 0x4F: { return fan::input::key_numpad_1; }
		case 0x50: { return fan::input::key_numpad_2; }
		case 0x51: { return fan::input::key_numpad_3; }
		case 0x52: { return fan::input::key_numpad_0; }
		case 0x53: { return fan::input::key_numpad_substract; }
    case 0x56: { return fan::input::key_less_than; }
    case 0x156: { return fan::input::key_greater_than; }

    case 0x11c: { return fan::input::key_numpad_enter; }
    case 0x11d: { return fan::input::key_right_control; }
    case 0x12a: { return fan::input::key_left_shift; }
    case 0x135: { return fan::input::key_numpad_divide; }
    case 0x136: { return fan::input::key_right_shift; }
    case 0x137: { return fan::input::key_control_print_screen; }
    case 0x138: { return fan::input::key_right_alt; }
    case 0x146: { return fan::input::key_control_break; }
    case 0x147: { return fan::input::key_home; }
    case 0x148: { return fan::input::key_up; }
    case 0x149: { return fan::input::key_page_up; }
    case 0x14b: { return fan::input::key_left; }
    case 0x14d: { return fan::input::key_right; }
    case 0x14f: { return fan::input::key_end; }
    case 0x150: { return fan::input::key_down; }
    case 0x151: { return fan::input::key_page_down; }
    case 0x152: { return fan::input::key_insert; }
    case 0x153: { return fan::input::key_delete; }
    case 0x15b: { return fan::input::key_left_super; }
    case 0x15c: { return fan::input::key_right_super; }
    case 0x15d: { return fan::input::key_menu; }


		#elif defined(fan_platform_unix)

		case 0x01: { return fan::input::key_escape; }
		case 0x02: { return fan::input::key_1; }
		case 0x03: { return fan::input::key_2; }
		case 0x04: { return fan::input::key_3; }
		case 0x05: { return fan::input::key_4; }
		case 0x06: { return fan::input::key_5; }
		case 0x07: { return fan::input::key_6; }
		case 0x08: { return fan::input::key_7; }
		case 0x09: { return fan::input::key_8; }
		case 0x0A: { return fan::input::key_9; }
		case 0x0B: { return fan::input::key_0; }
		case 0x0C: { return fan::input::key_minus; }
		case 0x0D: { return fan::input::key_plus; }
		case 0x0E: { return fan::input::key_backspace; }
		case 0x0F: { return fan::input::key_tab; }
		case 0x10: { return fan::input::key_q; }
		case 0x11: { return fan::input::key_w; }
		case 0x12: { return fan::input::key_e; }
		case 0x13: { return fan::input::key_r; }
		case 0x14: { return fan::input::key_t; }
		case 0x15: { return fan::input::key_y; }
		case 0x16: { return fan::input::key_u; }
		case 0x17: { return fan::input::key_i; }
		case 0x18: { return fan::input::key_o; }
    case 0x19: { return fan::input::key_p; }
    case 0x1A: { return fan::input::key_left_bracket; }
    case 0x1B: { return fan::input::key_right_bracket; }
    case 0x1C: { return fan::input::key_enter; }
    case 0x1D: { return fan::input::key_left_control; }
    case 0x1E: { return fan::input::key_a; }
    case 0x1F: { return fan::input::key_s; }
    case 0x20: { return fan::input::key_d; }
		case 0x21: { return fan::input::key_f; }
		case 0x22: { return fan::input::key_g; }
		case 0x23: { return fan::input::key_h; }
		case 0x24: { return fan::input::key_j; }
		case 0x25: { return fan::input::key_k; }
		case 0x26: { return fan::input::key_l; }
		case 0x27: { return fan::input::key_semicolon; }
		case 0x28: { return fan::input::key_quote; }
		case 0x29: { return fan::input::key_tilde; }
		case 0x2A: { return fan::input::key_left_shift; }
		case 0x2B: { return fan::input::key_backslash; }
		case 0x2C: { return fan::input::key_z; }
		case 0x2D: { return fan::input::key_x; }
		case 0x2E: { return fan::input::key_c; }
		case 0x2F: { return fan::input::key_v; }
		case 0x30: { return fan::input::key_b; }
		case 0x31: { return fan::input::key_n; }
		case 0x32: { return fan::input::key_m; }
		case 0x33: { return fan::input::key_less_than; }
		case 0x34: { return fan::input::key_greater_than; }
		case 0x35: { return fan::input::key_slash; }
		case 0x36: { return fan::input::key_right_shift; }
		case 0x37: { return fan::input::key_numpad_multiply; }
    case 0x38: { return fan::input::key_left_alt; }
    case 0x39: { return fan::input::key_space; }
    case 0x3A: { return fan::input::key_caps_lock; }
    case 0x3B: { return fan::input::key_f1; }
    case 0x3C: { return fan::input::key_f2; }
    case 0x3D: { return fan::input::key_f3; }
    case 0x3E: { return fan::input::key_f4; }
    case 0x3F: { return fan::input::key_f5; }
    case 0x40: { return fan::input::key_f6; }
		case 0x41: { return fan::input::key_f7; }
		case 0x42: { return fan::input::key_f8; }
		case 0x43: { return fan::input::key_f9; }
		case 0x44: { return fan::input::key_f10; }
		case 0x45: { return fan::input::key_num_lock; }
		case 0x46: { return fan::input::key_scroll_lock; }
		case 0x47: { return fan::input::key_numpad_7; }
		case 0x48: { return fan::input::key_numpad_8; }
		case 0x49: { return fan::input::key_numpad_9; }
		case 0x4A: { return fan::input::key_numpad_substract; }
		case 0x4B: { return fan::input::key_numpad_4; }
		case 0x4C: { return fan::input::key_numpad_5; }
		case 0x4D: { return fan::input::key_numpad_6; }
		case 0x4E: { return fan::input::key_numpad_add; }
		case 0x4F: { return fan::input::key_numpad_1; }
		case 0x50: { return fan::input::key_numpad_2; }
		case 0x51: { return fan::input::key_numpad_3; }
		case 0x52: { return fan::input::key_numpad_0; }
		case 0x53: { return fan::input::key_numpad_substract; }
    case 0x56: { return fan::input::key_less_than; }
    case 0xe056: { return fan::input::key_greater_than; }

    case 0xe01c: { return fan::input::key_numpad_enter; }
    case 0xe01d: { return fan::input::key_right_control; }
    case 0xe02a: { return fan::input::key_left_shift; }
    case 0xe035: { return fan::input::key_numpad_divide; }
    case 0xe036: { return fan::input::key_right_shift; }
    case 0xe037: { return fan::input::key_control_print_screen; }
    case 0xe038: { return fan::input::key_right_alt; }
    case 0xe046: { return fan::input::key_control_break; }
    case 0xe047: { return fan::input::key_home; }
    case 0xe048: { return fan::input::key_up; }
    case 0xe049: { return fan::input::key_page_up; }
    case 0xe04b: { return fan::input::key_left; }
    case 0xe04d: { return fan::input::key_right; }
    case 0xe04f: { return fan::input::key_end; }
    case 0xe050: { return fan::input::key_down; }
    case 0xe051: { return fan::input::key_page_down; }
    case 0xe052: { return fan::input::key_insert; }
    case 0xe053: { return fan::input::key_delete; }
    case 0xe05b: { return fan::input::key_left_super; }
    case 0xe05c: { return fan::input::key_right_super; }
    case 0xe05d: { return fan::input::key_menu; }

		#else

		static_assert("not implemented os");

		#endif

		default:   { return fan::input::key_invalid; }

	}
}

static uint16_t convert_fan_to_scancode(uint16_t key) {
  // non us less/greater than sign
 /* if (key == fan::input::key_less_than) {
    return 0x56;
  }
  else if (key == fan::input::key_greater_than) {

  }
  case 0x156: { return; }*/

	switch (key) {
		//shifts and ctrls require lparam check for which side the button is pressed this is usually checked before converting
#ifdef fan_platform_windows
	case fan::input::key_escape: { return 0x01; }
	case fan::input::key_1: { return 0x02; }
	case fan::input::key_2: { return 0x03; }
	case fan::input::key_3: { return 0x04; }
	case fan::input::key_4: { return 0x05; }
	case fan::input::key_5: { return 0x06; }
	case fan::input::key_6: { return 0x07; }
	case fan::input::key_7: { return 0x08; }
	case fan::input::key_8: { return 0x09; }
	case fan::input::key_9: { return 0x0A; }
	case fan::input::key_0: { return 0x0B; }
	case fan::input::key_minus: { return 0x0C; }
	case fan::input::key_plus: { return 0x0D; }
	case fan::input::key_backspace: { return 0x0E; }
	case fan::input::key_tab: { return 0x0F; }
	case fan::input::key_q: { return 0x10; }
	case fan::input::key_w: { return 0x11; }
	case fan::input::key_e: { return 0x12; }
	case fan::input::key_r: { return 0x13; }
	case fan::input::key_t: { return 0x14; }
	case fan::input::key_y: { return 0x15; }
	case fan::input::key_u: { return 0x16; }
	case fan::input::key_i: { return 0x17; }
	case fan::input::key_o: { return 0x18; }
  case fan::input::key_p: { return 0x19; }
  case fan::input::key_left_bracket: { return 0x1A; }
  case fan::input::key_right_bracket: { return 0x1B; }
  case fan::input::key_enter: { return 0x1C; }
  case fan::input::key_left_control: { return 0x1D; }
  case fan::input::key_a: { return 0x1E; }
  case fan::input::key_s: { return 0x1F; }
  case fan::input::key_d: { return 0x20; }
	case fan::input::key_f: { return 0x21; }
	case fan::input::key_g: { return 0x22; }
	case fan::input::key_h: { return 0x23; }
	case fan::input::key_j: { return 0x24; }
	case fan::input::key_k: { return 0x25; }
	case fan::input::key_l: { return 0x26; }
	case fan::input::key_semicolon: { return 0x27; }
	case fan::input::key_quote: { return 0x28; }
	case fan::input::key_tilde: { return 0x29; }
	case fan::input::key_left_shift: { return 0x2A; }
	case fan::input::key_backslash: { return 0x2B; }
	case fan::input::key_z: { return 0x2C; }
	case fan::input::key_x: { return 0x2D; }
	case fan::input::key_c: { return 0x2E; }
	case fan::input::key_v: { return 0x2F; }
	case fan::input::key_b: { return 0x30; }
	case fan::input::key_n: { return 0x31; }
	case fan::input::key_m: { return 0x32; }
	case fan::input::key_less_than: { return 0x33; }
	case fan::input::key_greater_than: { return 0x34; }
	case fan::input::key_slash: { return 0x35; }
	case fan::input::key_right_shift: { return 0x36; }
	case fan::input::key_numpad_multiply: { return 0x37; }
  case fan::input::key_left_alt: { return 0x38; }
  case fan::input::key_space: { return 0x39; }
  case fan::input::key_caps_lock: { return 0x3A; }
  case fan::input::key_f1: { return 0x3B; }
  case fan::input::key_f2: { return 0x3C; }
  case fan::input::key_f3: { return 0x3D; }
  case fan::input::key_f4: { return 0x3E; }
  case fan::input::key_f5: { return 0x3F; }
  case fan::input::key_f6: { return 0x40; }
	case fan::input::key_f7: { return 0x41; }
	case fan::input::key_f8: { return 0x42; }
	case fan::input::key_f9: { return 0x43; }
	case fan::input::key_f10: { return 0x44; }
	case fan::input::key_num_lock: { return 0x45; }
	case fan::input::key_scroll_lock: { return 0x46; }
	case fan::input::key_numpad_7: { return 0x47; }
	case fan::input::key_numpad_8: { return 0x48; }
	case fan::input::key_numpad_9: { return 0x49; }
	case fan::input::key_numpad_substract: { return 0x4A; }
	case fan::input::key_numpad_4: { return 0x4B; }
	case fan::input::key_numpad_5: { return 0x4C; }
	case fan::input::key_numpad_6: { return 0x4D; }
	case fan::input::key_numpad_add: { return 0x4E; }
	case fan::input::key_numpad_1: { return 0x4F; }
	case fan::input::key_numpad_2: { return 0x50; }
	case fan::input::key_numpad_3: { return 0x51; }
	case fan::input::key_numpad_0: { return 0x52; }
	//case fan::input::key_numpad_substract: { return 0x53; }

  case fan::input::key_numpad_enter:{ return 0x11c; }
  case fan::input::key_right_control: { return 0x11d; }
  //case fan::input::key_left_shift: { return 0x12a; }
  case fan::input::key_numpad_divide: { return 0x135; }
  //case fan::input::key_right_shift: { return 0x136; }
  case fan::input::key_control_print_screen: { return 0x137; }
  case fan::input::key_right_alt: { return 0x138; }
  case fan::input::key_control_break: { return 0x146; }
  case fan::input::key_home: { return 0x147; }
  case fan::input::key_up: { return 0x148; }
  case fan::input::key_page_up: { return 0x149; }
  case fan::input::key_left: { return 0x14b; }
  case fan::input::key_right: { return 0x14d; }
  case fan::input::key_end: { return 0x14f; }
  case fan::input::key_down: { return 0x150; }
  case fan::input::key_page_down: { return 0x151; }
  case fan::input::key_insert: { return 0x152; }
  case fan::input::key_delete: { return 0x153; }
  case fan::input::key_left_super: { return 0x15b; }
  case fan::input::key_right_super: { return 0x15c; }
  case fan::input::key_menu: { return 0x15d; }

#elif defined(fan_platform_unix)

switch (input) {
  case fan::input::key_escape: { return 0x01; }
  case fan::input::key_1: { return 0x02; }
  case fan::input::key_2: { return 0x03; }
  case fan::input::key_3: { return 0x04; }
  case fan::input::key_4: { return 0x05; }
  case fan::input::key_5: { return 0x06; }
  case fan::input::key_6: { return 0x07; }
  case fan::input::key_7: { return 0x08; }
  case fan::input::key_8: { return 0x09; }
  case fan::input::key_9: { return 0x0A; }
  case fan::input::key_0: { return 0x0B; }
  case fan::input::key_minus: { return 0x0C; }
  case fan::input::key_plus: { return 0x0D; }
  case fan::input::key_backspace: { return 0x0E; }
  case fan::input::key_tab: { return 0x0F; }
  case fan::input::key_q: { return 0x10; }
  case fan::input::key_w: { return 0x11; }
  case fan::input::key_e: { return 0x12; }
  case fan::input::key_r: { return 0x13; }
  case fan::input::key_t: { return 0x14; }
  case fan::input::key_y: { return 0x15; }
  case fan::input::key_u: { return 0x16; }
  case fan::input::key_i: { return 0x17; }
  case fan::input::key_o: { return 0x18; }
  case fan::input::key_p: { return 0x19; }
  case fan::input::key_left_bracket: { return 0x1A; }
  case fan::input::key_right_bracket: { return 0x1B; }
  case fan::input::key_enter: { return 0x1C; }
  case fan::input::key_left_control: { return 0x1D; }
  case fan::input::key_a: { return 0x1E; }
  case fan::input::key_s: { return 0x1F; }
  case fan::input::key_d: { return 0x20; }
  case fan::input::key_f: { return 0x21; }
  case fan::input::key_g: { return 0x22; }
  case fan::input::key_h: { return 0x23; }
  case fan::input::key_j: { return 0x24; }
  case fan::input::key_k: { return 0x25; }
  case fan::input::key_l: { return 0x26; }
  case fan::input::key_semicolon: { return 0x27; }
  case fan::input::key_quote: { return 0x28; }
  case fan::input::key_tilde: { return 0x29; }
  case fan::input::key_left_shift: { return 0x2A; }
  case fan::input::key_backslash: { return 0x2B; }
  case fan::input::key_z: { return 0x2C; }
  case fan::input::key_x: { return 0x2D; }
  case fan::input::key_c: { return 0x2E; }
  case fan::input::key_v: { return 0x2F; }
  case fan::input::key_b: { return 0x30; }
  case fan::input::key_n: { return 0x31; }
  case fan::input::key_m: { return 0x32; }
  case fan::input::key_less_than: { return 0x33; }
  case fan::input::key_greater_than: { return 0x34; }
  case fan::input::key_slash: { return 0x35; }
  case fan::input::key_right_shift: { return 0x36; }
  case fan::input::key_numpad_multiply: { return 0x37; }
  case fan::input::key_left_alt: { return 0x38; }
  case fan::input::key_space: { return 0x39; }
  case fan::input::key_caps_lock: { return 0x3A; }
  case fan::input::key_f1: { return 0x3B; }
  case fan::input::key_f2: { return 0x3C; }
  case fan::input::key_f3: { return 0x3D; }
  case fan::input::key_f4: { return 0x3E; }
  case fan::input::key_f5: { return 0x3F; }
  case fan::input::key_f6: { return 0x40; }
  case fan::input::key_f7: { return 0x41; }
  case fan::input::key_f8: { return 0x42; }
  case fan::input::key_f9: { return 0x43; }
  case fan::input::key_f10: { return 0x44; }
  case fan::input::key_num_lock: { return 0x45; }
  case fan::input::key_scroll_lock: { return 0x46; }
  case fan::input::key_numpad_7: { return 0x47; }
  case fan::input::key_numpad_8: { return 0x48; }
  case fan::input::key_numpad_9: { return 0x49; }
  case fan::input::key_numpad_substract: { return 0x4A; }
  case fan::input::key_numpad_4: { return 0x4B; }
  case fan::input::key_numpad_5: { return 0x4C; }
  case fan::input::key_numpad_6: { return 0x4D; }
  case fan::input::key_numpad_add: { return 0x4E; }
  case fan::input::key_numpad_1: { return 0x4F; }
  case fan::input::key_numpad_2: { return 0x50; }
  case fan::input::key_numpad_3: { return 0x51; }
  case fan::input::key_numpad_0: { return 0x52; }
  case fan::input::key_numpad_substract: { return 0x53; }
  case fan::input::key_less_than: { return 0x56; }
  case fan::input::key_greater_than: { return 0xe056; }
  case fan::input::key_numpad_enter: { return 0xe01c; }
  case fan::input::key_right_control: { return 0xe01d; }
  case fan::input::key_left_shift: { return 0xe02a; }
  case fan::input::key_numpad_divide: { return 0xe035; }
  case fan::input::key_right_shift: { return 0xe036; }
  case fan::input::key_control_print_screen: { return 0xe037; }
  case fan::input::key_right_alt: { return 0xe038; }
  case fan::input::key_control_break: { return 0xe046; }
  case fan::input::key_home: { return 0xe047; }
  case fan::input::key_up: { return 0xe048; }
  case fan::input::key_page_up: { return 0xe049; }
  case fan::input::key_left: { return 0xe04b; }
  case fan::input::key_right: { return 0xe04d; }
  case fan::input::key_end: { return 0xe04f; }
  case fan::input::key_down: { return 0xe050; }
  case fan::input::key_page_down: { return 0xe051; }
  case fan::input::key_insert: { return 0xe052; }
  case fan::input::key_delete: { return 0xe053; }
  case fan::input::key_left_super: { return 0xe05b; }
  case fan::input::key_right_super: { return 0xe05c; }
  case fan::input::key_menu: { return 0xe05d; }

#else

		static_assert("not implemented os");

#endif

	default: { return fan::input::key_invalid; }

	}
}