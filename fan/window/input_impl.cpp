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

module fan.window.input;

#if defined (FAN_WINDOW)

import std;

int fan::input_enum_to_array_index(int key) {
  switch (key) {
  case fan::input::key_space: return 0;
  case fan::input::key_0: return 1;
  case fan::input::key_1: return 2;
  case fan::input::key_2: return 3;
  case fan::input::key_3: return 4;
  case fan::input::key_4: return 5;
  case fan::input::key_5: return 6;
  case fan::input::key_6: return 7;
  case fan::input::key_7: return 8;
  case fan::input::key_8: return 9;
  case fan::input::key_9: return 10;
  case fan::input::key_apostrophe: return 11;
  case fan::input::key_period: return 12;
  case fan::input::key_comma: return 13;
  case fan::input::key_plus: return 14;
  case fan::input::key_minus: return 15;
  case fan::input::key_slash: return 16;
  case fan::input::key_semicolon: return 17;
  case fan::input::key_a: return 18;
  case fan::input::key_b: return 19;
  case fan::input::key_c: return 20;
  case fan::input::key_d: return 21;
  case fan::input::key_e: return 22;
  case fan::input::key_f: return 23;
  case fan::input::key_g: return 24;
  case fan::input::key_h: return 25;
  case fan::input::key_i: return 26;
  case fan::input::key_j: return 27;
  case fan::input::key_k: return 28;
  case fan::input::key_l: return 29;
  case fan::input::key_m: return 30;
  case fan::input::key_n: return 31;
  case fan::input::key_o: return 32;
  case fan::input::key_p: return 33;
  case fan::input::key_q: return 34;
  case fan::input::key_r: return 35;
  case fan::input::key_s: return 36;
  case fan::input::key_t: return 37;
  case fan::input::key_u: return 38;
  case fan::input::key_v: return 39;
  case fan::input::key_w: return 40;
  case fan::input::key_x: return 41;
  case fan::input::key_y: return 42;
  case fan::input::key_z: return 43;
  case fan::input::key_left_bracket: return 44;
  case fan::input::key_backslash: return 45;
  case fan::input::key_right_bracket: return 46;
  case fan::input::key_grave_accent: return 47;
  case fan::input::key_less_than: return 48;
  case fan::input::key_greater_than: return 49;
  case fan::input::key_escape: return 50;
  case fan::input::key_enter: return 51;
  case fan::input::key_tab: return 52;
  case fan::input::key_backspace: return 53;
  case fan::input::key_insert: return 54;
  case fan::input::key_delete: return 55;
  case fan::input::key_right: return 56;
  case fan::input::key_left: return 57;
  case fan::input::key_down: return 58;
  case fan::input::key_up: return 59;
  case fan::input::key_page_up: return 60;
  case fan::input::key_page_down: return 61;
  case fan::input::key_home: return 62;
  case fan::input::key_end: return 63;
  case fan::input::key_caps_lock: return 64;
  case fan::input::key_scroll_lock: return 65;
  case fan::input::key_num_lock: return 66;
  case fan::input::key_print_screen: return 67;
  case fan::input::key_break: return 68;
  case fan::input::key_f1: return 69;
  case fan::input::key_f2: return 70;
  case fan::input::key_f3: return 71;
  case fan::input::key_f4: return 72;
  case fan::input::key_f5: return 73;
  case fan::input::key_f6: return 74;
  case fan::input::key_f7: return 75;
  case fan::input::key_f8: return 76;
  case fan::input::key_f9: return 77;
  case fan::input::key_f10: return 78;
  case fan::input::key_f11: return 79;
  case fan::input::key_f12: return 80;
  case fan::input::key_f13: return 81;
  case fan::input::key_f14: return 82;
  case fan::input::key_f15: return 83;
  case fan::input::key_f16: return 84;
  case fan::input::key_f17: return 85;
  case fan::input::key_f18: return 86;
  case fan::input::key_f19: return 87;
  case fan::input::key_f20: return 88;
  case fan::input::key_f21: return 89;
  case fan::input::key_f22: return 90;
  case fan::input::key_f23: return 91;
  case fan::input::key_f24: return 92;
  case fan::input::key_numpad_0: return 93;
  case fan::input::key_numpad_1: return 94;
  case fan::input::key_numpad_2: return 95;
  case fan::input::key_numpad_3: return 96;
  case fan::input::key_numpad_4: return 97;
  case fan::input::key_numpad_5: return 98;
  case fan::input::key_numpad_6: return 99;
  case fan::input::key_numpad_7: return 100;
  case fan::input::key_numpad_8: return 101;
  case fan::input::key_numpad_9: return 102;
  case fan::input::key_numpad_decimal: return 103;
  case fan::input::key_numpad_divide: return 104;
  case fan::input::key_numpad_multiply: return 105;
  case fan::input::key_numpad_subtract: return 106;
  case fan::input::key_numpad_add: return 107;
  case fan::input::key_numpad_enter: return 108;
  case fan::input::key_numpad_equal: return 109;
  case fan::input::key_shift: return 110;
  case fan::input::key_control: return 111;
  case fan::input::key_alt: return 112;
  case fan::input::key_left_super: return 116;
  case fan::input::key_right_shift: return 117;
  case fan::input::key_right_control: return 118;
  case fan::input::key_right_alt: return 119;
  case fan::input::key_right_super: return 120;
  case fan::input::key_menu: return 121;
  case fan::input::mouse_left: return 122;
  case fan::input::mouse_right: return 123;
  case fan::input::mouse_middle: return 124;
  case fan::input::mouse_scroll_up: return 125;
  case fan::input::mouse_scroll_down: return 126;
  case fan::input::gamepad_a: return 127;
  case fan::input::gamepad_b: return 128;
  case fan::input::gamepad_x: return 129;
  case fan::input::gamepad_y: return 130;
  case fan::input::gamepad_left_bumper: return 131;
  case fan::input::gamepad_right_bumper: return 132;
  case fan::input::gamepad_back: return 133;
  case fan::input::gamepad_start: return 134;
  case fan::input::gamepad_left_thumb: return 135;
  case fan::input::gamepad_right_thumb: return 136;
  case fan::input::gamepad_up: return 137;
  case fan::input::gamepad_right: return 138;
  case fan::input::gamepad_down: return 139;
  case fan::input::gamepad_left: return 140;
  case fan::input::gamepad_guide: return 141;
  case fan::input::key_invalid: return 149;
  default: return -1;
  }
}

int fan::array_index_to_enum_input(int index) {
  switch (index) {
  case 0: return fan::input::key_space;
  case 1: return fan::input::key_0;
  case 2: return fan::input::key_1;
  case 3: return fan::input::key_2;
  case 4: return fan::input::key_3;
  case 5: return fan::input::key_4;
  case 6: return fan::input::key_5;
  case 7: return fan::input::key_6;
  case 8: return fan::input::key_7;
  case 9: return fan::input::key_8;
  case 10: return fan::input::key_9;
  case 11: return fan::input::key_apostrophe;
  case 12: return fan::input::key_period;
  case 13: return fan::input::key_comma;
  case 14: return fan::input::key_plus;
  case 15: return fan::input::key_minus;
  case 16: return fan::input::key_slash;
  case 17: return fan::input::key_semicolon;
  case 18: return fan::input::key_a;
  case 19: return fan::input::key_b;
  case 20: return fan::input::key_c;
  case 21: return fan::input::key_d;
  case 22: return fan::input::key_e;
  case 23: return fan::input::key_f;
  case 24: return fan::input::key_g;
  case 25: return fan::input::key_h;
  case 26: return fan::input::key_i;
  case 27: return fan::input::key_j;
  case 28: return fan::input::key_k;
  case 29: return fan::input::key_l;
  case 30: return fan::input::key_m;
  case 31: return fan::input::key_n;
  case 32: return fan::input::key_o;
  case 33: return fan::input::key_p;
  case 34: return fan::input::key_q;
  case 35: return fan::input::key_r;
  case 36: return fan::input::key_s;
  case 37: return fan::input::key_t;
  case 38: return fan::input::key_u;
  case 39: return fan::input::key_v;
  case 40: return fan::input::key_w;
  case 41: return fan::input::key_x;
  case 42: return fan::input::key_y;
  case 43: return fan::input::key_z;
  case 44: return fan::input::key_left_bracket;
  case 45: return fan::input::key_backslash;
  case 46: return fan::input::key_right_bracket;
  case 47: return fan::input::key_grave_accent;
  case 48: return fan::input::key_less_than;
  case 49: return fan::input::key_greater_than;
  case 50: return fan::input::key_escape;
  case 51: return fan::input::key_enter;
  case 52: return fan::input::key_tab;
  case 53: return fan::input::key_backspace;
  case 54: return fan::input::key_insert;
  case 55: return fan::input::key_delete;
  case 56: return fan::input::key_right;
  case 57: return fan::input::key_left;
  case 58: return fan::input::key_down;
  case 59: return fan::input::key_up;
  case 60: return fan::input::key_page_up;
  case 61: return fan::input::key_page_down;
  case 62: return fan::input::key_home;
  case 63: return fan::input::key_end;
  case 64: return fan::input::key_caps_lock;
  case 65: return fan::input::key_scroll_lock;
  case 66: return fan::input::key_num_lock;
  case 67: return fan::input::key_print_screen;
  case 68: return fan::input::key_break;
  case 69: return fan::input::key_f1;
  case 70: return fan::input::key_f2;
  case 71: return fan::input::key_f3;
  case 72: return fan::input::key_f4;
  case 73: return fan::input::key_f5;
  case 74: return fan::input::key_f6;
  case 75: return fan::input::key_f7;
  case 76: return fan::input::key_f8;
  case 77: return fan::input::key_f9;
  case 78: return fan::input::key_f10;
  case 79: return fan::input::key_f11;
  case 80: return fan::input::key_f12;
  case 81: return fan::input::key_f13;
  case 82: return fan::input::key_f14;
  case 83: return fan::input::key_f15;
  case 84: return fan::input::key_f16;
  case 85: return fan::input::key_f17;
  case 86: return fan::input::key_f18;
  case 87: return fan::input::key_f19;
  case 88: return fan::input::key_f20;
  case 89: return fan::input::key_f21;
  case 90: return fan::input::key_f22;
  case 91: return fan::input::key_f23;
  case 92: return fan::input::key_f24;
  case 93: return fan::input::key_numpad_0;
  case 94: return fan::input::key_numpad_1;
  case 95: return fan::input::key_numpad_2;
  case 96: return fan::input::key_numpad_3;
  case 97: return fan::input::key_numpad_4;
  case 98: return fan::input::key_numpad_5;
  case 99: return fan::input::key_numpad_6;
  case 100: return fan::input::key_numpad_7;
  case 101: return fan::input::key_numpad_8;
  case 102: return fan::input::key_numpad_9;
  case 103: return fan::input::key_numpad_decimal;
  case 104: return fan::input::key_numpad_divide;
  case 105: return fan::input::key_numpad_multiply;
  case 106: return fan::input::key_numpad_subtract;
  case 107: return fan::input::key_numpad_add;
  case 108: return fan::input::key_numpad_enter;
  case 109: return fan::input::key_numpad_equal;
  case 110: return fan::input::key_shift;
  case 111: return fan::input::key_control;
  case 112: return fan::input::key_alt;
  case 113: return fan::input::key_left_shift;
  case 114: return fan::input::key_left_control;
  case 115: return fan::input::key_left_alt;
  case 116: return fan::input::key_left_super;
  case 117: return fan::input::key_right_shift;
  case 118: return fan::input::key_right_control;
  case 119: return fan::input::key_right_alt;
  case 120: return fan::input::key_right_super;
  case 121: return fan::input::key_menu;
  case 122: return fan::input::mouse_left;
  case 123: return fan::input::mouse_right;
  case 124: return fan::input::mouse_middle;
  case 125: return fan::input::mouse_scroll_up;
  case 126: return fan::input::mouse_scroll_down;
  case 127: return fan::input::gamepad_a;
  case 128: return fan::input::gamepad_b;
  case 129: return fan::input::gamepad_x;
  case 130: return fan::input::gamepad_y;
  case 131: return fan::input::gamepad_left_bumper;
  case 132: return fan::input::gamepad_right_bumper;
  case 133: return fan::input::gamepad_back;
  case 134: return fan::input::gamepad_start;
  case 135: return fan::input::gamepad_left_thumb;
  case 136: return fan::input::gamepad_right_thumb;
  case 137: return fan::input::gamepad_up;
  case 138: return fan::input::gamepad_right;
  case 139: return fan::input::gamepad_down;
  case 140: return fan::input::gamepad_left;
  case 141: return fan::input::gamepad_guide;
  case 142: return fan::input::gamepad_last;
  case 143: return fan::input::gamepad_cross;
  case 144: return fan::input::gamepad_circle;
  case 145: return fan::input::gamepad_square;
  case 146: return fan::input::gamepad_triangle;
  case 147: return fan::input::gamepad_l2;
  case 148: return fan::input::gamepad_r2;
  case 149: return fan::input::key_invalid;
  default: return -1;
  }
}

const char* fan::get_key_name(int key) {
  if (key == -1) {
    return "None";
  }

  switch (key) {
  case fan::key_space: return "Space";
  case fan::key_apostrophe: return "'";
  case fan::key_comma: return ",";
  case fan::key_minus: return "-";
  case fan::key_period: return ".";
  case fan::key_slash: return "/";
  case fan::key_0: return "0";
  case fan::key_1: return "1";
  case fan::key_2: return "2";
  case fan::key_3: return "3";
  case fan::key_4: return "4";
  case fan::key_5: return "5";
  case fan::key_6: return "6";
  case fan::key_7: return "7";
  case fan::key_8: return "8";
  case fan::key_9: return "9";
  case fan::key_semicolon: return ";";
  case fan::key_plus: return "=";
  case fan::key_a: return "A";
  case fan::key_b: return "B";
  case fan::key_c: return "C";
  case fan::key_d: return "D";
  case fan::key_e: return "E";
  case fan::key_f: return "F";
  case fan::key_g: return "G";
  case fan::key_h: return "H";
  case fan::key_i: return "I";
  case fan::key_j: return "J";
  case fan::key_k: return "K";
  case fan::key_l: return "L";
  case fan::key_m: return "M";
  case fan::key_n: return "N";
  case fan::key_o: return "O";
  case fan::key_p: return "P";
  case fan::key_q: return "Q";
  case fan::key_r: return "R";
  case fan::key_s: return "S";
  case fan::key_t: return "T";
  case fan::key_u: return "U";
  case fan::key_v: return "V";
  case fan::key_w: return "W";
  case fan::key_x: return "X";
  case fan::key_y: return "Y";
  case fan::key_z: return "Z";
  case fan::key_left_bracket: return "[";
  case fan::key_backslash: return "\\";
  case fan::key_right_bracket: return "]";
  case fan::key_grave_accent: return "`";
  case fan::key_escape: return "Escape";
  case fan::key_enter: return "Enter";
  case fan::key_tab: return "Tab";
  case fan::key_backspace: return "Backspace";
  case fan::key_insert: return "Insert";
  case fan::key_delete: return "Delete";
  case fan::key_right: return "Right";
  case fan::key_left: return "Left";
  case fan::key_down: return "Down";
  case fan::key_up: return "Up";
  case fan::key_page_up: return "Page Up";
  case fan::key_page_down: return "Page Down";
  case fan::key_home: return "Home";
  case fan::key_end: return "End";
  case fan::key_caps_lock: return "Caps Lock";
  case fan::key_scroll_lock: return "Scroll Lock";
  case fan::key_num_lock: return "Num Lock";
  case fan::key_print_screen: return "Print Screen";
  case fan::key_break: return "Pause/Break";
  case fan::key_f1: return "F1";
  case fan::key_f2: return "F2";
  case fan::key_f3: return "F3";
  case fan::key_f4: return "F4";
  case fan::key_f5: return "F5";
  case fan::key_f6: return "F6";
  case fan::key_f7: return "F7";
  case fan::key_f8: return "F8";
  case fan::key_f9: return "F9";
  case fan::key_f10: return "F10";
  case fan::key_f11: return "F11";
  case fan::key_f12: return "F12";
  case fan::key_left_shift: return "Left Shift";
  case fan::key_left_control: return "Left Ctrl";
  case fan::key_left_alt: return "Left Alt";
  case fan::key_left_super: return "Left Super";
  case fan::key_right_shift: return "Right Shift";
  case fan::key_right_control: return "Right Ctrl";
  case fan::key_right_alt: return "Right Alt";
  case fan::key_right_super: return "Right Super";
  default:
    switch (key) {
    case fan::gamepad_a: return "Gamepad A";
    case fan::gamepad_b: return "Gamepad B";
    case fan::gamepad_x: return "Gamepad X";
    case fan::gamepad_y: return "Gamepad Y";
    case fan::gamepad_left_bumper: return "Gamepad LB";
    case fan::gamepad_right_bumper: return "Gamepad RB";
    case fan::gamepad_back: return "Gamepad Back";
    case fan::gamepad_start: return "Gamepad Start";
    case fan::gamepad_left_thumb: return "Gamepad LThumb";
    case fan::gamepad_right_thumb: return "Gamepad RThumb";
    case fan::gamepad_up: return "Gamepad DPad Up";
    case fan::gamepad_right: return "Gamepad DPad Right";
    case fan::gamepad_down: return "Gamepad DPad Down";
    case fan::gamepad_left: return "Gamepad DPad Left";
    case fan::gamepad_guide: return "Gamepad Guide";
    case fan::gamepad_l2: return "Gamepad L2";
    case fan::gamepad_r2: return "Gamepad R2";
    default: return "Unknown";
    }
  }
}

int fan::key_name_to_code(const std::string& name) {
  std::string trimmed = fan::trim_ws(name);
  if (trimmed.empty()) return -1;
  std::string lowered = fan::to_lower(trimmed);

  if (fan::iequals(lowered, "none") || fan::iequals(lowered, "unknown")) {
    return -1;
  }

  for (int key = fan::mouse_first; key <= fan::mouse_last; ++key) {
    const char* mn = fan::get_mouse_name(key);
    if (!mn) continue;
    if (fan::iequals(lowered, fan::to_lower(mn))) {
      return key;
    }
  }

  for (int key = fan::gamepad_a; key <= fan::gamepad_last; ++key) {
    const char* gn = fan::get_key_name(key);
    if (!gn) continue;
    if (fan::iequals(lowered, fan::to_lower(gn))) {
      return key;
    }
  }

  for (int key = fan::key_first; key <= fan::key_last; ++key) {
    const char* kn = fan::get_key_name(key);
    if (!kn) continue;
    if (fan::iequals(lowered, fan::to_lower(kn))) {
      return key;
    }
  }

  return -1;
}

std::string fan::key_code_to_name(int key) {
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

#endif
