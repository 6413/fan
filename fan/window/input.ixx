module;

#include <fan/utility.h>

#ifdef fan_platform_windows

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#pragma comment(lib, "user32.lib")

#endif

#if defined(FAN_VULKAN)
#include <vulkan/vulkan.h>
#endif
#if defined(fan_platform_windows)
  #define GLFW_EXPOSE_NATIVE_WIN32
  #define GLFW_EXPOSE_NATIVE_WGL
  #define GLFW_NATIVE_INCLUDE_NONE
#endif
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <string>
#include <vector>
#include <string_view>
#include <cstring>

export module fan.window.input;

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
  
  constexpr int input_enum_to_array_index(int key) {
    switch (key) {
    case input::key_space: return 0;
    case input::key_0: return 1;
    case input::key_1: return 2;
    case input::key_2: return 3;
    case input::key_3: return 4;
    case input::key_4: return 5;
    case input::key_5: return 6;
    case input::key_6: return 7;
    case input::key_7: return 8;
    case input::key_8: return 9;
    case input::key_9: return 10;
    case input::key_apostrophe: return 11;
    case input::key_period: return 12;
    case input::key_comma: return 13;
    case input::key_plus: return 14;
    case input::key_minus: return 15;
    case input::key_slash: return 16;
    case input::key_semicolon: return 17;
    case input::key_a: return 18;
    case input::key_b: return 19;
    case input::key_c: return 20;
    case input::key_d: return 21;
    case input::key_e: return 22;
    case input::key_f: return 23;
    case input::key_g: return 24;
    case input::key_h: return 25;
    case input::key_i: return 26;
    case input::key_j: return 27;
    case input::key_k: return 28;
    case input::key_l: return 29;
    case input::key_m: return 30;
    case input::key_n: return 31;
    case input::key_o: return 32;
    case input::key_p: return 33;
    case input::key_q: return 34;
    case input::key_r: return 35;
    case input::key_s: return 36;
    case input::key_t: return 37;
    case input::key_u: return 38;
    case input::key_v: return 39;
    case input::key_w: return 40;
    case input::key_x: return 41;
    case input::key_y: return 42;
    case input::key_z: return 43;
    case input::key_left_bracket: return 44;
    case input::key_backslash: return 45;
    case input::key_right_bracket: return 46;
    case input::key_grave_accent: return 47;
    case input::key_less_than: return 48;
    case input::key_greater_than: return 49;
    case input::key_escape: return 50;
    case input::key_enter: return 51;
    case input::key_tab: return 52;
    case input::key_backspace: return 53;
    case input::key_insert: return 54;
    case input::key_delete: return 55;
    case input::key_right: return 56;
    case input::key_left: return 57;
    case input::key_down: return 58;
    case input::key_up: return 59;
    case input::key_page_up: return 60;
    case input::key_page_down: return 61;
    case input::key_home: return 62;
    case input::key_end: return 63;
    case input::key_caps_lock: return 64;
    case input::key_scroll_lock: return 65;
    case input::key_num_lock: return 66;
    case input::key_print_screen: return 67;
    case input::key_break: return 68;
    case input::key_f1: return 69;
    case input::key_f2: return 70;
    case input::key_f3: return 71;
    case input::key_f4: return 72;
    case input::key_f5: return 73;
    case input::key_f6: return 74;
    case input::key_f7: return 75;
    case input::key_f8: return 76;
    case input::key_f9: return 77;
    case input::key_f10: return 78;
    case input::key_f11: return 79;
    case input::key_f12: return 80;
    case input::key_f13: return 81;
    case input::key_f14: return 82;
    case input::key_f15: return 83;
    case input::key_f16: return 84;
    case input::key_f17: return 85;
    case input::key_f18: return 86;
    case input::key_f19: return 87;
    case input::key_f20: return 88;
    case input::key_f21: return 89;
    case input::key_f22: return 90;
    case input::key_f23: return 91;
    case input::key_f24: return 92;
    case input::key_numpad_0: return 93;
    case input::key_numpad_1: return 94;
    case input::key_numpad_2: return 95;
    case input::key_numpad_3: return 96;
    case input::key_numpad_4: return 97;
    case input::key_numpad_5: return 98;
    case input::key_numpad_6: return 99;
    case input::key_numpad_7: return 100;
    case input::key_numpad_8: return 101;
    case input::key_numpad_9: return 102;
    case input::key_numpad_decimal: return 103;
    case input::key_numpad_divide: return 104;
    case input::key_numpad_multiply: return 105;
    case input::key_numpad_subtract: return 106;
    case input::key_numpad_add: return 107;
    case input::key_numpad_enter: return 108;
    case input::key_numpad_equal: return 109;
    case input::key_shift: return 110;
    case input::key_control: return 111;
    case input::key_alt: return 112;
    case input::key_left_super: return 116;
    case input::key_right_shift: return 117;
    case input::key_right_control: return 118;
    case input::key_right_alt: return 119;
    case input::key_right_super: return 120;
    case input::key_menu: return 121;
    case input::mouse_left: return 122;
    case input::mouse_right: return 123;
    case input::mouse_middle: return 124;
    case input::mouse_scroll_up: return 125;
    case input::mouse_scroll_down: return 126;
    case input::gamepad_a: return 127;
    case input::gamepad_b: return 128;
    case input::gamepad_x: return 129;
    case input::gamepad_y: return 130;
    case input::gamepad_left_bumper: return 131;
    case input::gamepad_right_bumper: return 132;
    case input::gamepad_back: return 133;
    case input::gamepad_start: return 134;
    case input::gamepad_left_thumb: return 135;
    case input::gamepad_right_thumb: return 136;
    case input::gamepad_up: return 137;
    case input::gamepad_right: return 138;
    case input::gamepad_down: return 139;
    case input::gamepad_left: return 140;
    case input::gamepad_guide: return 141;
    case input::key_invalid: return 149;
    default: return -1; // or some other error code
    }
  }

  constexpr int array_index_to_enum_input(int index) {
    switch (index) {
        case 0: return input::key_space;
        case 1: return input::key_0;
        case 2: return input::key_1;
        case 3: return input::key_2;
        case 4: return input::key_3;
        case 5: return input::key_4;
        case 6: return input::key_5;
        case 7: return input::key_6;
        case 8: return input::key_7;
        case 9: return input::key_8;
        case 10: return input::key_9;
        case 11: return input::key_apostrophe;
        case 12: return input::key_period;
        case 13: return input::key_comma;
        case 14: return input::key_plus;
        case 15: return input::key_minus;
        case 16: return input::key_slash;
        case 17: return input::key_semicolon;
        case 18: return input::key_a;
        case 19: return input::key_b;
        case 20: return input::key_c;
        case 21: return input::key_d;
        case 22: return input::key_e;
        case 23: return input::key_f;
        case 24: return input::key_g;
        case 25: return input::key_h;
        case 26: return input::key_i;
        case 27: return input::key_j;
        case 28: return input::key_k;
        case 29: return input::key_l;
        case 30: return input::key_m;
        case 31: return input::key_n;
        case 32: return input::key_o;
        case 33: return input::key_p;
        case 34: return input::key_q;
        case 35: return input::key_r;
        case 36: return input::key_s;
        case 37: return input::key_t;
        case 38: return input::key_u;
        case 39: return input::key_v;
        case 40: return input::key_w;
        case 41: return input::key_x;
        case 42: return input::key_y;
        case 43: return input::key_z;
        case 44: return input::key_left_bracket;
        case 45: return input::key_backslash;
        case 46: return input::key_right_bracket;
        case 47: return input::key_grave_accent;
        case 48: return input::key_less_than;
        case 49: return input::key_greater_than;
        case 50: return input::key_escape;
        case 51: return input::key_enter;
        case 52: return input::key_tab;
        case 53: return input::key_backspace;
        case 54: return input::key_insert;
        case 55: return input::key_delete;
        case 56: return input::key_right;
        case 57: return input::key_left;
        case 58: return input::key_down;
        case 59: return input::key_up;
        case 60: return input::key_page_up;
        case 61: return input::key_page_down;
        case 62: return input::key_home;
        case 63: return input::key_end;
        case 64: return input::key_caps_lock;
        case 65: return input::key_scroll_lock;
        case 66: return input::key_num_lock;
        case 67: return input::key_print_screen;
        case 68: return input::key_break;
        case 69: return input::key_f1;
        case 70: return input::key_f2;
        case 71: return input::key_f3;
        case 72: return input::key_f4;
        case 73: return input::key_f5;
        case 74: return input::key_f6;
        case 75: return input::key_f7;
        case 76: return input::key_f8;
        case 77: return input::key_f9;
        case 78: return input::key_f10;
        case 79: return input::key_f11;
        case 80: return input::key_f12;
        case 81: return input::key_f13;
        case 82: return input::key_f14;
        case 83: return input::key_f15;
        case 84: return input::key_f16;
        case 85: return input::key_f17;
        case 86: return input::key_f18;
        case 87: return input::key_f19;
        case 88: return input::key_f20;
        case 89: return input::key_f21;
        case 90: return input::key_f22;
        case 91: return input::key_f23;
        case 92: return input::key_f24;
        case 93: return input::key_numpad_0;
        case 94: return input::key_numpad_1;
        case 95: return input::key_numpad_2;
        case 96: return input::key_numpad_3;
        case 97: return input::key_numpad_4;
        case 98: return input::key_numpad_5;
        case 99: return input::key_numpad_6;
        case 100: return input::key_numpad_7;
        case 101: return input::key_numpad_8;
        case 102: return input::key_numpad_9;
        case 103: return input::key_numpad_decimal;
        case 104: return input::key_numpad_divide;
        case 105: return input::key_numpad_multiply;
        case 106: return input::key_numpad_subtract;
        case 107: return input::key_numpad_add;
        case 108: return input::key_numpad_enter;
        case 109: return input::key_numpad_equal;
        case 110: return input::key_shift;
        case 111: return input::key_control;
        case 112: return input::key_alt;
        case 113: return input::key_left_shift;
        case 114: return input::key_left_control;
        case 115: return input::key_left_alt;
        case 116: return input::key_left_super;
        case 117: return input::key_right_shift;
        case 118: return input::key_right_control;
        case 119: return input::key_right_alt;
        case 120: return input::key_right_super;
        case 121: return input::key_menu;
        case 122: return input::mouse_left;
        case 123: return input::mouse_right;
        case 124: return input::mouse_middle;
        case 125: return input::mouse_scroll_up;
        case 126: return input::mouse_scroll_down;
        case 127: return input::gamepad_a;
        case 128: return input::gamepad_b;
        case 129: return input::gamepad_x;
        case 130: return input::gamepad_y;
        case 131: return input::gamepad_left_bumper;
        case 132: return input::gamepad_right_bumper;
        case 133: return input::gamepad_back;
        case 134: return input::gamepad_start;
        case 135: return input::gamepad_left_thumb;
        case 136: return input::gamepad_right_thumb;
        case 137: return input::gamepad_up;
        case 138: return input::gamepad_right;
        case 139: return input::gamepad_down;
        case 140: return input::gamepad_left;
        case 141: return input::gamepad_guide;
        case 142: return input::gamepad_last;
        case 143: return input::gamepad_cross;
        case 144: return input::gamepad_circle;
        case 145: return input::gamepad_square;
        case 146: return input::gamepad_triangle;
        case 147: return input::gamepad_l2;
        case 148: return input::gamepad_r2;
        case 149: return input::key_invalid;
        default: return -1; // or some other error code
    }
  }

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

  constexpr const char* get_key_name(int key) {
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
    snprintf(buffer, sizeof(buffer), "Button %d", button);
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
        snprintf(buffer, sizeof(buffer), "Axis %d", axis);
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
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) ++b;
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
  }
  bool iequals(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
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
  int key_name_to_code(const std::string& name) {
    std::string trimmed = trim_ws(name);
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

  std::string key_code_to_name(int key) {
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
  std::string join_keys(const std::vector<int>& keys, std::string_view sep) {
    std::string r;
    r.reserve(keys.size() * 6);
    for (std::size_t i = 0; i < keys.size(); i++) {
      r += fan::get_key_name(keys[i]);
      if (i + 1 < keys.size()) r += sep;
    }
    return r;
  }
}