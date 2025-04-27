module;

#include <fan/types/types.h>

#include <cstdint>

import fan.window.input;

#if defined(fan_gui)
  #include <fan/imgui/imgui.h>
#endif

export module fan.window.input_common;

export namespace fan {
  namespace window {
    namespace input {
      uint16_t convert_scancode_to_fan(int key) {
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
        case 0x4A: { return fan::input::key_numpad_subtract; }
        case 0x4B: { return fan::input::key_numpad_4; }
        case 0x4C: { return fan::input::key_numpad_5; }
        case 0x4D: { return fan::input::key_numpad_6; }
        case 0x4E: { return fan::input::key_numpad_add; }
        case 0x4F: { return fan::input::key_numpad_1; }
        case 0x50: { return fan::input::key_numpad_2; }
        case 0x51: { return fan::input::key_numpad_3; }
        case 0x52: { return fan::input::key_numpad_0; }
        case 0x53: { return fan::input::key_numpad_subtract; }
        case 0x56: { return fan::input::key_less_than; }
        case 0x57: { return fan::input::key_f11; }
        case 0x58: { return fan::input::key_f12; }
        case 0x156: { return fan::input::key_greater_than; }

        case 0x11c: { return fan::input::key_numpad_enter; }
        case 0x11d: { return fan::input::key_right_control; }
        case 0x12a: { return fan::input::key_left_shift; }
        case 0x135: { return fan::input::key_numpad_divide; }
        case 0x136: { return fan::input::key_right_shift; }
        case 0x138: { return fan::input::key_right_alt; }
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
        case 0x4A: { return fan::input::key_numpad_subtract; }
        case 0x4B: { return fan::input::key_numpad_4; }
        case 0x4C: { return fan::input::key_numpad_5; }
        case 0x4D: { return fan::input::key_numpad_6; }
        case 0x4E: { return fan::input::key_numpad_add; }
        case 0x4F: { return fan::input::key_numpad_1; }
        case 0x50: { return fan::input::key_numpad_2; }
        case 0x51: { return fan::input::key_numpad_3; }
        case 0x52: { return fan::input::key_numpad_0; }
        case 0x53: { return fan::input::key_numpad_subtract; }
        case 0x56: { return fan::input::key_less_than; }
        case 0x57: { return fan::input::key_f11; }
        case 0x58: { return fan::input::key_f12; }
        case 0xe056: { return fan::input::key_greater_than; }

        case 0xe01c: { return fan::input::key_numpad_enter; }
        case 0xe01d: { return fan::input::key_right_control; }
        case 0xe02a: { return fan::input::key_left_shift; }
        case 0xe035: { return fan::input::key_numpad_divide; }
        case 0xe036: { return fan::input::key_right_shift; }
        case 0xe038: { return fan::input::key_right_alt; }
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

        default: { return fan::input::key_invalid; }

        }
      }

      uint16_t convert_fan_to_scancode(int key) {
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
        case fan::input::key_f11: { return 0x57; }
        case fan::input::key_f12: { return 0x58; }
        case fan::input::key_num_lock: { return 0x45; }
        case fan::input::key_scroll_lock: { return 0x46; }
        case fan::input::key_numpad_7: { return 0x47; }
        case fan::input::key_numpad_8: { return 0x48; }
        case fan::input::key_numpad_9: { return 0x49; }
        case fan::input::key_numpad_subtract: { return 0x4A; }
        case fan::input::key_numpad_4: { return 0x4B; }
        case fan::input::key_numpad_5: { return 0x4C; }
        case fan::input::key_numpad_6: { return 0x4D; }
        case fan::input::key_numpad_add: { return 0x4E; }
        case fan::input::key_numpad_1: { return 0x4F; }
        case fan::input::key_numpad_2: { return 0x50; }
        case fan::input::key_numpad_3: { return 0x51; }
        case fan::input::key_numpad_0: { return 0x52; }
                                     //case fan::input::key_numpad_subtract: { return 0x53; }

        case fan::input::key_numpad_enter: { return 0x11c; }
        case fan::input::key_right_control: { return 0x11d; }
                                          //case fan::input::key_left_shift: { return 0x12a; }
        case fan::input::key_numpad_divide: { return 0x135; }
                                          //case fan::input::key_right_shift: { return 0x136; }
        case fan::input::key_right_alt: { return 0x138; }
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
        case fan::input::key_left_shift: { return 0x2A; }
        case fan::input::key_backslash: { return 0x2B; }
        case fan::input::key_z: { return 0x2C; }
        case fan::input::key_x: { return 0x2D; }
        case fan::input::key_c: { return 0x2E; }
        case fan::input::key_v: { return 0x2F; }
        case fan::input::key_b: { return 0x30; }
        case fan::input::key_n: { return 0x31; }
        case fan::input::key_m: { return 0x32; }
                              //case fan::input::key_less_than: { return 0x33; }
                              //case fan::input::key_greater_than: { return 0x34; }
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
        case fan::input::key_f11: { return 0x57; }
        case fan::input::key_f12: { return 0x58; }
        case fan::input::key_num_lock: { return 0x45; }
        case fan::input::key_scroll_lock: { return 0x46; }
        case fan::input::key_numpad_7: { return 0x47; }
        case fan::input::key_numpad_8: { return 0x48; }
        case fan::input::key_numpad_9: { return 0x49; }
                                     //case fan::input::key_numpad_subtract: { return 0x4A; }
        case fan::input::key_numpad_4: { return 0x4B; }
        case fan::input::key_numpad_5: { return 0x4C; }
        case fan::input::key_numpad_6: { return 0x4D; }
        case fan::input::key_numpad_add: { return 0x4E; }
        case fan::input::key_numpad_1: { return 0x4F; }
        case fan::input::key_numpad_2: { return 0x50; }
        case fan::input::key_numpad_3: { return 0x51; }
        case fan::input::key_numpad_0: { return 0x52; }
        case fan::input::key_numpad_subtract: { return 0x53; }
        case fan::input::key_less_than: { return 0x56; }
        case fan::input::key_greater_than: { return 0xe056; }
        case fan::input::key_numpad_enter: { return 0xe01c; }
        case fan::input::key_right_control: { return 0xe01d; }
                                          //case fan::input::key_left_shift: { return 0xe02a; }
        case fan::input::key_numpad_divide: { return 0xe035; }
                                          //case fan::input::key_right_shift: { return 0xe036; }
        case fan::input::key_right_alt: { return 0xe038; }
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

      #if defined(fan_gui)
      int fan_to_imguikey(int key) {
        switch (key)
        {
        case fan::mouse_left: return (ImGuiKey)ImGuiMouseButton_Left;
        case fan::mouse_middle: return (ImGuiKey)ImGuiMouseButton_Middle;
        case fan::mouse_right: return (ImGuiKey)ImGuiMouseButton_Right;
        case fan::key_tab: return ImGuiKey_Tab;
        case fan::key_left: return ImGuiKey_LeftArrow;
        case fan::key_right: return ImGuiKey_RightArrow;
        case fan::key_up: return ImGuiKey_UpArrow;
        case fan::key_down: return ImGuiKey_DownArrow;
        case fan::key_page_up: return ImGuiKey_PageUp;
        case fan::key_page_down: return ImGuiKey_PageDown;
        case fan::key_home: return ImGuiKey_Home;
        case fan::key_end: return ImGuiKey_End;
        case fan::key_insert: return ImGuiKey_Insert;
        case fan::key_delete: return ImGuiKey_Delete;
        case fan::key_backspace: return ImGuiKey_Backspace;
        case fan::key_space: return ImGuiKey_Space;
        case fan::key_enter: return ImGuiKey_Enter;
        case fan::key_escape: return ImGuiKey_Escape;
        case fan::key_apostrophe: return ImGuiKey_Apostrophe;
        case fan::key_comma: return ImGuiKey_Comma;
        case fan::key_minus: return ImGuiKey_Minus;
        case fan::key_period: return ImGuiKey_Period;
        case fan::key_slash: return ImGuiKey_Slash;
        case fan::key_semicolon: return ImGuiKey_Semicolon;
          //case fan::key_equal: return ImGuiKey_Equal;
        case fan::key_left_bracket: return ImGuiKey_LeftBracket;
        case fan::key_backslash: return ImGuiKey_Backslash;
        case fan::key_right_bracket: return ImGuiKey_RightBracket;
        case fan::key_grave_accent: return ImGuiKey_GraveAccent;
        case fan::key_caps_lock: return ImGuiKey_CapsLock;
        case fan::key_scroll_lock: return ImGuiKey_ScrollLock;
        case fan::key_num_lock: return ImGuiKey_NumLock;
        case fan::key_print_screen: return ImGuiKey_PrintScreen;
        case fan::key_break: return ImGuiKey_Pause;
        case fan::key_numpad_0: return ImGuiKey_Keypad0;
        case fan::key_numpad_1: return ImGuiKey_Keypad1;
        case fan::key_numpad_2: return ImGuiKey_Keypad2;
        case fan::key_numpad_3: return ImGuiKey_Keypad3;
        case fan::key_numpad_4: return ImGuiKey_Keypad4;
        case fan::key_numpad_5: return ImGuiKey_Keypad5;
        case fan::key_numpad_6: return ImGuiKey_Keypad6;
        case fan::key_numpad_7: return ImGuiKey_Keypad7;
        case fan::key_numpad_8: return ImGuiKey_Keypad8;
        case fan::key_numpad_9: return ImGuiKey_Keypad9;
        case fan::key_numpad_decimal: return ImGuiKey_KeypadDecimal;
        case fan::key_numpad_divide: return ImGuiKey_KeypadDivide;
        case fan::key_numpad_multiply: return ImGuiKey_KeypadMultiply;
        case fan::key_numpad_subtract: return ImGuiKey_KeypadSubtract;
        case fan::key_numpad_add: return ImGuiKey_KeypadAdd;
        case fan::key_numpad_enter: return ImGuiKey_KeypadEnter;
        case fan::key_numpad_equal: return ImGuiKey_KeypadEqual;
        case fan::key_left_shift: return ImGuiKey_LeftShift;
        case fan::key_left_control: return ImGuiKey_LeftCtrl;
        case fan::key_left_alt: return ImGuiKey_LeftAlt;
        case fan::key_left_super: return ImGuiKey_LeftSuper;
        case fan::key_right_shift: return ImGuiKey_RightShift;
        case fan::key_right_control: return ImGuiKey_RightCtrl;
        case fan::key_right_alt: return ImGuiKey_RightAlt;
        case fan::key_right_super: return ImGuiKey_RightSuper;
        case fan::key_menu: return ImGuiKey_Menu;
        case fan::key_0: return ImGuiKey_0;
        case fan::key_1: return ImGuiKey_1;
        case fan::key_2: return ImGuiKey_2;
        case fan::key_3: return ImGuiKey_3;
        case fan::key_4: return ImGuiKey_4;
        case fan::key_5: return ImGuiKey_5;
        case fan::key_6: return ImGuiKey_6;
        case fan::key_7: return ImGuiKey_7;
        case fan::key_8: return ImGuiKey_8;
        case fan::key_9: return ImGuiKey_9;
        case fan::key_a: return ImGuiKey_A;
        case fan::key_b: return ImGuiKey_B;
        case fan::key_c: return ImGuiKey_C;
        case fan::key_d: return ImGuiKey_D;
        case fan::key_e: return ImGuiKey_E;
        case fan::key_f: return ImGuiKey_F;
        case fan::key_g: return ImGuiKey_G;
        case fan::key_h: return ImGuiKey_H;
        case fan::key_i: return ImGuiKey_I;
        case fan::key_j: return ImGuiKey_J;
        case fan::key_k: return ImGuiKey_K;
        case fan::key_l: return ImGuiKey_L;
        case fan::key_m: return ImGuiKey_M;
        case fan::key_n: return ImGuiKey_N;
        case fan::key_o: return ImGuiKey_O;
        case fan::key_p: return ImGuiKey_P;
        case fan::key_q: return ImGuiKey_Q;
        case fan::key_r: return ImGuiKey_R;
        case fan::key_s: return ImGuiKey_S;
        case fan::key_t: return ImGuiKey_T;
        case fan::key_u: return ImGuiKey_U;
        case fan::key_v: return ImGuiKey_V;
        case fan::key_w: return ImGuiKey_W;
        case fan::key_x: return ImGuiKey_X;
        case fan::key_y: return ImGuiKey_Y;
        case fan::key_z: return ImGuiKey_Z;
        case fan::key_f1: return ImGuiKey_F1;
        case fan::key_f2: return ImGuiKey_F2;
        case fan::key_f3: return ImGuiKey_F3;
        case fan::key_f4: return ImGuiKey_F4;
        case fan::key_f5: return ImGuiKey_F5;
        case fan::key_f6: return ImGuiKey_F6;
        case fan::key_f7: return ImGuiKey_F7;
        case fan::key_f8: return ImGuiKey_F8;
        case fan::key_f9: return ImGuiKey_F9;
        case fan::key_f10: return ImGuiKey_F10;
        case fan::key_f11: return ImGuiKey_F11;
        case fan::key_f12: return ImGuiKey_F12;
        case fan::key_f13: return ImGuiKey_F13;
        case fan::key_f14: return ImGuiKey_F14;
        case fan::key_f15: return ImGuiKey_F15;
        case fan::key_f16: return ImGuiKey_F16;
        case fan::key_f17: return ImGuiKey_F17;
        case fan::key_f18: return ImGuiKey_F18;
        case fan::key_f19: return ImGuiKey_F19;
        case fan::key_f20: return ImGuiKey_F20;
        case fan::key_f21: return ImGuiKey_F21;
        case fan::key_f22: return ImGuiKey_F22;
        case fan::key_f23: return ImGuiKey_F23;
        case fan::key_f24: return ImGuiKey_F24;
        default: return ImGuiKey_None;
        }
      }
      #endif
    }
  }
}