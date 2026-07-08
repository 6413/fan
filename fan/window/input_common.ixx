module;

#if defined (FAN_WINDOW)

#include <fan/utility.h>

#if defined(fan_compiler_gcc)
	#ifndef _GCC_MAX_ALIGN_T
		#define _GCC_MAX_ALIGN_T
	#endif
#endif

#if defined(FAN_GUI)
  #include <fan/imgui/imgui.h>
#endif

#endif

export module fan.window.input_common;

#if defined (FAN_WINDOW)

import std;

import fan.window.input;

export namespace fan {
  namespace window {
    namespace input {
      std::uint16_t convert_scancode_to_fan(int key);

      std::uint16_t convert_fan_to_scancode(int key);

      #if defined(FAN_GUI)
      ImGuiKey to_imgui_key(int key);
      #endif
    }
  }
}

#endif