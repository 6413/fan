#pragma once

#ifndef fan_platform_android

#include _FAN_PATH(system.h)

#ifdef fan_compiler_msvc
	#define _CRT_SECURE_NO_WARNINGS
#endif

#include _FAN_PATH(graphics/opengl/gl_defines.h)

#include _FAN_PATH(math/random.h)

#include _FAN_PATH(types/vector.h)
#include _FAN_PATH(types/matrix.h)

#include _FAN_PATH(types/color.h)
#include _FAN_PATH(time/time.h)
#include _FAN_PATH(window/window_input.h)

#include _FAN_PATH(bll.h)
#include _FAN_PATH(types/memory.h)

#ifdef fan_platform_windows

#include <dwmapi.h>

#pragma comment(lib, "Dwmapi.lib")

#include <windowsx.h>

#include <mbctype.h>

#include <hidusage.h>

#undef min
#undef max

#elif defined(fan_platform_unix)

#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS

#include <locale>
#include <codecvt>

#include <X11/extensions/Xrandr.h>

#include <errno.h>

#endif

#include <deque>
#include <codecvt>
#include <locale>
#include <climits>
#include <type_traits>
#include <optional>

#ifdef fan_platform_windows
	#pragma comment(lib, "opengl32.lib")

	#include <Windows.h>
	#include <shellapi.h>

	#undef min
	#undef max

#elif defined(fan_platform_unix)

	#include <iostream>
	#include <cstring>

	#include <X11/Xlib.h>
	#include <X11/Xutil.h>
	#include <X11/Xos.h>
	#include <X11/Xatom.h>
	#include <X11/keysym.h>
	#include <X11/XKBlib.h>
  #include <X11/Xcursor/Xcursor.h>
  #include <X11/extensions/Xfixes.h>
  #include <X11/cursorfont.h>
  #include <X11/Xutil.h>

	#undef index // xos.h

	#include <sys/time.h>
	#include <unistd.h>
	#include <dlfcn.h>

	#define GLX_CONTEXT_MAJOR_VERSION_ARB       0x2091
	#define GLX_CONTEXT_MINOR_VERSION_ARB       0x2092

#undef index

static int cleanupHandler(Display* display) {
  XUngrabPointer(display, CurrentTime);
  return 0;
}
#endif

#if defined(fan_platform_windows)
static constexpr const char* shared_library = "opengl32.dll";
#elif defined(fan_platform_unix) || defined(fan_platform_android)
static constexpr const char* shared_library = "libGL.so.1";
#endif

namespace fan {

	static void set_console_visibility(bool visible) {
		#ifdef fan_platform_windows
			ShowWindow(GetConsoleWindow(), visible ? SW_SHOW : SW_HIDE);
		#endif
	}

	#ifdef fan_platform_windows


	using window_handle_t = HWND;

	#define FAN_API static


	#elif defined(fan_platform_unix)

	#define FAN_API

	using window_handle_t = Window;

	#endif

	template <typename T>
	constexpr auto get_flag_value(T value) {
		return (1 << value);
	}

	struct pair_hash
	{
		template <class T1, class T2>
		constexpr uint32_t operator() (const std::pair<T1, T2> &pair) const
		{
			return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
		}
	};

	struct window_t;

	void* get_proc_address(const char* name);

	template <typename T>
	constexpr auto initialized(T value) {
		return value != (T)uninitialized;
	}
	struct window_id_storage_t {
		fan::window_handle_t window_handle;
		fan::window_t* window_ptr;
	};

	inline bll_t<window_id_storage_t> window_id_storage;

  static fan::window_t* get_window_by_id(fan::window_handle_t wid) {

    uint32_t it = window_id_storage.begin();
    while (it != window_id_storage.end()) {
      if (window_id_storage[it].window_handle == wid) {
        return window_id_storage[it].window_ptr;
      }
      it = window_id_storage.next(it);
    }

    return 0;
  }
	static void set_window_by_id(fan::window_handle_t wid, fan::window_t* window) {
    window_id_storage.push_back({wid, window});
  }
  static void erase_window_id(fan::window_handle_t wid) {
    uint32_t it = window_id_storage.begin();
    while (it != window_id_storage.end()) {
      if (window_id_storage[it].window_handle == wid) {
        window_id_storage.erase(it);
        return;
      }
      it = window_id_storage.next(it);
    }
  }

	struct window_t {

    // should be X11
    #if defined(fan_platform_unix)

    // https://www.win.tue.nl/~aeb/linux/kbd/scancodes-1.html

    static inline fan::string xcb_get_scancode_name(XkbDescPtr KbDesc, uint16_t keycode) {
      fan::string str(KbDesc->names->keys[keycode].name, KbDesc->names->keys[keycode].name + XkbKeyNameLength);
      str.erase(std::remove(str.begin(), str.end(), '\0'), str.end());
      return str;
    }

    static inline fan::string xcb_get_scancode_name(uint16_t keycode) {
      XkbDescPtr KbDesc = XkbGetMap(fan::sys::m_display, 0, XkbUseCoreKbd);
      XkbGetNames(fan::sys::m_display, XkbKeyNamesMask, KbDesc);
      return xcb_get_scancode_name(KbDesc, keycode);
    }

    void generate_keycode_to_scancode_table() {
      XkbDescPtr KbDesc = XkbGetMap(fan::sys::m_display, 0, XkbUseCoreKbd);
      XkbGetNames(fan::sys::m_display, XkbKeyNamesMask, KbDesc);
      for (uint16_t i = KbDesc->min_key_code; i < fan::min(KbDesc->max_key_code, max_keycode); ++i) {

        static constexpr std::pair<const char*, uint16_t> table[] = {
          {"TLDE", 0x29}, {"AE01", 0x02}, {"AE02", 0x03}, {"AE03", 0x04}, {"AE04", 0x05}, {"AE05", 0x06}, {"AE06", 0x07},
          {"AE07", 0x08}, {"AE08", 0x09}, {"AE09", 0x0a}, {"AE10", 0x0b}, {"AE11", 0x0c}, {"AE12", 0x0d}, {"AD01", 0x10},
          {"AD02", 0x11}, {"AD03", 0x12}, {"AD04", 0x13}, {"AD05", 0x14}, {"AD06", 0x15}, {"AD07", 0x16}, {"AD08", 0x17},
          {"AD09", 0x18}, {"AD10", 0x19}, {"AD11", 0x1a}, {"AD12", 0x1b}, {"AC01", 0x1e}, {"AC02", 0x1f}, {"AC03", 0x20},
          {"AC04", 0x21}, {"AC05", 0x22}, {"AC06", 0x23}, {"AC07", 0x24}, {"AC08", 0x25}, {"AC09", 0x26}, {"AC10", 0x27},
          {"AC11", 0x28}, {"AB01", 0x2c}, {"AB02", 0x2d}, {"AB03", 0x2e}, {"AB04", 0x2f}, {"AB05", 0x30}, {"AB06", 0x31},
          {"AB07", 0x32}, {"AB08", 0x33}, {"AB09", 0x34}, {"AB10", 0x35}, {"BKSL", 0x2b}, {"ESC", 0x01}, {"TAB", 0x0f},
          {"CAPS", 0x3a}, {"LFSH", 0x2a}, {"RTSH", 0x36}, {"LCTL", 0x1d}, {"LALT", 0x38}, {"RALT", 0xe038}, {"LWIN", 0xe05b},
          {"RWIN", 0xe05c}, {"LSGT", 0x33}, {"RSGT", 0x34}, {"COMP", 0xe05d}, {"SPCE", 0x39}, {"RTRN", 0x1c}, {"FK01", 0x3b},
          {"FK02", 0x3c}, {"FK03", 0x3d}, {"FK04", 0x3e}, {"FK05", 0x3f}, {"FK06", 0x40}, {"FK07", 0x41}, {"FK08", 0x42},
          {"FK09", 0x43}, {"FK10", 0x44}, {"LEFT", 0xe04b}, {"RGHT", 0xe04d}, {"UP", 0xe048}, {"DOWN", 0xe050}, {"PGUP", 0xe049},
          {"PGDN", 0xe051}, {"HOME", 0xe047}, {"END", 0xe04f}, {"INS", 0x52}, {"DELE", 0xe053}, {"SCLK", 0x46}, {"KP0", 0x52}, {"KP1", 0x4f},
          {"KP2", 0x50}, {"KP3", 0x51}, {"KP4", 0x4b}, {"KP5", 0x4c}, {"KP6", 0x4d}, {"KP7", 0x47}, {"KP8", 0x48}, {"KP9", 0x49},
          {"KPEN", 0xe01c}, {"RCTL", 0xe01d}, {"BKSP", 0x0e}, {"NMLK", 0x45}, {"KPDL", 0x53}, {"KPAD", 0x4e},
          {"KPDV", 0xe035}, {"KPMU", 0x37}, {"KPSU", 0x4a}, {"KPAD", 0x4e}
        };

        bool found = false;
        for (const auto instance : table) {
          auto str = xcb_get_scancode_name(KbDesc, i);
          //fan::print(string_to_hex(str.c_str()), string_to_hex(instance.first), str.size(), strlen(instance.first), str);
          if (str == instance.first) {
            keycode_to_scancode_table[i] = instance.second;
            found = true;
            break;
          }
        }
        if (!found) {
          //fan::print_warning((std::string("scancode not found for (dec) keycode:") + std::to_string(i)).c_str());
        }
      }
    }

      static bool isExtensionSupported(const char* extList, const char* extension) {
      const char* start;
      const char* where, * terminator;

      where = strchr(extension, ' ');
      if (where || *extension == '\0') {
        return false;
      }

      for (start = extList;;) {
        where = strstr(start, extension);

        if (!where) {
          break;
        }

        terminator = where + strlen(extension);

        if (where == start || *(where - 1) == ' ') {
          if (*terminator == ' ' || *terminator == '\0') {
            return true;
          }
        }

        start = terminator;
      }

      return false;
    }
    #endif

		enum class mode {
			not_set,
			windowed,
			borderless,
			full_screen
		};

		struct events {
			static constexpr uint32_t none = 0;
			static constexpr uint32_t close = 1 << 0;
		};

		struct resolutions {
			constexpr static fan::vec2i r_800x600 = fan::vec2(800, 600);
			constexpr static fan::vec2i r_1024x768 = fan::vec2i(1024, 768);
			constexpr static fan::vec2i r_1280x720 = fan::vec2i(1280, 720);
			constexpr static fan::vec2i r_1280x800 = fan::vec2i(1280, 800);
			constexpr static fan::vec2i r_1280x900 = fan::vec2i(1280, 900);
			constexpr static fan::vec2i r_1280x1024 = fan::vec2i(1280, 1024);
			constexpr static fan::vec2i r_1360x768 = fan::vec2(1360, 768);
			constexpr static fan::vec2i r_1440x900 = fan::vec2i(1440, 900);
			constexpr static fan::vec2i r_1600x900 = fan::vec2i(1600, 900);
			constexpr static fan::vec2i r_1600x1024 = fan::vec2i(1600, 1024);
			constexpr static fan::vec2i r_1680x1050 = fan::vec2i(1680, 1050);
			constexpr static fan::vec2i r_1920x1080 = fan::vec2i(1920, 1080);
		};

		struct mouse_buttons_cb_data_t {
			fan::window_t* window;
			uint16_t button;
			fan::mouse_state state;
		};
		using mouse_buttons_cb_t = fan::function_t<void(const mouse_buttons_cb_data_t&)>;

		struct keyboard_keys_cb_data_t {
			fan::window_t* window;
			uint16_t key;
			fan::keyboard_state state;
      uint16_t scancode;
		};
		using keyboard_keys_cb_t = fan::function_t<void(const keyboard_keys_cb_data_t&)>;

		struct keyboard_key_cb_data_t {
			fan::window_t* window;
			uint16_t key;
		};
		using keyboard_key_cb_t = fan::function_t<void(const keyboard_key_cb_data_t&)>;

		struct text_cb_data_t {
			fan::window_t* window;
			uint32_t character;
      fan::keyboard_state state;
		};
		using text_cb_t = fan::function_t<void(const text_cb_data_t&)>;

		struct mouse_move_cb_data_t {
			fan::window_t* window;
			fan::vec2i position;
		};
		using mouse_move_cb_t = fan::function_t<void(const mouse_move_cb_data_t&)>;

    struct mouse_motion_cb_data_t {
			fan::window_t* window;
      fan::vec2i motion;
		};
		using mouse_motion_cb_t = fan::function_t<void(const mouse_motion_cb_data_t&)>;

		struct close_cb_data_t {
			fan::window_t* window;
		};
		using close_cb_t = fan::function_t<void(const close_cb_data_t&)>;

		struct resize_cb_data_t {
			fan::window_t* window;
			fan::vec2i size;
		};
		using resize_cb_t = fan::function_t<void(const resize_cb_data_t&)>;

		struct move_cb_data_t {
			fan::window_t* window;
		};
		using move_cb_t = fan::function_t<void(const move_cb_data_t&)>;

		struct keyboard_cb_store_t {
			uint16_t key;
			keyboard_state state;

			keyboard_key_cb_t function;
		};

		struct flags {
			static constexpr int no_mouse = get_flag_value(0);
			static constexpr int no_resize = get_flag_value(1);
			static constexpr int mode = get_flag_value(3);
			static constexpr int borderless = get_flag_value(4);
			static constexpr int full_screen = get_flag_value(5);
		};

		static constexpr const char* default_window_name = "window";
		static constexpr vec2i default_window_size = fan::vec2i(800, 600);
		static constexpr mode default_size_mode = mode::windowed;

		// for static value storing
		static constexpr int reserved_storage = -1;

		window_t(const fan::vec2i& window_size = fan::window_t::default_window_size, const fan::string& name = default_window_name, uint64_t flags = 0) {
      m_size = window_size;
      m_mouse_position = 0;
      m_max_fps = 0;
      m_fps_counter = 0;
      m_last_frame = fan::time::clock::now();
      m_current_frame = fan::time::clock::now();
      m_delta_time = 0;
      m_name = name;
      m_flags = flags;
      m_current_key = 0;
      m_reserved_flags = 0;
      m_focused = true;
      m_event_flags = 0;
      m_mouse_motion = 0;

      if (flag_values.m_size_mode == fan::window_t::mode::not_set) {
        flag_values.m_size_mode = fan::window_t::default_size_mode;
      }
      if (static_cast<bool>(flags & fan::window_t::flags::no_mouse)) {
        fan::window_t::flag_values.m_no_mouse = true;
      }
      if (static_cast<bool>(flags & fan::window_t::flags::no_resize)) {
        fan::window_t::flag_values.m_no_resize = true;
      }
      if (static_cast<bool>(flags & fan::window_t::flags::borderless)) {
        fan::window_t::flag_values.m_size_mode = fan::window_t::mode::borderless;
      }
      if (static_cast<bool>(flags & fan::window_t::flags::full_screen)) {
        fan::window_t::flag_values.m_size_mode = fan::window_t::mode::full_screen;
      }

      window_id_storage.open();

      initialize_window(name, window_size, flags);

      this->calculate_delta_time();

      m_may_center = fan::sys::get_screen_resolution() / 2;
    }
		~window_t() {
      this->destroy_window();
    }

		window_t(window_t&&) = delete;
		window_t(const window_t&) = delete;
		window_t& operator=(const window_t&) = delete;
		window_t& operator=(window_t&&) = delete;

		void destroy() {
			#ifdef fan_platform_windows
		#if fan_renderer == fan_renderer_opengl
			wglDeleteContext(m_context);
		#endif

			#elif defined(fan_platform_unix)

		#if fan_renderer == fan_renderer_opengl
			//glXDestroyContext(fan::sys::m_display, m_context);
		#endif
			XCloseDisplay(fan::sys::m_display);
			fan::sys::m_display = 0;

			#endif

		#if fan_renderer == fan_renderer_opengl
			m_context = 0;
		#endif

		}

		fan::string get_name() const {
      return m_name;
    }
		void set_name(const fan::string& name) {

      m_name = name;

      #ifdef fan_platform_windows

      SetWindowTextA(m_window_handle, m_name.c_str());

      #elif defined(fan_platform_unix)

      XStoreName(fan::sys::m_display, m_window_handle, m_name.c_str());
      XSetIconName(fan::sys::m_display, m_window_handle, m_name.c_str());

      #endif

    }

		void calculate_delta_time() {
      m_delta_time = (f64_t)fan::time::clock::elapsed(m_current_frame) / 1000000000;
      m_current_frame = fan::time::clock::now();
    }

		f64_t get_delta_time() const {
      return m_delta_time;
    }

		fan::vec2i get_mouse_position() const {
      return m_mouse_position;
    }
		fan::vec2i get_previous_mouse_position() const {
      return m_previous_mouse_position;
    }

		fan::vec2i get_size() const {
      return m_size;
    }
		fan::vec2i get_previous_size() const {
      return m_previous_size;
    }
		void set_size(const fan::vec2i& size) {
      #ifdef fan_platform_windows

      RECT rect = {0, 0, size.x, size.y};

      AdjustWindowRectEx(&rect, GetWindowStyle(m_window_handle), FALSE, GetWindowExStyle(m_window_handle));

      if (!SetWindowPos(m_window_handle, 0, 0, 0, rect.right - rect.left, rect.bottom - rect.top, SWP_NOZORDER | SWP_SHOWWINDOW | SWP_NOMOVE)) {
        fan::print("fan window error: failed to set window position", GetLastError());
        exit(1);
      }

      #elif defined(fan_platform_unix)

      int result = XResizeWindow(fan::sys::m_display, m_window_handle, size.x, size.y);

      if (result == BadValue || result == BadWindow) {
        fan::print("fan window error: failed to set window position");
        exit(1);
      }

      const fan::vec2i move_offset = (size - get_previous_size()) / 2;

      this->set_position(this->get_position() - move_offset);

      #endif
      m_previous_size = m_size;
      m_size = size;
    }

		fan::vec2i get_position() const {
      return m_position;
    }

		void set_position(const fan::vec2i& position) {
      #ifdef fan_platform_windows

      if (!SetWindowPos(m_window_handle, 0, position.x, position.y, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_SHOWWINDOW)) {
        fan::print("fan window error: failed to set window position", GetLastError());
        exit(1);
      }

      #elif defined(fan_platform_unix)

      int result = XMoveWindow(fan::sys::m_display, m_window_handle, position.x, position.y);

      if (result == BadValue || result == BadWindow) {
        fan::print("fan window error: failed to set window position");
        exit(1);
      }

      #endif
    }


		uintptr_t get_max_fps() const {
      return m_max_fps;
    }

		void set_max_fps(uintptr_t fps) {
      m_max_fps = fps;
      m_fps_next_tick = fan::time::clock::now();
    }

    void lock_cursor_and_set_invisible(bool flag) {

        if (flag == 0) {
       #if defined(fan_platform_windows)
          SetCursor(LoadCursor(NULL, IDC_ARROW));
          ReleaseCapture();
          SetCapture(NULL);

          ClipCursor(NULL);
        #elif defined(fan_platform_unix)
          XUngrabPointer(fan::sys::m_display, CurrentTime);
          XFixesShowCursor(fan::sys::m_display, DefaultRootWindow(fan::sys::m_display));
          XFlush(fan::sys::m_display);
          //XDefineCursor(fan::sys::m_display, m_window_handle, XC_arrow);
          
        #endif
        }
        else {
        #if defined(fan_platform_windows)
          SetCapture(m_window_handle);
          RECT rect;
          GetClientRect(m_window_handle, &rect);

          POINT ul;
          ul.x = rect.left;
          ul.y = rect.top;

          POINT lr;
          lr.x = rect.right;
          lr.y = rect.bottom;

          MapWindowPoints(m_window_handle, nullptr, &ul, 1);
          MapWindowPoints(m_window_handle, nullptr, &lr, 1);

          rect.left = ul.x;
          rect.top = ul.y;

          rect.right = lr.x;
          rect.bottom = lr.y;

          SetCursor(NULL);
          ClipCursor(&rect);
        #elif defined(fan_platform_unix)

          XGrabPointer(
              fan::sys::m_display, m_window_handle,
              True,
              ButtonPressMask | ButtonReleaseMask | PointerMotionMask,
              GrabModeAsync, GrabModeAsync,
              None, None, CurrentTime
          );

          //XDefineCursor(fan::sys::m_display, m_window_handle, invisibleCursor);
          XFixesHideCursor(fan::sys::m_display, DefaultRootWindow(fan::sys::m_display));
          XFlush(fan::sys::m_display);
        #endif
        }
    }

		// use fan::window_t::resolutions for window sizes
		void set_full_screen(const fan::vec2i& size = uninitialized) {
      flag_values.m_size_mode = fan::window_t::mode::full_screen;

      fan::vec2i new_size;

      if (size == uninitialized) {
        new_size = fan::sys::get_screen_resolution();
      }
      else {
        new_size = size;
      }

      #ifdef fan_platform_windows

      this->set_resolution(new_size, fan::window_t::get_size_mode());

      this->set_windowed_full_screen();

      #elif defined(fan_platform_unix)

      this->set_windowed_full_screen(); // yeah

      #endif

    }
		void set_windowed_full_screen(const fan::vec2i& size = uninitialized) {
      flag_values.m_size_mode = fan::window_t::mode::borderless;

      fan::vec2i new_size;

      if (size == uninitialized) {
        new_size = fan::sys::get_screen_resolution();
      }
      else {
        new_size = size;
      }

      #ifdef fan_platform_windows

      DWORD dwStyle = GetWindowLong(m_window_handle, GWL_STYLE);

      MONITORINFO mi = {sizeof(mi)};

      if (GetMonitorInfo(MonitorFromWindow(m_window_handle, MONITOR_DEFAULTTOPRIMARY), &mi)) {
        SetWindowLong(m_window_handle, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
        SetWindowPos(
          m_window_handle, HWND_TOP,
          mi.rcMonitor.left, mi.rcMonitor.top,
          new_size.x,
          new_size.y,
          SWP_NOOWNERZORDER | SWP_FRAMECHANGED
        );
      }

      #elif defined(fan_platform_unix)

      struct MwmHints {
        unsigned long flags;
        unsigned long functions;
        unsigned long decorations;
        long input_mode;
        unsigned long status;
      };

      enum {
        MWM_HINTS_FUNCTIONS = (1L << 0),
        MWM_HINTS_DECORATIONS = (1L << 1),

        MWM_FUNC_ALL = (1L << 0),
        MWM_FUNC_RESIZE = (1L << 1),
        MWM_FUNC_MOVE = (1L << 2),
        MWM_FUNC_MINIMIZE = (1L << 3),
        MWM_FUNC_MAXIMIZE = (1L << 4),
        MWM_FUNC_CLOSE = (1L << 5)
      };

      Atom mwmHintsProperty = XInternAtom(fan::sys::m_display, "_MOTIF_WM_HINTS", 0);
      struct MwmHints hints;
      hints.flags = MWM_HINTS_DECORATIONS;
      hints.functions = 0;
      hints.decorations = 0;
      XChangeProperty(fan::sys::m_display, m_window_handle, mwmHintsProperty, mwmHintsProperty, 32,
        PropModeReplace, (unsigned char*)&hints, 5);

      XMoveResizeWindow(fan::sys::m_display, m_window_handle, 0, 0, size.x, size.y);

      #endif

    }
		void set_windowed(const fan::vec2i& size = uninitialized) {
      flag_values.m_size_mode = mode::windowed;

      fan::vec2i new_size;
      if (size == uninitialized) {
        new_size = this->get_previous_size();
      }
      else {
        new_size = size;
      }

      #ifdef fan_platform_windows

      this->set_resolution(0, fan::window_t::get_size_mode());

      const fan::vec2i position = fan::sys::get_screen_resolution() / 2 - new_size / 2;

      ShowWindow(m_window_handle, SW_SHOW);

      SetWindowLongPtr(m_window_handle, GWL_STYLE, WS_OVERLAPPEDWINDOW | WS_VISIBLE);

      SetWindowPos(
        m_window_handle,
        0,
        position.x,
        position.y,
        new_size.x,
        new_size.y,
        SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED
      );

      #elif defined(fan_platform_unix)



      #endif
    }

		void set_resolution(const fan::vec2i& size, const mode& mode) const {
      if (mode == mode::full_screen) {
        fan::sys::set_screen_resolution(size);
      }
      else {
        fan::sys::reset_screen_resolution();
      }
    }

		mode get_size_mode() const {
      return flag_values.m_size_mode;
    }
		void set_size_mode(const mode& mode) {
      if (flag_values.m_size_mode == mode) {
        return;
      }

      switch (mode) {
        case mode::windowed: {
          set_windowed();
          break;
        }
        case mode::full_screen: {
          set_full_screen();
          break;
        }
        case mode::borderless: {
          set_windowed_full_screen();
          break;
        }
      }
      flag_values.m_size_mode = mode;
    }

		template <uintptr_t flag, typename T = 
			typename std::conditional<flag & fan::window_t::flags::no_mouse, bool,
			typename std::conditional<flag & fan::window_t::flags::no_resize, bool,
			typename std::conditional<flag & fan::window_t::flags::mode, fan::window_t::mode, int
			>>>::type>
			constexpr void set_flag_value(T value) {
			if constexpr(static_cast<bool>(flag & fan::window_t::flags::no_mouse)) {
        flag_values.m_no_mouse = value;
        lock_cursor_and_set_invisible(value);
			}
			else if constexpr(static_cast<bool>(flag & fan::window_t::flags::no_resize)) {
				flag_values.m_no_resize = value;
			}
			else if constexpr(static_cast<bool>(flag & fan::window_t::flags::mode)) {
				if ((int)value > (int)fan::window_t::mode::full_screen) {
					fan::throw_error("fan window error: failed to set window mode flag to: " + fan::to_string((int)value));
				}
				flag_values.m_size_mode = (fan::window_t::mode)value;
			}
			else if constexpr (static_cast<bool>(flag & fan::window_t::flags::borderless)) {
				flag_values.m_size_mode = value ? fan::window_t::mode::borderless : flag_values.m_size_mode;
			}
			else if constexpr (static_cast<bool>(flag & fan::window_t::flags::full_screen)) {
				flag_values.m_size_mode = value ? fan::window_t::mode::full_screen : flag_values.m_size_mode;
			}
		}

		template <uint64_t flags>
		constexpr void set_flags() {
			// clang requires manual casting (c++11-narrowing)
			if constexpr(static_cast<bool>(flags & fan::window_t::flags::no_mouse)) {
				fan::window_t::flag_values.m_no_mouse = true;
			}
			if constexpr (static_cast<bool>(flags & fan::window_t::flags::no_resize)) {
				fan::window_t::flag_values.m_no_resize = true;
			}
			if constexpr (static_cast<bool>(flags & fan::window_t::flags::borderless)) {
				fan::window_t::flag_values.m_size_mode = fan::window_t::mode::borderless;
			}
			if constexpr (static_cast<bool>(flags & fan::window_t::flags::full_screen)) {
				fan::window_t::flag_values.m_size_mode = fan::window_t::mode::full_screen;
			}
		}

		#define BLL_set_prefix buttons_callback
		#define BLL_set_NodeData mouse_buttons_cb_t data;
		#include "cb_list_builder_settings.h"
		#include _FAN_PATH(BLL/BLL.h)

		#define BLL_set_prefix keys_callback
		#define BLL_set_NodeData keyboard_keys_cb_t data;
		#include "cb_list_builder_settings.h"
		#include _FAN_PATH(BLL/BLL.h)

		#define BLL_set_prefix key_callback
		#define BLL_set_NodeData keyboard_cb_store_t data;
		#include "cb_list_builder_settings.h"
		#include _FAN_PATH(BLL/BLL.h)

		#define BLL_set_prefix text_callback
		#define BLL_set_NodeData text_cb_t data;
		#include "cb_list_builder_settings.h"
		#include _FAN_PATH(BLL/BLL.h)

		#define BLL_set_prefix move_callback
		#define BLL_set_NodeData move_cb_t data;
		#include "cb_list_builder_settings.h"
		#include _FAN_PATH(BLL/BLL.h)

		#define BLL_set_prefix resize_callback
		#define BLL_set_NodeData resize_cb_t data;
		#include "cb_list_builder_settings.h"
		#include _FAN_PATH(BLL/BLL.h)

		#define BLL_set_prefix close_callback
		#define BLL_set_NodeData close_cb_t data;
		#include "cb_list_builder_settings.h"
		#include _FAN_PATH(BLL/BLL.h)

		#define BLL_set_prefix mouse_position_callback
		#define BLL_set_NodeData mouse_move_cb_t data;
		#include "cb_list_builder_settings.h"
		#include _FAN_PATH(BLL/BLL.h)

    #define BLL_set_prefix mouse_motion_callback
		#define BLL_set_NodeData mouse_motion_cb_t data;
		#include "cb_list_builder_settings.h"
		#include _FAN_PATH(BLL/BLL.h)

		buttons_callback_NodeReference_t add_buttons_callback(mouse_buttons_cb_t function) {
      auto nr = m_buttons_callback.NewNodeLast();

      m_buttons_callback[nr].data = function;
      return nr;
    }
		void remove_buttons_callback(buttons_callback_NodeReference_t id) {
      m_buttons_callback.Unlink(id);
      m_buttons_callback.Recycle(id);
    }

		keys_callback_NodeReference_t add_keys_callback(keyboard_keys_cb_t function) {
      auto nr = m_keys_callback.NewNodeLast();
      m_keys_callback[nr].data = function;
      return nr;
    }
		void remove_keys_callback(keys_callback_NodeReference_t id) {
      m_keys_callback.Unlink(id);
      m_keys_callback.Recycle(id);
    }

		key_callback_NodeReference_t add_key_callback(uint16_t key, keyboard_state state, keyboard_key_cb_t function) {
      auto nr = m_key_callback.NewNodeLast();
      m_key_callback[nr].data = keyboard_cb_store_t{key, state, function, };
      return nr;
    }
		void edit_key_callback(key_callback_NodeReference_t id, uint16_t key, keyboard_state state) {
      m_key_callback[id].data.key = key;
      m_key_callback[id].data.state = state;
    }
		void remove_key_callback(key_callback_NodeReference_t id) {
      m_key_callback.unlrec(id);
    }

		text_callback_NodeReference_t add_text_callback(text_cb_t function) {
      auto nr = m_text_callback.NewNodeLast();
      m_text_callback[nr].data = function;
      return nr;
    }

		void remove_text_callback(text_callback_NodeReference_t id) {
      m_text_callback.Unlink(id);
      m_text_callback.Recycle(id);
    }

		close_callback_NodeReference_t add_close_callback(close_cb_t function) {
      auto nr = m_close_callback.NewNodeLast();
      m_close_callback[nr].data = function;
      return nr;
    }
		void remove_close_callback(close_callback_NodeReference_t id) {
      m_close_callback.Unlink(id);
      m_close_callback.Recycle(id);
    }

		mouse_position_callback_NodeReference_t add_mouse_move_callback(mouse_move_cb_t function) {
      auto nr = m_mouse_position_callback.NewNodeLast();
      m_mouse_position_callback[nr].data = function;
      return nr;
    }
		void remove_mouse_move_callback(mouse_position_callback_NodeReference_t id) {
      m_mouse_position_callback.Unlink(id);
      m_mouse_position_callback.Recycle(id);
    }

    mouse_motion_callback_NodeReference_t add_mouse_motion(mouse_motion_cb_t function) {
      auto nr = m_mouse_motion_callback.NewNodeLast();
      m_mouse_motion_callback[nr].data = function;
      return nr;
    }
		void erase_mouse_motion_callback(mouse_motion_callback_NodeReference_t id) {
      m_mouse_motion_callback.Unlink(id);
      m_mouse_motion_callback.Recycle(id);
    }

		resize_callback_NodeReference_t add_resize_callback(resize_cb_t function) {
      auto nr = m_resize_callback.NewNodeLast();
      m_resize_callback[nr].data = function;
      return nr;
    }
		void remove_resize_callback(resize_callback_NodeReference_t id) {
      m_resize_callback.Unlink(id);
      m_resize_callback.Recycle(id);
    }

		move_callback_NodeReference_t add_move_callback(move_cb_t function) {
      auto nr = m_move_callback.NewNodeLast();
      m_move_callback[nr].data = function;
      return nr;
    }
		void remove_move_callback(move_callback_NodeReference_t idt) {
      m_move_callback.unlrec(idt);
    }


		void set_background_color(const fan::color& color) {
      m_background_color = color;
    }

		fan::window_handle_t get_handle() const {
      return m_window_handle;
    }

		// when finished getting fps returns fps otherwise 0
		// ms
		uintptr_t get_fps(uint32_t frame_update = 1, bool window_title = true, bool print = true) {
      int temp_fps = m_fps_counter;
      auto time_diff = (m_current_frame - m_last_frame) / 1e+9;
      if (time_diff >= 1.0 / frame_update) {
        fan::string fps_info;
        if (window_title || print) {
          f64_t fps, frame_time;
          fps = (1.0 / time_diff) * m_fps_counter;
          frame_time = time_diff / m_fps_counter;
          fps_info.append(
            fan::string("fps: ") +
            fan::to_string(fps) +
            fan::string(" frame time: ") +
            fan::to_string(frame_time) +
            fan::string(" ms")
          );
          m_last_frame = m_current_frame;
          m_fps_counter = 0;
        }
        if (window_title) {
          this->set_name(fps_info.c_str());
        }
        if (print) {
          fan::print(fps_info);
        }
        return temp_fps;
      }

      m_fps_counter++;
      return 0;
    }


		bool focused() const {
      #ifdef fan_platform_windows
      return m_focused;
      #elif defined(fan_platform_unix)
      return 1;
      #endif
    }

		void destroy_window_internal() {
      fan::erase_window_id(this->m_window_handle);

      #if defined(fan_platform_windows)

      if (!m_window_handle || !m_hdc
        #if fan_renderer == fan_renderer_opengl
        || !m_context
        #endif
        ) {
        return;
      }

      PostQuitMessage(0);

      #if fan_renderer == fan_renderer_opengl
      wglMakeCurrent(m_hdc, 0);
      #endif

      ReleaseDC(m_window_handle, m_hdc);
      DestroyWindow(m_window_handle);

      #elif defined(fan_platform_unix)

      if (!fan::sys::m_display || !m_visual || !m_window_attribs.colormap) {
        return;
      }

      cleanupHandler(fan::sys::m_display);

      XSetIOErrorHandler(NULL);

      XFree(m_visual);
      XFreeColormap(fan::sys::m_display, m_window_attribs.colormap);
      XDestroyWindow(fan::sys::m_display, m_window_handle);

      //    glXDestroyContext(fan::sys::m_display, m_context);

      XCloseDisplay(fan::sys::m_display);
      #if fan_debug >= fan_debug_low
      m_visual = 0;
      m_window_attribs.colormap = 0;
      #endif

      #endif
    }
		void destroy_window() {
      destroy_window_internal();
    }


		uint16_t get_current_key() const {
      return m_current_key;
    }

		fan::vec2i get_raw_mouse_offset() const {
      return m_raw_mouse_offset;
    }

		uint32_t handle_events() {

      this->calculate_delta_time();

      if (m_max_fps) {

        uint64_t dt = fan::time::clock::now() - m_fps_next_tick;

        uint64_t goal_fps = m_max_fps;

        int64_t frame_time = std::pow(10, 9) * (1.0 / goal_fps);
        frame_time -= dt;

        fan::delay(fan::time::nanoseconds(std::max((int64_t)0, frame_time)));

        m_fps_next_tick = fan::time::clock::now();
      }

      if (call_mouse_motion_cb) {
        auto it = m_mouse_motion_callback.GetNodeFirst();

        while (it != m_mouse_motion_callback.dst) {

          mouse_motion_cb_data_t cbd;
          cbd.window = this;
          cbd.motion = m_average_motion;
          m_mouse_motion_callback[it].data(cbd);

          it = it.Next(&m_mouse_motion_callback);
        }
        const fan::vec2i center = fan::sys::get_screen_resolution() / 2;
        m_may_center = center;
        m_mouse_motion = 0;
        m_average_motion = 0;
        call_mouse_motion_cb = false;
      }

      if (call_mouse_move_cb) {

        auto it = m_mouse_position_callback.GetNodeFirst();
        while (it != m_mouse_position_callback.dst) {

          m_mouse_position_callback.StartSafeNext(it);

          mouse_move_cb_data_t cbd;
          cbd.window = this;
          cbd.position = m_mouse_position;
          m_mouse_position_callback[it].data(cbd);

          it = m_mouse_position_callback.EndSafeNext();
        }
      }

      m_previous_mouse_position = m_mouse_position;
      call_mouse_move_cb = false;

      #ifdef fan_platform_windows

      MSG msg{};

      while (PeekMessageW(&msg, m_window_handle, 0, 0, PM_REMOVE))
      {
        switch (msg.message) {
          case WM_SYSKEYDOWN:
          case WM_KEYDOWN:
          {
            auto window = fan::get_window_by_id(msg.hwnd);

            if (!window) {
              break;
            }

            uint16_t key;
            key = fan::window_input::convert_scancode_to_fan((msg.lParam >> 16) & 0x1ff);

            bool repeat = msg.lParam & (1 << 30);

            keyboard_keys_cb_data_t cdb;
            cdb.window = window;
            cdb.key = key;
            cdb.state = repeat ? fan::keyboard_state::repeat : fan::keyboard_state::press;
            cdb.scancode = (msg.lParam >> 16) & 0x1ff;

            fan::window_t::window_input_action(window->m_window_handle, key);

            window->m_current_key = key;

            m_keycode_action_map[msg.wParam] = true;
            m_scancode_action_map[cdb.scancode] = true;

            auto it = window->m_keys_callback.GetNodeFirst();

            while (it != window->m_keys_callback.dst) {

              window->m_keys_callback.StartSafeNext(it);

              window->m_keys_callback[it].data(cdb);

              it = window->m_keys_callback.EndSafeNext();
            }

            break;
          }
          case WM_CHAR:
          {
            auto window = fan::get_window_by_id(msg.hwnd);

            if (!window) {
              break;
            }

            if (msg.wParam < 8) {
              window->m_reserved_flags |= msg.wParam;
            }
            else {

              bool found = false;

              if (!found) {

                uint16_t fan_key = fan::window_input::convert_scancode_to_fan((msg.lParam >> 16) & 0x1ff);

                found = false;

                for (auto i : banned_keys) {
                  if (fan_key == i) {

                    found = true;
                    break;
                  }
                }

                if (!found) {

                  auto src = msg.wParam + (window->m_reserved_flags << 8);

                  UINT u = VkKeyScan(src);
                  UINT high_byte = HIBYTE(u);
                  UINT low_byte = LOBYTE(u);
                  fan::string str;
                  fan::utf16_to_utf8((wchar_t*)&src, &str);
                  src = str.get_utf8(0);
                  if (m_prev_text_flag == u) {
                    auto it = window->m_text_callback.GetNodeFirst();

                    while (it != window->m_text_callback.dst) {

                      fan::window_t::text_cb_data_t d;
                      d.character = src;
                      d.window = window;
                      d.state = fan::keyboard_state::repeat;
                      window->m_text_callback[it].data(d);

                      it = it.Next(&window->m_text_callback);
                    }
                    break;
                  }
                  if (m_prev_text_flag) {
                    window->m_keymap[fan::key_shift] = 0;
                    window->m_keymap[fan::key_control] = 0;
                    window->m_keymap[fan::key_alt] = 0;
                    window->m_keymap[fan::window_input::convert_scancode_to_fan(LOBYTE(m_prev_text_flag))] = false;
                  }
                  m_prev_text_flag = u;
                  window->m_keymap[fan::key_shift] = high_byte & 0x1;
                  window->m_keymap[fan::key_control] = high_byte & 0x2;
                  window->m_keymap[fan::key_alt] = high_byte & 0x4;
                  window->m_keymap[fan::window_input::convert_scancode_to_fan(low_byte)] = true;
                  m_prev_text = src;

                  // UTF-8
                  // auto utf8_str = fan::utf16_to_utf8((wchar_t*)&src);

                  /* uint32_t value = 0;

                   for (int i = 0, j = 0; i < utf8_str.size(); i++, j += 0x08) {
                     value |= (uint8_t)utf8_str[i] << j;
                   }*/

                  auto it = window->m_text_callback.GetNodeFirst();

                  while (it != window->m_text_callback.dst) {

                    fan::window_t::text_cb_data_t d;
                    d.character = src;
                    d.window = window;
                    d.state = fan::keyboard_state::press;
                    window->m_text_callback[it].data(d);

                    it = it.Next(&window->m_text_callback);
                  }
                }

                window->m_reserved_flags = 0;
              }

            }

            break;
          }
          case WM_LBUTTONDOWN:
          {
            auto window = fan::get_window_by_id(msg.hwnd);

            if (!window) {
              break;
            }
            const uint16_t button = fan::input::mouse_left;

            fan::window_t::window_input_mouse_action(window->m_window_handle, button);

            auto it = window->m_buttons_callback.GetNodeFirst();

            while (it != window->m_buttons_callback.dst) {

              mouse_buttons_cb_data_t cbd;
              cbd.window = window;
              cbd.button = button;
              cbd.state = fan::mouse_state::press;
              window->m_buttons_callback[it].data(cbd);

              it = it.Next(&window->m_buttons_callback);
            }

            break;
          }
          case WM_RBUTTONDOWN:
          {
            auto window = fan::get_window_by_id(msg.hwnd);

            if (!window) {
              break;
            }

            const uint16_t button = fan::input::mouse_right;

            fan::window_t::window_input_mouse_action(window->m_window_handle, button);

            auto it = window->m_buttons_callback.GetNodeFirst();

            while (it != window->m_buttons_callback.dst) {

              mouse_buttons_cb_data_t cbd;
              cbd.window = window;
              cbd.button = button;
              cbd.state = fan::mouse_state::press;
              window->m_buttons_callback[it].data(cbd);

              it = it.Next(&window->m_buttons_callback);
            }

            break;
          }
          case WM_MBUTTONDOWN:
          {
            auto window = fan::get_window_by_id(msg.hwnd);

            if (!window) {
              break;
            }

            const uint16_t button = fan::input::mouse_middle;

            fan::window_t::window_input_mouse_action(window->m_window_handle, button);

            auto it = window->m_buttons_callback.GetNodeFirst();

            while (it != window->m_buttons_callback.dst) {

              mouse_buttons_cb_data_t cbd;
              cbd.window = window;
              cbd.button = button;
              cbd.state = fan::mouse_state::press;
              window->m_buttons_callback[it].data(cbd);

              it = it.Next(&window->m_buttons_callback);
            }

            break;
          }
          case WM_SYSKEYUP:
          case WM_KEYUP:
          {
            auto window = fan::get_window_by_id(msg.hwnd);

            if (!window) {
              break;
            }

            uint16_t key = 0;

            key = fan::window_input::convert_scancode_to_fan((msg.lParam >> 16) & 0x1ff);

            window_input_up(window->m_window_handle, key);

            keyboard_keys_cb_data_t cbd;
            cbd.window = window;
            cbd.key = key;
            cbd.state = fan::keyboard_state::release;
            cbd.scancode = (msg.lParam >> 16) & 0x1ff;

            m_keycode_action_map[msg.wParam] = false;
            m_scancode_action_map[cbd.scancode] = false;

            auto it = window->m_keys_callback.GetNodeFirst();

            while (it != window->m_keys_callback.dst) {
              window->m_keys_callback.StartSafeNext(it);

              window->m_keys_callback[it].data(cbd);

              it = window->m_keys_callback.EndSafeNext();
            }

            break;
          }
          case WM_MOUSEWHEEL:
          {
            auto zDelta = GET_WHEEL_DELTA_WPARAM(msg.wParam);

            auto window = fan::get_window_by_id(msg.hwnd);

            fan::window_t::window_input_mouse_action(window->m_window_handle, zDelta < 0 ? fan::input::mouse_scroll_down : fan::input::mouse_scroll_up);

            auto it = window->m_buttons_callback.GetNodeFirst();

            while (it != window->m_buttons_callback.dst) {

              mouse_buttons_cb_data_t cbd;
              cbd.window = window;
              cbd.button = zDelta < 0 ? fan::input::mouse_scroll_down : fan::input::mouse_scroll_up;
              cbd.state = fan::mouse_state::press;
              window->m_buttons_callback[it].data(cbd);

              it = it.Next(&window->m_buttons_callback);
            }

            break;
          }

          case WM_INPUT:
          {
            auto window = fan::get_window_by_id(msg.hwnd);

            if (!window) {
              break;
            }

            if (!window->focused()) {
              break;
            }

            UINT size = sizeof(RAWINPUT);
            BYTE data[sizeof(RAWINPUT)];

            GetRawInputData((HRAWINPUT)msg.lParam, RID_INPUT, data, &size, sizeof(RAWINPUTHEADER));

            RAWINPUT* raw = (RAWINPUT*)data;

            static bool allow_outside = false;

            const auto cursor_in_range = [](const fan::vec2i& position, const fan::vec2& window_size) {
              return position.x >= 0 && position.x < window_size.x &&
                position.y >= 0 && position.y < window_size.y;
              };

            if (raw->header.dwType == RIM_TYPEMOUSE)
            {

              const auto get_cursor_position = [&] {
                POINT p;
                GetCursorPos(&p);
                ScreenToClient(window->m_window_handle, &p);

                return fan::vec2i(p.x, p.y);
                };

              if (raw->header.dwType == RIM_TYPEMOUSE) {
                // get the average sum of motion for one frame
                window->m_average_motion += fan::vec2i(raw->data.mouse.lLastX, raw->data.mouse.lLastY);
                window->call_mouse_motion_cb = true;
              }

              if (fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_LEFT_BUTTON_DOWN) ||
                fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_MIDDLE_BUTTON_DOWN) ||
                fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_RIGHT_BUTTON_DOWN)
                ) {

                const fan::vec2i position(get_cursor_position());

                if (cursor_in_range(position, window->get_size())) {
                  allow_outside = true;
                }

                if (!fan::window_t::flag_values.m_no_mouse) {
                  window->m_mouse_position = position;
                }
              }

              else if (fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_LEFT_BUTTON_UP)) {

                auto it = window->m_buttons_callback.GetNodeFirst();

                while (it != window->m_buttons_callback.dst) {

                  mouse_buttons_cb_data_t cbd;
                  cbd.window = window;
                  cbd.button = fan::input::mouse_left;
                  cbd.state = fan::mouse_state::release;
                  window->m_buttons_callback[it].data(cbd);

                  it = it.Next(&window->m_buttons_callback);
                }

                window_input_up(window->m_window_handle, fan::input::mouse_left); allow_outside = false;
              }

              else if (fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_MIDDLE_BUTTON_UP)) {

                auto it = window->m_buttons_callback.GetNodeFirst();

                while (it != window->m_buttons_callback.dst) {

                  mouse_buttons_cb_data_t cbd;
                  cbd.window = window;
                  cbd.button = fan::input::mouse_middle;
                  cbd.state = fan::mouse_state::release;
                  window->m_buttons_callback[it].data(cbd);

                  it = it.Next(&window->m_buttons_callback);
                }

                window_input_up(window->m_window_handle, fan::input::mouse_right); allow_outside = false;
              }

              else if (fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_RIGHT_BUTTON_UP)) {

                auto it = window->m_buttons_callback.GetNodeFirst();

                while (it != window->m_buttons_callback.dst) {

                  mouse_buttons_cb_data_t cbd;
                  cbd.window = window;
                  cbd.button = fan::input::mouse_right;
                  cbd.state = fan::mouse_state::release;
                  window->m_buttons_callback[it].data(cbd);

                  it = it.Next(&window->m_buttons_callback);
                }

                window_input_up(window->m_window_handle, fan::input::mouse_right); allow_outside = false;
              }

              else if ((raw->data.mouse.usFlags & MOUSE_MOVE_RELATIVE) == MOUSE_MOVE_RELATIVE) {

                const fan::vec2i position(get_cursor_position());

                window->m_raw_mouse_offset = fan::vec2i(raw->data.mouse.lLastX, raw->data.mouse.lLastY);

                if ((!cursor_in_range(position, window->get_size()) && !allow_outside)) {
                  break;
                }
              }

            }
            break;
          }
        }

        TranslateMessage(&msg);
        DispatchMessage(&msg);
      }



      #elif defined(fan_platform_unix)

      if (invisibleCursor == None) {
        Window root = DefaultRootWindow(fan::sys::m_display);
        Pixmap cursor_pixmap = XCreatePixmap(fan::sys::m_display, root, 1, 1, 1);
        XColor black;
        black.red = black.green = black.blue = 0;
        invisibleCursor = XCreatePixmapCursor(fan::sys::m_display, cursor_pixmap, cursor_pixmap, &black, &black, 0, 0);
        //invisibleCursor = XCreateFontCursor(fan::sys::m_display, "XC_boat");
      }

      XEvent event;

      int nevents = XEventsQueued(fan::sys::m_display, QueuedAfterReading);

      while (nevents--) {
        XNextEvent(fan::sys::m_display, &event);
        // if (XFilterEvent(&m_event, m_window))
        // 	continue;

        switch (event.type) {

          case Expose:
          {
            auto window = fan::get_window_by_id(event.xexpose.window);

            if (!window) {
              break;
            }

            XWindowAttributes attribs;
            XGetWindowAttributes(fan::sys::m_display, window->m_window_handle, &attribs);

            window->m_previous_size = window->m_size;
            window->m_size = fan::vec2i(attribs.width, attribs.height);

            auto it = window->m_resize_callback.GetNodeFirst();

            while (it != window->m_resize_callback.dst) {

              resize_cb_data_t cdb;
              cdb.window = window;
              cdb.size = window->m_size;
              window->m_resize_callback[it].data(cdb);

              it = it.Next(&window->m_resize_callback);
            }

            break;
          }
          // case ConfigureNotify:
          // {

          // 	for (const auto& i : window.m_move_callback) {
          // 		if (i) {
          // 			i();
          // 		}
          // 	}

          // 	break;
          // }
          case ClientMessage:
          {
            auto window = fan::get_window_by_id(event.xclient.window);

            if (!window) {
              break;
            }

            if (event.xclient.data.l[0] == (long)m_atom_delete_window) {

              auto it = window->m_close_callback.GetNodeFirst();
              while (it != window->m_close_callback.dst) {
                close_cb_data_t cbd;
                cbd.window = window;
                window->m_close_callback[it].data(cbd);
                it = it.Next(&window->m_close_callback);
              }

              window->m_event_flags |= window_t::events::close;

            }

            break;
          }
          case KeyPress:
          {

            auto window = fan::get_window_by_id(event.xkey.window);

            if (!window) {
              break;
            }

            uint16_t key = fan::window_input::convert_scancode_to_fan(keycode_to_scancode_table[event.xkey.keycode]);

            fan::window_t::window_input_action(window->m_window_handle, key);

            bool repeat = window->m_keycode_action_map[event.xkey.keycode];

            keyboard_keys_cb_data_t cdb{};
            cdb.window = window;
            cdb.key = key;
            cdb.state = repeat ? fan::keyboard_state::repeat : fan::keyboard_state::press;
            if (event.xkey.keycode < fan::window_t::max_keycode) {
              cdb.scancode = keycode_to_scancode_table[event.xkey.keycode];
            }
            window->m_keycode_action_map[event.xkey.keycode] = true;
            window->m_scancode_action_map[(cdb.scancode & 0x7f) | ((!!(cdb.scancode >> 8)) << 8)] = true;

            //fan::print(xcb_get_scancode_name(event.xkey.keycode));

            window->m_current_key = key;

            auto it = window->m_keys_callback.GetNodeFirst();

            while (it != window->m_keys_callback.dst) {

              window->m_keys_callback.StartSafeNext(it);
              window->m_keys_callback[it].data(cdb);

              it = window->m_keys_callback.EndSafeNext();
            }

            KeySym keysym;

            char text[32] = {};

            Status status;
            std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
            std::wstring str;

            Xutf8LookupString(window->m_xic, &event.xkey, text, sizeof(text) - 1, &keysym, &status);

            str = converter.from_bytes(text);

            bool found = false;

            for (auto i : banned_keys) {
              if (key == i) {
                found = true;
                break;
              }
            }

            if (str.size() && !found) {
              if (!str.size()) {

                auto it = window->m_text_callback.GetNodeFirst();

                while (it != window->m_text_callback.dst) {

                  window->m_text_callback.StartSafeNext(it);

                  text_cb_data_t cdb;
                  cdb.window = window;
                  cdb.character = str[0];
                  window->m_text_callback[it].data(cdb);

                  it = window->m_text_callback.EndSafeNext();
                }
              }
              else {

                //auto utf8_str = fan::utf16_to_utf8((wchar_t*)str.data());
                //?
                auto it = window->m_text_callback.GetNodeFirst();

                while (it != window->m_text_callback.dst) {

                  fan::window_t::text_cb_data_t d;
                  d.window = window;
                  d.character = str[0];
                  window->m_text_callback[it].data(d);

                  it = it.Next(&window->m_text_callback);
                }
              }
            }

            break;
          }
          case KeyRelease:
          {

            auto window = fan::get_window_by_id(event.xkey.window);

            if (!window) {
              break;
            }

            if (XEventsQueued(fan::sys::m_display, QueuedAfterReading)) {
              XEvent nev;
              XPeekEvent(fan::sys::m_display, &nev);

              if (nev.type == KeyPress && nev.xkey.time == event.xkey.time &&
                nev.xkey.keycode == event.xkey.keycode) {
                break;
              }
            }

            const uint16_t key = fan::window_input::convert_scancode_to_fan(keycode_to_scancode_table[event.xkey.keycode]);

            window->window_input_up(window->m_window_handle, key);

            keyboard_keys_cb_data_t cdb{};
            cdb.window = window;
            cdb.key = key;
            cdb.state = fan::keyboard_state::release;
            if (event.xkey.keycode < fan::window_t::max_keycode) {
              cdb.scancode = keycode_to_scancode_table[event.xkey.keycode];
            }
            window->m_keycode_action_map[event.xkey.keycode] = false;
            window->m_scancode_action_map[(cdb.scancode & 0x7f) | ((!!(cdb.scancode >> 8)) << 8)] = false;

            auto it = window->m_keys_callback.GetNodeFirst();

            while (it != window->m_keys_callback.dst) {
              window->m_keys_callback.StartSafeNext(it);

              window->m_keys_callback[it].data(cdb);

              it = window->m_keys_callback.EndSafeNext();
            }

            break;
          }
          case MotionNotify:
          {
            auto window = fan::get_window_by_id(event.xmotion.window);

            if (!window) {
              break;
            }

            const fan::vec2i position(event.xmotion.x, event.xmotion.y);

            call_mouse_move_cb = true;

            window->m_previous_mouse_position = window->m_mouse_position;

            window->m_mouse_position = position;
            m_mouse_motion = fan::vec2i(event.xmotion.x_root, event.xmotion.y_root) - m_may_center;
            m_may_center = fan::vec2i(event.xmotion.x_root, event.xmotion.y_root);
            m_average_motion += m_mouse_motion;
            call_mouse_motion_cb = true;

            break;
          }
          case ButtonPress:
          {

            auto window = fan::get_window_by_id(event.xbutton.window);

            if (!window) {
              break;
            }

            /* TODO temp fix said by dürüm */
            uint16_t button;
            if (event.xbutton.button == Button1) {
              button = mouse_left;
            }
            else if (event.xbutton.button == Button2) {
              button = mouse_middle;
            }
            else if (event.xbutton.button == Button3) {
              button = mouse_right;
            }
            else if (event.xbutton.button == Button4) {
              button = mouse_scroll_up;
            }
            else if (event.xbutton.button == Button5) {
              button = mouse_scroll_down;
            }
            else {
              break;
            }

            window->window_input_mouse_action(window->m_window_handle, button);

            auto it = window->m_buttons_callback.GetNodeFirst();

            while (it != window->m_buttons_callback.dst) {

              mouse_buttons_cb_data_t cbd;
              cbd.window = window;
              cbd.button = button;
              cbd.state = fan::mouse_state::press;
              window->m_buttons_callback[it].data(cbd);

              it = it.Next(&window->m_buttons_callback);
            }

            break;
          }
          case ButtonRelease:
          {

            auto window = fan::get_window_by_id(event.xbutton.window);

            if (!window) {
              break;
            }

            if (XEventsQueued(fan::sys::m_display, QueuedAfterReading)) {
              XEvent nev;
              XPeekEvent(fan::sys::m_display, &nev);

              if (nev.type == ButtonPress && nev.xbutton.time == event.xbutton.time &&
                nev.xbutton.button == event.xbutton.button) {
                break;
              }
            }

            /* TODO temp fix said by dürüm */
            uint16_t button;
            if (event.xbutton.button == Button1) {
              button = mouse_left;
            }
            else if (event.xbutton.button == Button2) {
              button = mouse_middle;
            }
            else if (event.xbutton.button == Button3) {
              button = mouse_right;
            }
            else if (event.xbutton.button == Button4) {
              button = mouse_scroll_up;
            }
            else if (event.xbutton.button == Button5) {
              button = mouse_scroll_down;
            }
            else {
              break;
            }

            window->window_input_up(window->m_window_handle, button);

            auto it = window->m_buttons_callback.GetNodeFirst();

            while (it != window->m_buttons_callback.dst) {

              mouse_buttons_cb_data_t cbd;
              cbd.window = window;
              cbd.button = button;
              cbd.state = fan::mouse_state::release;
              window->m_buttons_callback[it].data(cbd);

              it = it.Next(&window->m_buttons_callback);
            }

            break;
          }
          case FocusOut:
          {
            auto fwindow = fan::get_window_by_id(event.xfocus.window);
            if (!fwindow) {
              break;
            }

            std::memset(fwindow->m_scancode_action_map, 0, sizeof(fwindow->m_scancode_action_map));

            for (uint16_t i = 0; i < 255; ++i) {
              auto fkey = fan::window_input::convert_scancode_to_fan(keycode_to_scancode_table[i]);
              if (fwindow->m_keycode_action_map[i] == false) {
                continue;
              }
              if (fkey >= fan::mouse_left) {
                auto it = fwindow->m_buttons_callback.GetNodeFirst();
                while (it != fwindow->m_buttons_callback.dst) {
                  fwindow->m_buttons_callback.StartSafeNext(it);

                  mouse_buttons_cb_data_t cbd;
                  cbd.window = fwindow;
                  cbd.button = fkey;
                  cbd.state = fan::mouse_state::release;
                  fwindow->m_buttons_callback[it].data(cbd);

                  it = fwindow->m_buttons_callback.EndSafeNext();
                }
              }
              else {
                auto it = fwindow->m_keys_callback.GetNodeFirst();
                while (it != fwindow->m_keys_callback.dst) {
                  fwindow->m_keys_callback.StartSafeNext(it);

                  keyboard_keys_cb_data_t cbd{};
                  cbd.window = fwindow;
                  cbd.key = fkey;
                  cbd.state = fan::keyboard_state::release;
                  if (i < fan::window_t::max_keycode) {
                    cbd.scancode = keycode_to_scancode_table[i];
                  }
                  fwindow->m_keycode_action_map[i] = false;
                  fwindow->m_keys_callback[it].data(cbd);

                  it = fwindow->m_keys_callback.EndSafeNext();
                }
              }
            }

            fwindow->m_focused = false;
            break;
          }
        }
      }

      if (flag_values.m_no_mouse == true) {
        int screen_num;
        int screen_width, screen_height;

        screen_num = DefaultScreen(fan::sys::m_display);
        auto root_window = RootWindow(fan::sys::m_display, screen_num);
        screen_width = DisplayWidth(fan::sys::m_display, screen_num);
        screen_height = DisplayHeight(fan::sys::m_display, screen_num);

        XWarpPointer(fan::sys::m_display, None, root_window, 0, 0, 0, 0, screen_width / 2, screen_height / 2);
      }

      #endif
      return m_event_flags;
    }

		bool key_pressed(uint16_t key) const {
      return m_scancode_action_map[fan::window_input::convert_fan_to_scancode(key)];
    }

		static constexpr fan::input banned_keys[]{
			fan::key_enter,
			fan::key_tab,
			fan::key_escape,
			fan::key_backspace,
			fan::key_delete
		};

		struct keystate_t {
			uint16_t key;
			bool press;
		};

		using keymap_t = fan::hector_t<keystate_t>;
		using timer_interval_t = fan::time::milliseconds;

		static void window_input_action(fan::window_handle_t window, uint16_t key) {

      fan::window_t* fwindow;

      #ifdef fan_platform_windows

      fwindow = get_window_by_id(window);

      #elif defined(fan_platform_unix)

      fwindow = get_window_by_id(window);

      #endif

      auto it = fwindow->m_key_callback.GetNodeFirst();

      while (it != fwindow->m_key_callback.dst) {

        fwindow->m_key_callback.StartSafeNext(it);

        if (key != fwindow->m_key_callback[it].data.key || fwindow->m_key_callback[it].data.state == keyboard_state::release) {
          it = fwindow->m_key_callback.EndSafeNext();
          continue;
        }

        keyboard_key_cb_data_t cbd;
        cbd.window = fwindow;
        cbd.key = key;
        fwindow->m_key_callback[it].data.function(cbd);

        it = fwindow->m_key_callback.EndSafeNext();
      }
    }
		FAN_API void window_input_mouse_action(fan::window_handle_t window, uint16_t key) {
      fan::window_t* fwindow;

      #ifdef fan_platform_windows

      fwindow = fan::get_window_by_id(window);

      #elif defined(fan_platform_unix)

      fwindow = this;

      #endif

      auto it = fwindow->m_key_callback.GetNodeFirst();

      while (it != fwindow->m_key_callback.dst) {

        fwindow->m_key_callback.StartSafeNext(it);

        if (key != fwindow->m_key_callback[it].data.key || fwindow->m_key_callback[it].data.state == keyboard_state::release) {
          it = fwindow->m_key_callback.EndSafeNext();
          continue;
        }

        keyboard_key_cb_data_t cbd;
        cbd.window = fwindow;
        cbd.key = key;
        fwindow->m_key_callback[it].data.function(cbd);

        it = fwindow->m_key_callback.EndSafeNext();
      }
    }

		FAN_API void window_input_up(fan::window_handle_t window, uint16_t key) {
      fan::window_t* fwindow;
      #ifdef fan_platform_windows

      fwindow = fan::get_window_by_id(window);

      #elif defined(fan_platform_unix)

      fwindow = this;

      #endif

      if (key <= fan::input::key_menu) {
        fan::window_t::window_input_action_reset(window, key);
      }

      auto it = fwindow->m_key_callback.GetNodeFirst();

      while (it != fwindow->m_key_callback.dst) {

        fwindow->m_key_callback.StartSafeNext(it);

        if (key != fwindow->m_key_callback[it].data.key || fwindow->m_key_callback[it].data.state == keyboard_state::press) {
          it = fwindow->m_key_callback.EndSafeNext();
          continue;
        }

        keyboard_key_cb_data_t cbd;
        cbd.window = fwindow;
        cbd.key = key;
        fwindow->m_key_callback[it].data.function(cbd);

        it = fwindow->m_key_callback.EndSafeNext();
      }
    }
		FAN_API void window_input_action_reset(fan::window_handle_t window, uint16_t key) {

    }

		#ifdef fan_platform_windows

		static LRESULT CALLBACK window_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
      switch (msg) {
        case WM_MOUSEMOVE:
        {
          auto window = fan::get_window_by_id(hwnd);

          if (!window) {
            break;
          }

          const auto get_cursor_position = [&] {
            POINT p;
            GetCursorPos(&p);
            ScreenToClient(window->m_window_handle, &p);

            return fan::vec2i(p.x, p.y);
            };

          const fan::vec2i position(get_cursor_position());

          window->m_previous_mouse_position = window->m_mouse_position;
          window->m_mouse_position = position;
          window->m_mouse_motion = window->m_mouse_position - window->m_previous_mouse_position;

          window->call_mouse_move_cb = true;

          break;
        }
        case WM_MOVE:
        {

          auto window = fan::get_window_by_id(hwnd);

          if (!window) {
            break;
          }

          window->m_position = fan::vec2i(
            static_cast<int>(static_cast<short>(LOWORD(lparam))),
            static_cast<int>(static_cast<short>(HIWORD(lparam)))
          );

          auto it = window->m_move_callback.GetNodeFirst();

          while (it != window->m_move_callback.dst) {
            it = it.Next(&window->m_move_callback);
          }

          break;
        }
        case WM_SIZE:
        {
          fan::window_t* fwindow = fan::get_window_by_id(hwnd);

          if (!fwindow) {
            break;
          }

          RECT rect;
          GetClientRect(hwnd, &rect);

          fwindow->m_previous_size = fwindow->m_size;
          fwindow->m_size = fan::vec2i(rect.right - rect.left, rect.bottom - rect.top);

          #if fan_renderer == fan_renderer_opengl
          wglMakeCurrent(fwindow->m_hdc, m_context);
          #endif

          auto it = fwindow->m_resize_callback.GetNodeFirst();

          while (it != fwindow->m_resize_callback.dst) {

            resize_cb_data_t cbd;
            cbd.window = fwindow;
            cbd.size = fwindow->m_size;
            fwindow->m_resize_callback[it].data(cbd);

            it = it.Next(&fwindow->m_resize_callback);
          }

          break;
        }
        case WM_SETFOCUS:
        {
          fan::window_t* fwindow = fan::get_window_by_id(hwnd);

          if (!fwindow) {
            break;
          }

          fwindow->m_focused = true;
          break;
        }
        case WM_KILLFOCUS:
        {
          fan::window_t* fwindow = fan::get_window_by_id(hwnd);
          if (!fwindow) {
            break;
          }

          std::memset(fwindow->m_scancode_action_map, 0, sizeof(fwindow->m_scancode_action_map));

          for (uint16_t i = fan::first; i != fan::last; i++) {
            auto kd = fan::window_input::convert_fan_to_scancode(i);
            if (fwindow->m_keycode_action_map[kd] == false) {
              continue;
            }
            if (i >= fan::mouse_left) {
              auto it = fwindow->m_buttons_callback.GetNodeFirst();
              while (it != fwindow->m_buttons_callback.dst) {
                fwindow->m_buttons_callback.StartSafeNext(it);

                mouse_buttons_cb_data_t cbd;
                cbd.window = fwindow;
                cbd.button = i;
                cbd.state = fan::mouse_state::release;
                fwindow->m_buttons_callback[it].data(cbd);

                it = fwindow->m_buttons_callback.EndSafeNext();
              }
            }
            else {
              auto it = fwindow->m_keys_callback.GetNodeFirst();
              while (it != fwindow->m_keys_callback.dst) {
                fwindow->m_keys_callback.StartSafeNext(it);

                keyboard_keys_cb_data_t cbd;
                cbd.window = fwindow;
                cbd.key = i;
                cbd.state = fan::keyboard_state::release;
                cbd.scancode = MapVirtualKeyA(i, MAPVK_VK_TO_VSC);
                fwindow->m_keycode_action_map[kd] = false;
                fwindow->m_keys_callback[it].data(cbd);

                it = fwindow->m_keys_callback.EndSafeNext();
              }
            }
          }

          fwindow->m_focused = false;
          break;
        }
        case WM_SYSCOMMAND:
        {
          //auto fwindow = get_window_storage<fan::window_t*>(m_window, stringify(this_window));
          // disable alt action for window
          if (wparam == SC_KEYMENU && (lparam >> 16) <= 0) {
            return 0;
          }

          break;
        }
        case WM_DESTROY:
        {

          PostQuitMessage(0);

          break;
        }
        case WM_CLOSE:
        {
          fan::window_t* fwindow = fan::get_window_by_id(hwnd);

          //if (fwindow->key_press(fan::key_alt)) {
          //	return 0;
          //}

          auto it = fwindow->m_close_callback.GetNodeFirst();

          while (it != fwindow->m_close_callback.dst) {
            close_cb_data_t cbd;
            cbd.window = fwindow;
            fwindow->m_close_callback[it].data(cbd);
            it = it.Next(&fwindow->m_close_callback);
          }

          fwindow->m_event_flags |= fan::window_t::events::close;

          break;
        }
      }

      return DefWindowProc(hwnd, msg, wparam, lparam);
    }

		HDC m_hdc;

	#if fan_renderer == fan_renderer_opengl
		static inline HGLRC m_context;
	#endif


		#elif defined(fan_platform_unix)

		inline static Atom m_atom_delete_window;
		XSetWindowAttributes m_window_attribs;
		XVisualInfo* m_visual;

	//#if fan_renderer == fan_renderer_opengl
		inline static fan::opengl::glx::GLXContext m_context;
	//#endif

		XIM m_xim;
		XIC m_xic;

    static constexpr uint16_t max_keycode = 1024;
    uint16_t keycode_to_scancode_table[max_keycode]{};

    inline static Cursor invisibleCursor = None;

		#endif

		void initialize_window(const fan::string& name, const fan::vec2i& window_size, uint64_t flags) {
      #ifdef fan_platform_windows

      auto instance = GetModuleHandle(NULL);

      WNDCLASS wc = {0};

      auto str = fan::random::string(10);

      wc.lpszClassName = str.c_str();

      wc.lpfnWndProc = fan::window_t::window_proc;

      wc.hCursor = LoadCursor(NULL, IDC_ARROW);

      wc.hInstance = instance;

      RegisterClass(&wc);

      const bool full_screen = flag_values.m_size_mode == fan::window_t::mode::full_screen;
      const bool borderless = flag_values.m_size_mode == fan::window_t::mode::borderless;

      RECT rect = {0, 0, window_size.x, window_size.y};
      AdjustWindowRect(&rect, full_screen || borderless ? WS_POPUP : WS_OVERLAPPEDWINDOW, FALSE);

      const fan::vec2i position = fan::sys::get_screen_resolution() / 2 - window_size / 2;

      if (full_screen) {
        this->set_resolution(window_size, fan::window_t::mode::full_screen);
      }

      m_window_handle = CreateWindow(str.c_str(), name.c_str(),
        (flag_values.m_no_resize ? ((full_screen || borderless ? WS_POPUP : (WS_OVERLAPPED | WS_MINIMIZEBOX | WS_SYSMENU)) | WS_SYSMENU) :
          (full_screen || borderless ? WS_POPUP : WS_OVERLAPPEDWINDOW)) | WS_VISIBLE,
        position.x, position.y,
        rect.right - rect.left, rect.bottom - rect.top,
        0, 0, 0, 0);

      if (!m_window_handle) {
        fan::throw_error("failed to initialize window:" + fan::to_string(GetLastError()));
      }

      RAWINPUTDEVICE r_id[2];
      r_id[0].usUsagePage = HID_USAGE_PAGE_GENERIC;
      r_id[0].usUsage = HID_USAGE_GENERIC_MOUSE;
      r_id[0].dwFlags = RIDEV_INPUTSINK;
      r_id[0].hwndTarget = m_window_handle;

      r_id[1].usUsagePage = HID_USAGE_PAGE_GENERIC;
      r_id[1].usUsage = HID_USAGE_GENERIC_KEYBOARD;
      r_id[1].dwFlags = RIDEV_INPUTSINK;
      r_id[1].hwndTarget = m_window_handle;

      BOOL result = RegisterRawInputDevices(r_id, 2, sizeof(RAWINPUTDEVICE));

      if (!result) {
        fan::throw_error("failed to register raw input:" + fan::to_string(result));
      }

      //ShowCursor(!flag_values::m_no_mouse);
      if (flag_values.m_no_mouse) {
        ShowCursor(false);
        //auto middle = this->get_position() + this->get_size() / 2;
        //SetCursorPos(middle.x, middle.y);
      }

      #elif defined(fan_platform_unix)

      m_xim = 0;
      m_xic = 0;

      // if vulkan
      XInitThreads();

      if (!fan::sys::m_display) {
        fan::sys::m_display = XOpenDisplay(NULL);
        if (!fan::sys::m_display) {
          throw std::runtime_error("failed to initialize window");
        }

      }

      static bool init_once = true;

      if (init_once) {
        fan::sys::m_screen = DefaultScreen(fan::sys::m_display);
        init_once = false;
      }

      void* lib_handle;
      fan::sys::open_lib_handle(shared_library, &lib_handle);
      fan::opengl::glx::PFNGLXGETPROCADDRESSPROC glXGetProcAddress = (decltype(glXGetProcAddress))fan::sys::get_lib_proc(&lib_handle, "glXGetProcAddress");
      if (glXGetProcAddress == nullptr) {
        fan::throw_error("failed to initialize glxGetprocAddress");
      }
      fan::sys::close_lib_handle(&lib_handle);
      static fan::opengl::glx::PFNGLXMAKECURRENTPROC glXMakeCurrent = (decltype(glXMakeCurrent))glXGetProcAddress((const fan::opengl::GLubyte*)"glXMakeCurrent");
      static fan::opengl::glx::PFNGLXGETCURRENTDRAWABLEPROC glXGetCurrentDrawable = (decltype(glXGetCurrentDrawable))glXGetProcAddress((const fan::opengl::GLubyte*)"glXGetCurrentDrawable");
      static fan::opengl::glx::PFNGLXSWAPINTERVALEXTPROC glXSwapIntervalEXT = (decltype(glXSwapIntervalEXT))glXGetProcAddress((const fan::opengl::GLubyte*)"glXSwapIntervalEXT");
      static fan::opengl::glx::PFNGLXDESTROYCONTEXTPROC glXDestroyContext = (decltype(glXDestroyContext))glXGetProcAddress((const fan::opengl::GLubyte*)"glXDestroyContext");
      static fan::opengl::glx::PFNGLXCHOOSEFBCONFIGPROC glXChooseFBConfig = (decltype(glXChooseFBConfig))glXGetProcAddress((const fan::opengl::GLubyte*)"glXChooseFBConfig");
      static fan::opengl::glx::PFNGLXGETVISUALFROMFBCONFIGPROC glXGetVisualFromFBConfig = (decltype(glXGetVisualFromFBConfig))glXGetProcAddress((const fan::opengl::GLubyte*)"glXGetVisualFromFBConfig");
      static fan::opengl::glx::PFNGLXQUERYVERSIONPROC glXQueryVersion = (decltype(glXQueryVersion))glXGetProcAddress((const fan::opengl::GLubyte*)"glXQueryVersion");
      static fan::opengl::glx::PFNGLXGETFBCONFIGATTRIBPROC glXGetFBConfigAttrib = (decltype(glXGetFBConfigAttrib))glXGetProcAddress((const fan::opengl::GLubyte*)"glXGetFBConfigAttrib");
      static fan::opengl::glx::PFNGLXQUERYEXTENSIONSSTRINGPROC glXQueryExtensionsString = (decltype(glXQueryExtensionsString))glXGetProcAddress((const fan::opengl::GLubyte*)"glXQueryExtensionsString");
      static fan::opengl::glx::PFNGLXGETCURRENTCONTEXTPROC glXGetCurrentContext = (decltype(glXGetCurrentContext))glXGetProcAddress((const fan::opengl::GLubyte*)"glXGetCurrentContext");
      static fan::opengl::glx::PFNGLXSWAPBUFFERSPROC glXSwapBuffers = (decltype(glXSwapBuffers))glXGetProcAddress((const fan::opengl::GLubyte*)"glXSwapBuffers");
      static fan::opengl::glx::PFNGLXCREATENEWCONTEXTPROC glXCreateNewContext = (decltype(glXCreateNewContext))glXGetProcAddress((const fan::opengl::GLubyte*)"glXCreateNewContext");
      static fan::opengl::glx::PFNGLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB = (decltype(glXCreateContextAttribsARB))glXGetProcAddress((const fan::opengl::GLubyte*)"glXCreateContextAttribsARB");

      if (!glXGetFBConfigAttrib) {
        fan::throw_error("failed to glXGetFBConfigAttrib");
      }

      int minor_glx = 0, major_glx = 0;
      glXQueryVersion(fan::sys::m_display, &major_glx, &minor_glx);

      constexpr auto major = 3;
      constexpr auto minor = 2;

      if (minor_glx < minor && major_glx <= major) {
        fan::print("fan window error: too low glx version");
        XCloseDisplay(fan::sys::m_display);
        exit(1);
      }

      constexpr uint32_t samples = 0;

      int pixel_format_attribs[] = {
        GLX_X_RENDERABLE, True,
        GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
        GLX_RENDER_TYPE, GLX_RGBA_BIT,
        GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
        GLX_RED_SIZE, 8,
        GLX_GREEN_SIZE, 8,
        GLX_BLUE_SIZE, 8,
        GLX_ALPHA_SIZE, 8,
        GLX_DEPTH_SIZE, 24,
        GLX_STENCIL_SIZE, 8,
        GLX_DOUBLEBUFFER, True,
        GLX_SAMPLE_BUFFERS, samples ? 1 : 0,
        GLX_SAMPLES, samples ? samples : 0,
        None
      };

      int fbcount;
      auto fbc = glXChooseFBConfig(fan::sys::m_display, fan::sys::m_screen, pixel_format_attribs, &fbcount);

      int best_fbc = -1;
      for (int i = 0; i < fbcount; ++i) {
        XVisualInfo* vi = glXGetVisualFromFBConfig(fan::sys::m_display, fbc[i]);
        if (vi == NULL) {
          continue;
        }

        int fb_samples, samp_buf;

        int r;
        r = glXGetFBConfigAttrib(fan::sys::m_display, fbc[i], GLX_SAMPLES, &fb_samples);
        if (r != 0) {
          fan::print("glXGetFBConfigAttrib returned", r);
          continue;
        }
        r = glXGetFBConfigAttrib(fan::sys::m_display, fbc[i], GLX_SAMPLE_BUFFERS, &samp_buf);
        if (r != 0) {
          fan::print("glXGetFBConfigAttrib returned", r);
          continue;
        }

        if (samples == fb_samples) {
          best_fbc = i;
        }

        XFree(vi);
      }
      best_fbc = 0;

      if (best_fbc == -1) {
        fan::throw_error("adviced framebuffer not found");
      }

      fan::opengl::glx::GLXFBConfig bestFbc = fbc[best_fbc];

      XFree(fbc);

      m_visual = glXGetVisualFromFBConfig(fan::sys::m_display, bestFbc);

      if (!m_visual) {
        fan::print("fan window error: failed to create visual");
        XCloseDisplay(fan::sys::m_display);
        exit(1);
      }

      if (fan::sys::m_screen != m_visual->screen) {
        fan::print("fan window error: screen doesn't match with visual screen");
        XCloseDisplay(fan::sys::m_display);
        exit(1);
      }

      std::memset(&m_window_attribs, 0, sizeof(m_window_attribs));

      m_window_attribs.border_pixel = BlackPixel(fan::sys::m_display, fan::sys::m_screen);
      m_window_attribs.background_pixel = WhitePixel(fan::sys::m_display, fan::sys::m_screen);
      m_window_attribs.override_redirect = True;
      m_window_attribs.colormap = XCreateColormap(fan::sys::m_display, RootWindow(fan::sys::m_display, fan::sys::m_screen), m_visual->visual, AllocNone);
      m_window_attribs.event_mask = ExposureMask | KeyPressMask | ButtonPress |
        StructureNotifyMask | ButtonReleaseMask |
        KeyReleaseMask | EnterWindowMask | LeaveWindowMask |
        PointerMotionMask | Button1MotionMask | VisibilityChangeMask |
        ColormapChangeMask | FocusChangeMask;


      const fan::vec2i position = fan::sys::get_screen_resolution() / 2 - window_size / 2;

      m_window_handle = XCreateWindow(
        fan::sys::m_display,
        RootWindow(fan::sys::m_display, fan::sys::m_screen),
        position.x,
        position.y,
        window_size.x,
        window_size.y,
        0,
        m_visual->depth,
        InputOutput,
        m_visual->visual,
        CWBackPixel | CWColormap | CWBorderPixel | CWEventMask | CWCursor,
        &m_window_attribs
      );

      if (flags & fan::window_t::flags::no_resize) {
        auto sh = XAllocSizeHints();
        sh->flags = PMinSize | PMaxSize;
        sh->min_width = sh->max_width = window_size.x;
        sh->min_height = sh->max_height = window_size.y;
        XSetWMSizeHints(fan::sys::m_display, m_window_handle, sh, XA_WM_NORMAL_HINTS);
        XFree(sh);
      }

      this->set_name(name);

      if (!m_atom_delete_window) {
        m_atom_delete_window = XInternAtom(fan::sys::m_display, "WM_DELETE_WINDOW", False);
      }

      XSetWMProtocols(fan::sys::m_display, m_window_handle, &m_atom_delete_window, 1);

      //TODO FIX
      int gl_attribs[] = {
        GLX_CONTEXT_MINOR_VERSION_ARB, minor,
        GLX_CONTEXT_MAJOR_VERSION_ARB, major,
        GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
        0
      };

      bool initialize_context = !m_context;

      const char* glxExts = glXQueryExtensionsString(fan::sys::m_display, fan::sys::m_screen);
      if (!isExtensionSupported(glxExts, "GLX_ARB_create_context") && initialize_context) {
        std::cout << "GLX_ARB_create_context not supported\n";
        m_context = glXCreateNewContext(fan::sys::m_display, bestFbc, GLX_RGBA_TYPE, 0, True);
      }
      else if (initialize_context) {
        m_context = glXCreateContextAttribsARB(fan::sys::m_display, bestFbc, 0, true, gl_attribs);
      }

      initialize_context = false;

      XSync(fan::sys::m_display, True);

      #if fan_renderer == fan_renderer_opengl
      glXMakeCurrent(fan::sys::m_display, m_window_handle, m_context);
      #endif

      XClearWindow(fan::sys::m_display, m_window_handle);
      XMapRaised(fan::sys::m_display, m_window_handle);
      XAutoRepeatOn(fan::sys::m_display);

      m_xim = XOpenIM(fan::sys::m_display, 0, 0, 0);

      if (!m_xim) {
        // fallback to internal input method
        XSetLocaleModifiers("@im=none");
        m_xim = XOpenIM(fan::sys::m_display, 0, 0, 0);
      }

      m_xic = XCreateIC(m_xim,
        XNInputStyle, XIMPreeditNothing | XIMStatusNothing,
        XNClientWindow, m_window_handle,
        XNFocusWindow, m_window_handle,
        NULL);

      XSetICFocus(m_xic);

      generate_keycode_to_scancode_table();

      XSetIOErrorHandler(cleanupHandler);

      #endif

      m_position = position;

      m_previous_size = m_size;

      set_window_by_id(m_window_handle, this);
    }

		// crossplatform variables

    bool m_keycode_action_map[0x100]{};
    bool m_scancode_action_map[0x200]{};

    // for WM_CHAR
    uint16_t m_keymap[fan::last]{};
    uint32_t m_prev_text_flag = 0;
    uint32_t m_prev_text = 0;

		window_handle_t m_window_handle;
		uintptr_t m_max_fps;
		f64_t m_fps_next_tick;
		uintptr_t m_fps_counter;

		f64_t m_last_frame;
		f64_t m_current_frame;
		f64_t m_delta_time;
		fan::string m_name;

		uintptr_t m_flags;
		uint64_t m_event_flags;
		uint64_t m_reserved_flags;

		buttons_callback_t m_buttons_callback;
		keys_callback_t m_keys_callback;
		key_callback_t m_key_callback;
		text_callback_t m_text_callback;
		move_callback_t m_move_callback;
		resize_callback_t m_resize_callback;
		close_callback_t m_close_callback;
		mouse_position_callback_t m_mouse_position_callback;
    mouse_motion_callback_t m_mouse_motion_callback;

		fan::vec2i m_size;
		fan::vec2i m_previous_size;

		fan::vec2i m_position;
		fan::vec2i m_mouse_position;
		fan::vec2i m_raw_mouse_offset;
		fan::vec2i m_previous_mouse_position;

    fan::vec2i m_mouse_motion;
    fan::vec2i m_may_center;

		fan::color m_background_color;

		uint16_t m_current_key;

    bool call_mouse_motion_cb = false;
    fan::vec2i m_average_motion = 0;
		bool call_mouse_move_cb = false;

		bool m_close;
		bool m_focused;

		struct flag_values_t {

			bool m_no_mouse = false;
			bool m_no_resize = false;

			mode m_size_mode = mode::windowed;

		}flag_values;

	};

	namespace io {

		static fan::string get_clipboard_text(fan::window_handle_t window) {

			fan::string copied_text;

			#ifdef fan_platform_windows

			if (!OpenClipboard(nullptr)) {
				fan::throw_error("failed to open clipboard");
			}

			HANDLE data = GetClipboardData(CF_UNICODETEXT);

			if (data == nullptr) {
				return "";
			}

			wchar_t* text = static_cast<wchar_t*>(GlobalLock(data));
			if (text == nullptr) {
				fan::throw_error("copyboard text was nullptr");
			}

      std::wstring str = text;
      copied_text = fan::string(str.begin(), str.end());

			GlobalUnlock(data);

			CloseClipboard();

			#elif defined(fan_platform_unix)

			typedef std::codecvt_utf8<wchar_t> convert_type;
			std::wstring_convert<convert_type, wchar_t> converter;

			Display *display = XOpenDisplay(NULL);

			if (!display) {
				fan::throw_error("failed to open display");
			}

			XEvent ev;
			XSelectionEvent *sev;

			Atom da, incr, type, sel, p;
			int di = 0;
			unsigned long size = 0, dul = 0;
			unsigned char *prop_ret = NULL;

			auto target_window = XCreateSimpleWindow(display, RootWindow(display, DefaultScreen(display)), -10, -10, 1, 1, 0, 0, 0);

			sel = XInternAtom(display, "CLIPBOARD", False);
			p = XInternAtom(display, "PENGUIN", False);

			XConvertSelection(display, sel, XInternAtom(display, "UTF8_STRING", False), p, target_window,
				CurrentTime);

			for (;;)
			{
				XNextEvent(display, &ev);
				switch (ev.type)
				{
					case SelectionNotify:
					{
						sev = (XSelectionEvent*)&ev.xselection;
						if (sev->property == None)
						{
							fan::print("Conversion could not be performed.");
						}
						goto g_done;
					}

				}
			}

			g_done:

			if (XGetWindowProperty(display, target_window, p, 0, 0, False, AnyPropertyType,
				&type, &di, &dul, &size, &prop_ret) != Success) {
				fan::print("failed");
			}

			incr = XInternAtom(display, "INCR", False);

			if (type == incr)
			{
				printf("INCR not implemented\n");
				return "";
			}

			if (XGetWindowProperty(display, target_window, p, 0, size, False, AnyPropertyType,
				&da, &di, &dul, &dul, &prop_ret) != Success) {
				fan::print("failed data");
			}

			if (prop_ret) {
				copied_text = (const char*)prop_ret;
				XFree(prop_ret);
			}
			else {
				fan::print("no prop");
			}

			XDeleteProperty(display, target_window, p);
			XDestroyWindow(display, target_window);
			XCloseDisplay(display);

			#endif

			return copied_text;
		}
	}

}

#else

namespace fan {
	using window_t = int;
}

#endif

//#ifdef fan_subsystem_windows
//
//int main();
//
//int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {
//  return main();
//}
//
//#endif