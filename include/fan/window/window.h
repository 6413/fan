#pragma once

#include _FAN_PATH(types/types.h)

#ifndef fan_platform_android

#include _FAN_PATH(system.h)

#ifdef fan_compiler_visual_studio
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
  #include <X11/cursorfont.h>
  #include <X11/Xutil.h>

	#undef index // xos.h

	#include <sys/time.h>
	#include <unistd.h>
	#include <dlfcn.h>

	#define GLX_CONTEXT_MAJOR_VERSION_ARB       0x2091
	#define GLX_CONTEXT_MINOR_VERSION_ARB       0x2092

#undef index

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

	fan::window_t* get_window_by_id(fan::window_handle_t wid);
	void set_window_by_id(fan::window_handle_t wid, fan::window_t* window);
	void erase_window_id(fan::window_handle_t wid);

	struct window_t {

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
			fan::button_state state;
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
			wchar_t character;
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

		window_t(const fan::vec2i& window_size = fan::window_t::default_window_size, const fan::string& name = default_window_name, uint64_t flags = 0);
		~window_t();

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

		fan::string get_name() const;
		void set_name(const fan::string& name);

		void calculate_delta_time();
		f64_t get_delta_time() const;

		fan::vec2i get_mouse_position() const;
		fan::vec2i get_previous_mouse_position() const;

		fan::vec2i get_size() const;
		fan::vec2i get_previous_size() const;
		void set_size(const fan::vec2i& size);

		fan::vec2i get_position() const;
		void set_position(const fan::vec2i& position);

		uintptr_t get_max_fps() const;
		void set_max_fps(uintptr_t fps);

    void lock_cursor_and_set_invisible(bool flag) {

        if (flag == 0) {
       #if defined(fan_platform_windows)
          SetCursor(LoadCursor(NULL, IDC_ARROW));
          ReleaseCapture();
          SetCapture(NULL);

          // unlock the cursor from the client area
          ClipCursor(NULL);
        #elif defined(fan_platform_unix)
          XGrabPointer(
              fan::sys::m_display, m_window_handle,
              True,
              ButtonPressMask | ButtonReleaseMask | PointerMotionMask,
              GrabModeAsync, GrabModeAsync,
              None, None, CurrentTime
          );

    // Do something else here, such as handle window events or process input.

    // Release the grab and restore normal pointer behavior.
          fan::print("switch", XDefineCursor(fan::sys::m_display, m_window_handle, invisibleCursor));
          //;
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
          XUngrabPointer(fan::sys::m_display, CurrentTime);
          fan::print("switch", XDefineCursor(fan::sys::m_display, m_window_handle, invisibleCursor));
        #endif
        }
    }

		// use fan::window_t::resolutions for window sizes
		void set_full_screen(const fan::vec2i& size = uninitialized);
		void set_windowed_full_screen(const fan::vec2i& size = uninitialized);
		void set_windowed(const fan::vec2i& size = uninitialized);

		void set_resolution(const fan::vec2i& size, const mode& mode) const;

		mode get_size_mode() const;
		void set_size_mode(const mode& mode);

		template <uintptr_t flag, typename T = 
			typename std::conditional<flag & fan::window_t::flags::no_mouse, bool,
			typename std::conditional<flag & fan::window_t::flags::no_resize, bool,
			typename std::conditional<flag & fan::window_t::flags::mode, fan::window_t::mode, int
			>>>::type>
			constexpr void set_flag_value(T value) {
			if constexpr(static_cast<bool>(flag & fan::window_t::flags::no_mouse)) {
        flag_values::m_no_mouse = value;
        lock_cursor_and_set_invisible(value);
			}
			else if constexpr(static_cast<bool>(flag & fan::window_t::flags::no_resize)) {
				flag_values::m_no_resize = value;
			}
			else if constexpr(static_cast<bool>(flag & fan::window_t::flags::mode)) {
				if ((int)value > (int)fan::window_t::mode::full_screen) {
					fan::throw_error("fan window error: failed to set window mode flag to: " + fan::to_string((int)value));
				}
				flag_values::m_size_mode = (fan::window_t::mode)value;
			}
			else if constexpr (static_cast<bool>(flag & fan::window_t::flags::borderless)) {
				flag_values::m_size_mode = value ? fan::window_t::mode::borderless : flag_values::m_size_mode;
			}
			else if constexpr (static_cast<bool>(flag & fan::window_t::flags::full_screen)) {
				flag_values::m_size_mode = value ? fan::window_t::mode::full_screen : flag_values::m_size_mode;
			}
		}

		template <uint64_t flags>
		static constexpr void set_flags() {
			// clang requires manual casting (c++11-narrowing)
			if constexpr(static_cast<bool>(flags & fan::window_t::flags::no_mouse)) {
				fan::window_t::flag_values::m_no_mouse = true;
			}
			if constexpr (static_cast<bool>(flags & fan::window_t::flags::no_resize)) {
				fan::window_t::flag_values::m_no_resize = true;
			}
			if constexpr (static_cast<bool>(flags & fan::window_t::flags::borderless)) {
				fan::window_t::flag_values::m_size_mode = fan::window_t::mode::borderless;
			}
			if constexpr (static_cast<bool>(flags & fan::window_t::flags::full_screen)) {
				fan::window_t::flag_values::m_size_mode = fan::window_t::mode::full_screen;
			}
		}

		#define BLL_set_prefix buttons_callback
		#define BLL_set_node_data mouse_buttons_cb_t data;
		#include "cb_list_builder_settings.h"
		#include _FAN_PATH(BLL/BLL.h)

		#define BLL_set_prefix keys_callback
		#define BLL_set_node_data keyboard_keys_cb_t data;
		#include "cb_list_builder_settings.h"
		#include _FAN_PATH(BLL/BLL.h)

		#define BLL_set_prefix key_callback
		#define BLL_set_node_data keyboard_cb_store_t data;
		#include "cb_list_builder_settings.h"
		#include _FAN_PATH(BLL/BLL.h)

		#define BLL_set_prefix text_callback
		#define BLL_set_node_data text_cb_t data;
		#include "cb_list_builder_settings.h"
		#include _FAN_PATH(BLL/BLL.h)

		#define BLL_set_prefix move_callback
		#define BLL_set_node_data move_cb_t data;
		#include "cb_list_builder_settings.h"
		#include _FAN_PATH(BLL/BLL.h)

		#define BLL_set_prefix resize_callback
		#define BLL_set_node_data resize_cb_t data;
		#include "cb_list_builder_settings.h"
		#include _FAN_PATH(BLL/BLL.h)

		#define BLL_set_prefix close_callback
		#define BLL_set_node_data close_cb_t data;
		#include "cb_list_builder_settings.h"
		#include _FAN_PATH(BLL/BLL.h)

		#define BLL_set_prefix mouse_position_callback
		#define BLL_set_node_data mouse_move_cb_t data;
		#include "cb_list_builder_settings.h"
		#include _FAN_PATH(BLL/BLL.h)

    #define BLL_set_prefix mouse_motion_callback
		#define BLL_set_node_data mouse_motion_cb_t data;
		#include "cb_list_builder_settings.h"
		#include _FAN_PATH(BLL/BLL.h)

		buttons_callback_NodeReference_t add_buttons_callback(mouse_buttons_cb_t function);
		void remove_buttons_callback(buttons_callback_NodeReference_t id);

		keys_callback_NodeReference_t add_keys_callback(keyboard_keys_cb_t function);
		void remove_keys_callback(keys_callback_NodeReference_t id);

		key_callback_NodeReference_t add_key_callback(uint16_t key, keyboard_state state, keyboard_key_cb_t function);
		void edit_key_callback(key_callback_NodeReference_t id, uint16_t key, keyboard_state state);
		void remove_key_callback(key_callback_NodeReference_t id);

		text_callback_NodeReference_t add_text_callback(text_cb_t function);
		void remove_text_callback(text_callback_NodeReference_t id);

		close_callback_NodeReference_t add_close_callback(close_cb_t function);
		void remove_close_callback(close_callback_NodeReference_t id);

		mouse_position_callback_NodeReference_t add_mouse_move_callback(mouse_move_cb_t function);
		void remove_mouse_move_callback(mouse_position_callback_NodeReference_t id);

    mouse_motion_callback_NodeReference_t add_mouse_motion(mouse_motion_cb_t function);
		void erase_mouse_motion_callback(mouse_motion_callback_NodeReference_t id);

		resize_callback_NodeReference_t add_resize_callback(resize_cb_t function);
		void remove_resize_callback(resize_callback_NodeReference_t id);

		move_callback_NodeReference_t add_move_callback(move_cb_t function);
		void remove_move_callback(move_callback_NodeReference_t idt);

		void set_background_color(const fan::color& color);

		fan::window_handle_t get_handle() const;

		// when finished getting fps returns fps otherwise 0
		// ms
		uintptr_t get_fps(uint32_t frame_update = 1, bool window_title = true, bool print = true);

		bool focused() const;

		void destroy_window_internal();
		void destroy_window();

		uint16_t get_current_key() const;

		fan::vec2i get_raw_mouse_offset() const;

		uint32_t handle_events();

		bool key_pressed(uint16_t key) const;

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

		static void window_input_action(fan::window_handle_t window, uint16_t key);
		FAN_API void window_input_mouse_action(fan::window_handle_t window, uint16_t key);
		FAN_API void window_input_up(fan::window_handle_t window, uint16_t key);
		FAN_API void window_input_action_reset(fan::window_handle_t window, uint16_t key);

		#ifdef fan_platform_windows

		static LRESULT CALLBACK window_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam);

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

    static fan::string xcb_get_scancode_name(XkbDescPtr KbDesc, uint16_t keycode);
    static fan::string xcb_get_scancode_name(uint16_t keycode);
    void generate_keycode_to_scancode_table();

    static constexpr uint16_t max_keycode = 1024;
    uint16_t keycode_to_scancode_table[max_keycode]{};

    inline static Cursor invisibleCursor = None;

		#endif

		void initialize_window(const fan::string& name, const fan::vec2i& window_size, uint64_t flags);

		// crossplatform variables

    // for WM_CHAR
    uint16_t m_keymap[fan::last]{};
    uint32_t m_prev_text_flag = 0;
    wchar_t m_prev_text = 0;

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

		fan::color m_background_color;

		uint16_t m_current_key;

    bool call_mouse_motion_cb = false;
    fan::vec2i m_average_motion = 0;
		bool call_mouse_move_cb = false;

		bool m_close;
		bool m_focused;

		struct flag_values {

			static inline bool m_no_mouse = false;
			static inline bool m_no_resize = false;

			static inline mode m_size_mode;

		};

	};

	namespace io {

		static fan::wstring get_clipboard_text(fan::window_handle_t window) {

			fan::wstring copied_text;

			#ifdef fan_platform_windows

			if (!OpenClipboard(nullptr)) {
				fan::throw_error("failed to open clipboard");
			}

			HANDLE data = GetClipboardData(CF_UNICODETEXT);

			if (data == nullptr) {
				return L"";
			}

			wchar_t* text = static_cast<wchar_t*>(GlobalLock(data));
			if (text == nullptr) {
				fan::throw_error("copyboard text was nullptr");
			}

			copied_text = text;

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
				return L"";
			}

			if (XGetWindowProperty(display, target_window, p, 0, size, False, AnyPropertyType,
				&da, &di, &dul, &dul, &prop_ret) != Success) {
				fan::print("failed data");
			}

			if (prop_ret) {
				copied_text = converter.from_bytes((char*)prop_ret).data();
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