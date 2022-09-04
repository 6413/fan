#pragma once

#include _FAN_PATH(types/types.h)

#ifndef fan_platform_android

#include _FAN_PATH(system.h)

#ifdef fan_compiler_visual_studio
	#define _CRT_SECURE_NO_WARNINGS
#endif

#include _FAN_PATH(graphics/opengl/gl_defines.h)

#include _FAN_PATH(graphics/renderer.h)

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

	#undef index // xos.h

	#include <sys/time.h>
	#include <unistd.h>
	#include <dlfcn.h>

	#define GLX_CONTEXT_MAJOR_VERSION_ARB       0x2091
	#define GLX_CONTEXT_MINOR_VERSION_ARB       0x2092

#undef index

#endif

namespace fan {

#if fan_renderer == fan_renderer_vulkan

	class vulkan;

#endif

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

	fan::vec2i get_screen_resolution();

	template <typename T>
	constexpr auto initialized(T value) {
		return value != (T)uninitialized;
	}

	void set_screen_resolution(const fan::vec2i& size);
	void reset_screen_resolution();

	uintptr_t get_screen_refresh_rate();

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

		using buttons_callback_cb_t = std::function<void(fan::window_t*, uint16_t, key_state)>;
		using keys_callback_cb_t = std::function<void(fan::window_t*, uint16_t, key_state)>;
		using key_callback_cb_t = std::function<void(fan::window_t*, uint16_t key)>;
		using keys_combo_callback_cb_t = std::function<void(fan::window_t*)>;
		using key_combo_callback_t = struct{
			uint16_t last_key;
			fan::hector_t<uint16_t> key_combo;

			keys_combo_callback_cb_t function;
		};

		using text_callback_cb_t = std::function<void(fan::window_t*, uint32_t key)>;
		using mouse_position_callback_cb_t = std::function<void(fan::window_t* window, const fan::vec2i& position)>;
		using close_callback_cb_t = std::function<void(fan::window_t*)>;
		using resize_callback_cb_t = std::function<void(fan::window_t*, const fan::vec2i& window_size)>;
		using move_callback_cb_t = std::function<void(fan::window_t*)>;

		using key_callback_t = struct{

			uint16_t key;
			key_state state;

			key_callback_cb_t function;
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


		window_t() = default;
		window_t(window_t&&) = delete;
		window_t(const window_t&) = delete;
		window_t& operator=(const window_t&) = delete;
		window_t& operator=(window_t&&) = delete;

		void open(const fan::vec2i& window_size = fan::window_t::default_window_size, const std::string& name = default_window_name, uint64_t flags = 0);
		void close();

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

		std::string get_name() const;
		void set_name(const std::string& name);

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
			static constexpr void set_flag_value(T value) {
			if constexpr(static_cast<bool>(flag & fan::window_t::flags::no_mouse)) {
				flag_values::m_no_mouse = value;
			}
			else if constexpr(static_cast<bool>(flag & fan::window_t::flags::no_resize)) {
				flag_values::m_no_resize = value;
			}
			else if constexpr(static_cast<bool>(flag & fan::window_t::flags::mode)) {
				if ((int)value > (int)fan::window_t::mode::full_screen) {
					fan::throw_error("fan window error: failed to set window mode flag to: " + std::to_string((int)value));
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
		typedef uint32_t callback_id_t;

		callback_id_t add_buttons_callback(buttons_callback_cb_t function);
		void remove_buttons_callback(callback_id_t id);

		callback_id_t add_keys_callback(keys_callback_cb_t function);
		void remove_keys_callback(callback_id_t id);

		callback_id_t add_key_callback(uint16_t key, key_state state, key_callback_cb_t function);
		void edit_key_callback(callback_id_t id, uint16_t key, key_state state);
		void remove_key_callback(callback_id_t id);

		// the last key entered is the triggering key
		callback_id_t add_key_combo_callback(uint16_t* keys, uint32_t n, keys_combo_callback_cb_t function);

		callback_id_t add_text_callback(text_callback_cb_t function);
		void remove_text_callback(callback_id_t id);

		callback_id_t add_close_callback(close_callback_cb_t function);
		void remove_close_callback(callback_id_t id);

		callback_id_t add_mouse_move_callback(mouse_position_callback_cb_t function);
		void remove_mouse_move_callback(const callback_id_t id);

		callback_id_t add_resize_callback(resize_callback_cb_t function);
		void remove_resize_callback(callback_id_t id);

		callback_id_t add_move_callback(move_callback_cb_t function);
		void remove_move_callback(callback_id_t idt);

		void set_background_color(const fan::color& color);

		fan::window_handle_t get_handle() const;

		// when finished getting fps returns fps otherwise 0
		uintptr_t get_fps(bool window_title = true, bool print = true);

		bool focused() const;

		void destroy_window();

		uint16_t get_current_key() const;

		fan::vec2i get_raw_mouse_offset() const;

		uint32_t handle_events();

		bool key_pressed(uint16_t key) const;

#if fan_renderer == fan_renderer_vulkan


		fan::vulkan* m_vulkan = nullptr;

#endif

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

		#endif

		void initialize_window(const std::string& name, const fan::vec2i& window_size, uint64_t flags);

		// crossplatform variables

		window_handle_t m_window_handle;

		bll_t<keys_callback_cb_t> m_buttons_callback;
		bll_t<keys_callback_cb_t> m_keys_callback;
		bll_t<key_callback_t> m_key_callback;
		bll_t<key_combo_callback_t> m_key_combo_callback;

		bll_t<text_callback_cb_t> m_text_callback;
		bll_t<move_callback_cb_t> m_move_callback;
		bll_t<resize_callback_cb_t> m_resize_callback;
		bll_t<close_callback_cb_t> m_close_callback;
		bll_t<mouse_position_callback_cb_t> m_mouse_position_callback;

		fan::vec2i m_size;
		fan::vec2i m_previous_size;

		fan::vec2i m_position;

		bool call_mouse_move_cb = false;
		fan::vec2i m_mouse_position;

		uintptr_t m_max_fps;

		f64_t m_fps_next_tick;
		bool m_received_fps;
		uintptr_t m_fps;
		fan::time::clock m_fps_timer;

		f64_t m_last_frame;
		f64_t m_current_frame;
		f64_t m_delta_time;

		bool m_close;

		std::string* m_name;

		uintptr_t m_flags;
		uint64_t m_event_flags;

		uint16_t m_current_key;
		uint64_t m_reserved_flags;

		fan::vec2i m_raw_mouse_offset;

		bool m_focused;

		fan::color m_background_color;

		fan::vec2i m_previous_mouse_position;

		struct flag_values {

			static inline bool m_no_mouse = false;
			static inline bool m_no_resize = false;

			static inline mode m_size_mode;

		};

	};

	namespace io {

		static std::wstring get_clipboard_text(fan::window_handle_t window) {

			std::wstring copied_text;

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
				copied_text = converter.from_bytes((char*)prop_ret);
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