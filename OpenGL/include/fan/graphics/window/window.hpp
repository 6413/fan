#pragma once

#define GLEW_STATIC
#include <GL/glew.h>

#include <fan/types/types.hpp>
#include <fan/math/vector.hpp>

#include <codecvt>
#include <locale>
#include <climits>

#ifdef FAN_PLATFORM_WINDOWS

#include <GL/wglew.h>

#include <Windows.h>

#pragma comment(lib, "Gdi32.lib")
#pragma comment(lib, "User32.lib")

#elif defined(FAN_PLATFORM_LINUX)

#include <iostream>
#include <cstring>

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>
#include <X11/XKBlib.h>

#include <GL/glxew.h>

#include <sys/time.h>
#include <unistd.h>

#endif

#include <fan/graphics/color.hpp>
#include <fan/time/time.hpp>
#include <fan/graphics/window/window_input.hpp>

#include <type_traits>
#include <any>

namespace fan {

	#ifdef FAN_PLATFORM_WINDOWS

	using window_t = HWND;

	#define FAN_API static


	#elif defined(FAN_PLATFORM_LINUX)

	#define FAN_API

	using window_t = Window;

	#endif

	template <typename T>
	constexpr auto get_flag_value(T value) {
		return (1 << value);
	}

	struct pair_hash
	{
		template <class T1, class T2>
		constexpr std::size_t operator() (const std::pair<T1, T2> &pair) const
		{
			return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
		}
	};

	class window;

	#ifdef FAN_PLATFORM_WINDOWS

	constexpr int OPENGL_MINOR_VERSION = WGL_CONTEXT_MINOR_VERSION_ARB;
	constexpr int OPENGL_MAJOR_VERSION = WGL_CONTEXT_MAJOR_VERSION_ARB;

	constexpr int OPENGL_SAMPLE_BUFFER = WGL_SAMPLE_BUFFERS_ARB;
	constexpr int OPENGL_SAMPLES = WGL_SAMPLES_ARB;

	#elif defined(FAN_PLATFORM_LINUX)

	constexpr int OPENGL_MINOR_VERSION = GLX_CONTEXT_MINOR_VERSION_ARB;
	constexpr int OPENGL_MAJOR_VERSION = GLX_CONTEXT_MAJOR_VERSION_ARB;

	constexpr int OPENGL_SAMPLE_BUFFER = GLX_SAMPLE_BUFFERS;
	constexpr int OPENGL_SAMPLES = GLX_SAMPLES;

	#endif

	fan::vec2i get_resolution();

	template <typename T>
	constexpr auto initialized(T value) {
		return value != (T)uninitialized;
	}

	void set_screen_resolution(const fan::vec2i& size);
	void reset_screen_resolution();

	uint_t get_screen_refresh_rate();

	inline std::unordered_map<std::pair<fan::window_t, std::string>, std::any, pair_hash> m_window_storage;

	inline std::unordered_map<fan::window_t, fan::window*> window_id_storage;

	fan::window* get_window_by_id(fan::window_t wid);
	void set_window_by_id(fan::window_t wid, fan::window* window);

	class window {
	public:

		void* data;

		enum class mode {
			not_set,
			windowed,
			borderless,
			full_screen
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

			constexpr static auto size = 12;

			static constexpr fan::vec2i x[12] = { fan::vec2(800, 600),
				fan::vec2i(1024, 768),
				fan::vec2i(1280, 720),
				fan::vec2i(1280, 800),
				fan::vec2i(1280, 900),
				fan::vec2i(1280, 1024),
				fan::vec2(1360, 768),
				fan::vec2i(1440, 900),
				fan::vec2i(1600, 900),
				fan::vec2i(1600, 1024),
				fan::vec2i(1680, 1050),
				fan::vec2i(1920, 1080) };

		};

		// required type alias for function return types
		using keys_callback_t = std::function<void(fan::fstring::value_type key)>;
		using key_callback_t = struct{

			uint16_t key;
			std::function<void()> function;
			bool release;

		};

		using mouse_move_callback_t = std::function<void(fan::window*)>;
		using mouse_move_position_callback_t = std::function<void(const fan::vec2i& position)>;
		using scroll_callback_t = std::function<void(uint16_t key)>;

		struct flags {
			static constexpr int no_mouse = get_flag_value(0);
			static constexpr int no_resize = get_flag_value(1);
			static constexpr int anti_aliasing = get_flag_value(2);
			static constexpr int mode = get_flag_value(3);
			static constexpr int borderless = get_flag_value(4);
			static constexpr int full_screen = get_flag_value(5);
		};

		static constexpr const char* default_window_name = "window";
		static constexpr vec2i default_window_size = fan::vec2i(800, 600);
		static constexpr vec2i default_opengl_version = fan::vec2i(2, 1); // major minor
		static constexpr mode default_size_mode = mode::windowed;

		// for static value storing
		static constexpr int reserved_storage = -1;

		window(const fan::vec2i& window_size = fan::window::default_window_size, const std::string& name = default_window_name, uint64_t flags = 0);
		window(const window& window);
		window(window&& window);

		window& operator=(const window& window);
		window& operator=(window&& window);

		~window();

		void destroy() {
			#ifdef FAN_PLATFORM_WINDOWS

			wglDeleteContext(m_context);

			#elif defined(FAN_PLATFORM_LINUX)

			glXDestroyContext(m_display, m_context);
			XCloseDisplay(m_display);

			m_context = 0;
			m_display = 0;

			#endif

			m_context = 0;
		}

		void execute(const fan::color& background_color, const std::function<void()>& function);

		void loop(const fan::color& background_color, const std::function<void()>& function);

		void swap_buffers() const;

		std::string get_name() const;
		void set_name(const std::string& name);

		void calculate_delta_time();
		f_t get_delta_time() const;

		fan::vec2i get_mouse_position() const;

		fan::vec2i get_size() const;
		fan::vec2i get_previous_size() const;
		void set_size(const fan::vec2i& size);

		fan::vec2i get_position() const;
		void set_position(const fan::vec2i& position);

		uint_t get_max_fps() const;
		void set_max_fps(uint_t fps);

		bool vsync_enabled() const;
		void set_vsync(bool value);

		// use fan::window::resolutions for window sizes
		void set_full_screen(const fan::vec2i& size = uninitialized);
		void set_windowed_full_screen(const fan::vec2i& size = uninitialized);
		void set_windowed(const fan::vec2i& size = uninitialized);

		void set_resolution(const fan::vec2i& size, const mode& mode) const;

		mode get_size_mode() const;
		void set_size_mode(const mode& mode);

		template <typename type_t>
		static type_t get_window_storage(const fan::window_t& window, const std::string& location);
		static void set_window_storage(const fan::window_t& window, const std::string& location, std::any data);

		template <uint_t flag, typename T = 
			typename std::conditional<flag & fan::window::flags::no_mouse, bool,
			typename std::conditional<flag & fan::window::flags::no_resize, bool,
			typename std::conditional<flag & fan::window::flags::anti_aliasing, int,
			typename std::conditional<flag & fan::window::flags::mode, fan::window::mode, int
			>>>>::type>
			static constexpr void set_flag_value(T value) {
			if constexpr(static_cast<bool>(flag & fan::window::flags::no_mouse)) {
				flag_values::m_no_mouse = value;
			}
			else if constexpr(static_cast<bool>(flag & fan::window::flags::no_resize)) {
				flag_values::m_no_resize = value;
			}
			else if constexpr(static_cast<bool>(flag & fan::window::flags::anti_aliasing)) {
				flag_values::m_samples = value;
			}
			else if constexpr(static_cast<bool>(flag & fan::window::flags::mode)) {
				if (value > fan::window::mode::full_screen) {
					fan::print("fan window error: failed to set window mode flag to: ", fan::eti(value));
					exit(1);
				}
				flag_values::m_size_mode = value;
			}
			else if constexpr (static_cast<bool>(flag & fan::window::flags::borderless)) {
				flag_values::m_size_mode = value ? fan::window::mode::borderless : flag_values::m_size_mode;
			}
			else if constexpr (static_cast<bool>(flag & fan::window::flags::full_screen)) {
				flag_values::m_size_mode = value ? fan::window::mode::full_screen : flag_values::m_size_mode;
			}
		}

		template <uint64_t flags>
		static constexpr void set_flags() {
			// clang requires manual casting (c++11-narrowing)
			if constexpr(static_cast<bool>(flags & fan::window::flags::no_mouse)) {
				fan::window::flag_values::m_no_mouse = true;
			}
			if constexpr (static_cast<bool>(flags & fan::window::flags::no_resize)) {
				fan::window::flag_values::m_no_resize = true;
			}
			if constexpr (static_cast<bool>(flags & fan::window::flags::anti_aliasing)) {
				fan::window::flag_values::m_samples = 8;
			}
			if constexpr (static_cast<bool>(flags & fan::window::flags::borderless)) {
				fan::window::flag_values::m_size_mode = fan::window::mode::borderless;
			}
			if constexpr (static_cast<bool>(flags & fan::window::flags::full_screen)) {
				fan::window::flag_values::m_size_mode = fan::window::mode::full_screen;
			}
		}

		keys_callback_t get_keys_callback(uint_t i) const;
		void add_keys_callback(const keys_callback_t& function);

		key_callback_t get_key_callback(uint_t i) const;
		void add_key_callback(uint16_t key, const std::function<void()>& function, bool on_release = false);

		std::function<void()> get_close_callback(uint_t i) const;
		void add_close_callback(const std::function<void()>& function);

		mouse_move_position_callback_t get_mouse_move_callback(uint_t i) const;
		void add_mouse_move_callback(const mouse_move_position_callback_t& function);

		void add_mouse_move_callback(const mouse_move_callback_t& function);

		scroll_callback_t get_scroll_callback(uint_t i) const;
		void add_scroll_callback(const scroll_callback_t& function);

		std::function<void()> get_resize_callback(uint_t i) const;
		void add_resize_callback(const std::function<void()>& function);

		std::function<void()> get_move_callback(uint_t i) const;
		void add_move_callback(const std::function<void()>& function);

		void set_error_callback();

		static void set_background_color(const fan::color& color);

		fan::window_t get_handle() const;

		// when finished getting fps returns fps otherwise 0
		uint_t get_fps(bool window_title = true, bool print = true);

		bool key_press(uint16_t key) const;

		bool open() const;
		void close();

		bool focused() const;

		void destroy_window();

		uint16_t get_current_key() const;

		fan::vec2i get_raw_mouse_offset() const;

		std::vector<fan::input> m_key_exceptions{
			fan::key_left,
			fan::key_up,
			fan::key_right,
			fan::key_down,
			fan::key_delete,
			fan::key_enter,
			fan::key_home,
			fan::key_end,
			fan::key_shift,
			fan::key_left_shift,
			fan::key_right_shift
		};

		static void handle_events();

		void auto_close(bool state);

	private:

		using keymap_t = std::unordered_map<uint16_t, bool>;
		using timer_interval_t = fan::milliseconds;

		static void window_input_action(fan::window_t window, uint16_t key);
		FAN_API void window_input_mouse_action(fan::window_t window, uint16_t key);
		FAN_API void window_input_up(fan::window_t window, uint16_t key);
		FAN_API void window_input_action_reset(fan::window_t window, uint16_t key);

		#ifdef FAN_PLATFORM_WINDOWS

		static LRESULT CALLBACK window_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam);

		HDC m_hdc;
		static inline HGLRC m_context;

		#elif defined(FAN_PLATFORM_LINUX)

		inline static Display* m_display;
		inline static int m_screen;
		inline static Atom m_atom_delete_window;
		XSetWindowAttributes m_window_attribs;
		XVisualInfo* m_visual;
		inline static GLXContext m_context;
		XIM m_xim;
		XIC m_xic;

		#endif

		void reset_keys();

		void initialize_window(const std::string& name, const fan::vec2i& window_size, uint64_t flags);

		void handle_resize_callback(const fan::window_t& window, const fan::vec2i& size);
		void handle_move_callback(const fan::window_t& window);
		// crossplatform variables

		window_t m_window;

		std::vector<keys_callback_t> m_keys_callback;
		std::vector<key_callback_t> m_key_callback;
		std::vector<mouse_move_position_callback_t> m_mouse_move_position_callback;
		std::vector<mouse_move_callback_t> m_mouse_move_callback;
		std::vector<scroll_callback_t> m_scroll_callback;
		std::vector<std::function<void()>> m_move_callback;
		std::vector<std::function<void()>> m_resize_callback;
		std::vector<std::function<void()>> m_close_callback;

		keymap_t m_keys_down;

		// for releasing key after pressing it in key callback
		keymap_t m_keys_action;
		keymap_t m_keys_reset;

		fan::vec2i m_size;
		fan::vec2i m_previous_size;

		fan::vec2i m_position;

		fan::vec2i m_mouse_position;

		uint_t m_max_fps;

		f_t m_fps_next_tick;
		bool m_received_fps;
		uint_t m_fps;
		fan::timer<> m_fps_timer;

		f_t m_last_frame;
		f_t m_current_frame;
		f_t m_delta_time;

		bool m_vsync;

		bool m_close;

		std::string m_name;

		uint_t m_flags;

		uint16_t m_current_key;
		uint64_t m_reserved_flags;

		fan::vec2i m_raw_mouse_offset;

		bool m_focused;

		bool m_auto_close;

		struct flag_values {

			static inline int m_minor_version = fan::uninitialized;
			static inline int m_major_version = fan::uninitialized;

			static inline bool m_no_mouse = false;
			static inline bool m_no_resize = false;

			static inline uint8_t m_samples = fan::uninitialized;

			static inline mode m_size_mode;

		};

	};

	namespace io {

		static fan::fstring get_clipboard_text(fan::window_t window) {

			fan::fstring copied_text;

			#ifdef FAN_PLATFORM_WINDOWS

			if (!OpenClipboard(nullptr)) {
				throw std::runtime_error("failed to open clipboard");
			}

			HANDLE data = GetClipboardData(CF_UNICODETEXT);

			if (data == nullptr) {
				throw std::runtime_error("clipboard data was nullptr");
			}

			wchar_t* text = static_cast<wchar_t*>(GlobalLock(data));
			if (text == nullptr) {
				throw std::runtime_error("copyboard text was nullptr");
			}

			copied_text = text;

			GlobalUnlock(data);

			CloseClipboard();

			#elif defined(FAN_PLATFORM_LINUX)

			typedef std::codecvt_utf8<wchar_t> convert_type;
			std::wstring_convert<convert_type, wchar_t> converter;

			Display *display = XOpenDisplay(NULL);

			if (!display) {
				throw std::runtime_error("failed to open display");
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