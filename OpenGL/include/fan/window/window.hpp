#pragma once

#include <GL/glew.h>

#include <fan/types/types.hpp>
#include <fan/math/vector.hpp>

#ifdef FAN_PLATFORM_WINDOWS

#include <GL/wglew.h>

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

#include <fan/color.hpp>
#include <fan/time.hpp>
#include <fan/window/window_input.hpp>

#include <type_traits>
#include <any>

namespace fan {

#ifdef FAN_PLATFORM_WINDOWS

	#include <Windows.h>

	#pragma comment(lib, "Gdi32.lib")
	#pragma comment(lib, "User32.lib")

	#undef min
	#undef max

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
		std::size_t operator() (const std::pair<T1, T2> &pair) const
		{
			return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
		}
	};

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

	constexpr auto uninitialized = -1;

	template <typename T>
	constexpr auto initialized(T value) {
		return value != (T)uninitialized;
	}

	class window {
	public:

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

			//constexpr static ;

		};

		// required type alias for function return types
		using keys_callback_t = std::function<void(uint16_t key, bool action)>;
		using key_callback_t = struct{

			uint16_t key;
			std::function<void()> function;
			bool release;

		};

		using mouse_move_callback_t = std::function<void()>;
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
		static constexpr vec2i default_opengl_version = fan::vec2i(4, 5); // major minor
		static constexpr mode default_size_mode = mode::windowed;

		// for static value storing
		static constexpr int reserved_storage = -1;

		window(const std::string& name = default_window_name, const fan::vec2i& window_size = fan::window::default_window_size, uint64_t flags = 0);
		~window();

		void loop(const fan::color& background_color, const std::function<void()>& function);

		void swap_buffers() const;

		void set_title(const std::string& title) const;

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
		void vsync(bool value);

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
			if constexpr(flag & fan::window::flags::no_mouse) {
				flag_values::m_no_mouse = value;
			}
			else if constexpr(flag & fan::window::flags::no_resize) {
				flag_values::m_no_resize = value;
			}
			else if constexpr(flag & fan::window::flags::anti_aliasing) {
				flag_values::m_samples = value;
			}
			else if constexpr(flag & fan::window::flags::mode) {
				if (value > fan::window::mode::full_screen) {
					fan::print("fan window error: failed to set window mode flag to: ", fan::eti(value));
					exit(1);
				}
				flag_values::m_size_mode = value;
			}
			else if constexpr (flag & fan::window::flags::borderless) {
				flag_values::m_size_mode = value ? fan::window::mode::borderless : flag_values::m_size_mode;
			}
			else if constexpr (flag & fan::window::flags::full_screen) {
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

		keys_callback_t get_keys_callback() const;
		void set_keys_callback(const keys_callback_t& function);

		key_callback_t get_key_callback(uint_t i) const;
		void add_key_callback(uint16_t key, const std::function<void()>& function, bool on_release = false);

		mouse_move_position_callback_t get_mouse_move_callback(uint_t i) const;
		void add_mouse_move_callback(const mouse_move_position_callback_t& function);

		void add_mouse_move_callback(const mouse_move_callback_t& function);

		scroll_callback_t get_scroll_callback(uint_t i) const;
		void add_scroll_callback(const scroll_callback_t& function);

		std::function<void()> get_resize_callback(uint_t i) const;
		void add_resize_callback(const std::function<void()>& function);

		std::function<void()> get_move_callback(uint_t i) const;
		void add_move_callback(const std::function<void()>& function);

		void set_background_color(const fan::color& color);

		fan::window_t get_handle() const;

		// when finished getting fps returns fps otherwise 0
		uint_t get_fps(bool window_title = true, bool print = true);

		bool key_press(uint16_t key) const;

		bool open() const;
		void close();

		bool focused() const;
	
	private:

		using keymap_t = std::unordered_map<uint16_t, bool>;

		FAN_API void window_input_action(fan::window_t window, uint16_t key);
		FAN_API void window_input_mouse_action(fan::window_t window, uint16_t key);
		FAN_API void window_input_up(fan::window_t window, uint16_t key);
		FAN_API void window_input_action_reset(fan::window_t window, uint16_t key);

	#ifdef FAN_PLATFORM_WINDOWS

		static LRESULT CALLBACK window_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam);

		HDC m_hdc;

	#elif defined(FAN_PLATFORM_LINUX)

		Display* m_display;
		XEvent m_event;
		int m_screen;
		Atom m_atom_delete_window;
		XSetWindowAttributes m_window_attribs;
		GLXContext m_context;
		XVisualInfo* m_visual;

	#endif

		void reset_keys();

		void initialize_window(const std::string& name, const fan::vec2i& window_size, uint64_t flags);

		void handle_events();
		void handle_resize_callback(const fan::window_t& window, const fan::vec2i& size);
		void handle_move_callback(const fan::window_t& window);
		// crossplatform variables

		window_t m_window;

		static inline std::unordered_map<std::pair<fan::window_t, std::string>, std::any, pair_hash> m_window_storage;

		keys_callback_t m_keys_callback;
		std::vector<key_callback_t> m_key_callback;
		std::vector<mouse_move_position_callback_t> m_mouse_move_position_callback;
		std::vector<mouse_move_callback_t> m_mouse_move_callback;
		std::vector<scroll_callback_t> m_scroll_callback;
		std::vector<std::function<void()>> m_move_callback;
		std::vector<std::function<void()>> m_resize_callback;

		keymap_t m_keys_down;

		// for releasing key after pressing it in key callback
		keymap_t m_keys_action;
		keymap_t m_keys_reset;

		fan::vec2i m_size;
		fan::vec2i m_previous_size;

		fan::vec2i m_position;
		
		fan::vec2i m_mouse_position;

		uint_t m_max_fps;

		bool m_received_fps;
		uint_t m_fps;
		fan::timer<> m_fps_timer;

		f_t m_last_frame;
		f_t m_current_frame;
		f_t m_delta_time;

		bool m_vsync;

		bool m_close;

		struct flag_values {

			static int m_minor_version;
			static int m_major_version;

			static bool m_no_mouse;
			static bool m_no_resize;

			static uint8_t m_samples;

			static mode m_size_mode;

		};

	};
}