#pragma once

#include <GL/glew.h>

#include <fan/types.h>

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

#elif defined(FAN_PLATFORM_LINUX)

	constexpr int OPENGL_MINOR_VERSION = GLX_CONTEXT_MINOR_VERSION_ARB;
	constexpr int OPENGL_MAJOR_VERSION = GLX_CONTEXT_MAJOR_VERSION_ARB;

#endif

	#define stringify(name) #name

	fan::vec2i get_resolution();

	constexpr auto uninitialized = -1;

	class window {
	public:

		enum class size_mode {
			windowed,
			windowed_full_screen,
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
		};

		// required type alias for function return types
		using keys_callback_t = std::function<void(uint16_t key, bool action)>;
		using key_callback_t = struct{

			uint16_t key;
			std::function<void()> function;
			bool release;

		};
		using mouse_move_callback_t = std::function<void(const fan::vec2i& position)>;
		using scroll_callback_t = std::function<void(uint16_t key)>;
		using resize_callback_t = std::function<void()>;
		using move_callback_t = std::function<void()>;

		struct flags {
			static constexpr int no_decorate = get_flag_value(0);
			static constexpr int no_mouse = get_flag_value(1);
			static constexpr int no_resize = get_flag_value(2);
			static constexpr int anti_aliasing = get_flag_value(3);
			static constexpr int windowed_full_screen = get_flag_value(4);
			static constexpr int full_screen = get_flag_value(5);
		};

		static constexpr vec2i default_window_size = fan::vec2i(800, 600);
		static constexpr vec2i default_opengl_version = fan::vec2i(3, 1); // major minor
		// for static value storing
		static constexpr int reserved_storage = -1;

		window(const std::string& name, const fan::vec2i& window_size, uint64_t flags = 0);
		~window();

		void loop(const fan::color& background_color, const std::function<void()>& function);

		void swap_buffers() const;

		void set_window_title(const std::string& title) const;

		void calculate_delta_time();
		f_t get_delta_time() const;

		fan::vec2i get_cursor_position() const;

		fan::vec2i get_size() const;
		fan::vec2i get_previous_size() const;

		fan::vec2i get_position() const;

		void set_position(const fan::vec2i& position) const;

		uint_t get_max_fps() const;
		void set_max_fps(uint_t fps);

		bool vsync_enabled() const;
		void vsync(bool value);

		void set_opengl_version(int major, int minor);

		// use fan::window::resolutions for window sizes
		void set_full_screen(const fan::vec2i& size = uninitialized);
		void set_windowed_full_screen(const fan::vec2i& size = uninitialized);
		void set_windowed(const fan::vec2i& size = uninitialized);

		void set_resolution(const fan::vec2i& size, const size_mode& mode) const;

		size_mode get_size_mode() const;
		void set_size_mode(const size_mode& mode);

		template <typename type>
		static type get_window_storage(const fan::window_t& window, const std::string& location);
		static void set_window_storage(const fan::window_t& window, const std::string& location, std::any data);

		keys_callback_t get_keys_callback() const;
		void set_keys_callback(const keys_callback_t& function);

		key_callback_t get_key_callback(uint_t i) const;
		void add_key_callback(uint16_t key, const std::function<void()>& function, bool on_release = false);

		mouse_move_callback_t get_mouse_move_callback(uint_t i) const;
		void add_mouse_move_callback(const mouse_move_callback_t& function);

		scroll_callback_t get_scroll_callback(uint_t i) const;
		void add_scroll_callback(const scroll_callback_t& function);

		resize_callback_t get_resize_callback(uint_t i) const;
		void add_resize_callback(const resize_callback_t& function);

		move_callback_t get_move_callback(uint_t i) const;
		void add_move_callback(const move_callback_t& function);

		fan::window_t get_handle() const;

		// when finished getting fps returns fps otherwise 0
		uint_t get_fps(bool window_title = true, bool print = true);

		bool key_press(uint16_t key) const;

		bool open() const;
		void close();
	
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
		std::vector<mouse_move_callback_t> m_mouse_move_callback;
		std::vector<scroll_callback_t> m_scroll_callback;
		std::vector<move_callback_t> m_move_callback;
		std::vector<resize_callback_t> m_resize_callback;

		keymap_t m_keys_down;

		// for releasing key after pressing it in key callback
		keymap_t m_keys_action;
		keymap_t m_keys_reset;

		fan::vec2i m_size;
		fan::vec2i m_previous_size;

		fan::vec2i m_position;
		
		fan::vec2i m_cursor_position;

		uint_t m_max_fps;

		bool m_received_fps;
		uint_t m_fps;
		fan::timer<> m_fps_timer;

		f_t m_last_frame;
		f_t m_current_frame;
		f_t m_delta_time;

		bool m_vsync;

		bool m_close;

		int m_minor_version;
		int m_major_version;

		size_mode m_size_mode;

	};
}