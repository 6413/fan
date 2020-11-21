#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <fan/types.h>
#include <fan/time.hpp>
#include <fan/math.hpp>

#define GLFW_MOUSE_SCROLL_UP 200
#define GLFW_MOUSE_SCROLL_DOWN 201

namespace fan {

	class default_callback {
	public:

		template <typename ...Args, typename T>
		void add(const T& function, Args... args) {
			functions.emplace_back(std::bind(function, args...));
		}

		auto& get_function(uint64_t i);

		uint64_t size() const;

	private:
		std::vector<std::function<void()>> functions;
	};

	class KeyCallback {
	public:

		// key <= GLFW_MOUSE_BUTTON_8 required action true
		template <typename ...Args, typename T>
		void add(int key_, int action_, const T& function, Args... args) {
			this->action.push_back(key_ <= GLFW_MOUSE_BUTTON_8 ? true : action_);
			this->key.push_back(key_);
			functions.push_back(std::bind(function, args...));
		}

		int get_action(uint64_t i) const;

		int get_key(uint64_t i) const;

		uint64_t size() const;

		auto get_function(uint64_t i) const;

	private:
		std::vector<std::function<void()>> functions;
		std::vector<int> action;
		std::vector<int> key;
	};
}

namespace fan {

	template <typename T>
	constexpr auto get_flag_value(T value) {
		return (1 << value);
	}

	namespace window_flags {
		constexpr int NO_DECORATE = get_flag_value(0);
		constexpr int NO_MOUSE = get_flag_value(1);
		constexpr int NO_RESIZE = get_flag_value(2);
		constexpr int ANITIALISING = get_flag_value(3);
		constexpr int FULL_SCREEN = get_flag_value(4);
	}

	class window {
	public:

		static constexpr vec2i default_window_size = fan::vec2i(1024, 1024);

		window(GLFWwindow* window);
		window(uint64_t flags = 0, const fan::vec2i& size = fan::window::default_window_size, const std::string& name = "window");
		~window();

		bool close() const;

		int key_press(int key, bool action = 0);

		void set_window_title(const std::string& title) const;

		// updates events as well
		void swap_buffers() const;

		void calculate_delta_time();
		f_t get_delta_time() const;

		fan::vec2i get_cursor_position() const;

		fan::vec2i get_size() const;
		fan::vec2i get_previous_size() const;

		fan::mat4 get_projection_2d(const fan::mat4& projection) const;
		fan::mat4 get_view_translation_2d(const fan::mat4& view) const;

		fan::KeyCallback m_callback_key;
		fan::KeyCallback m_callback_key_release;
		fan::KeyCallback m_callback_scroll;
		fan::default_callback m_callback_window_resize;
		fan::default_callback m_callback_cursor_move;
		fan::default_callback m_callback_character;
		fan::default_callback m_callback_drop;

	private:

		int get_key(const fan::window& window, int key) const;

		static void ErrorCallback(int id, const char* error);
		static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
		static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
		static void CursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
		static void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
		static void CharacterCallback(GLFWwindow* window, unsigned int key);
		static void FrameSizeCallback(GLFWwindow* window, int width, int height);
		static void DropCallback(GLFWwindow* window, int path_count, const char* paths[]);

		GLFWwindow* m_window;
		fan::vec2i m_size;
		fan::vec2i m_previous_size;

		fan::vec2i m_cursor_position;
		
		f_t m_last_frame;
		f_t m_current_frame;
		f_t m_delta_time;

		bool m_input_action[GLFW_KEY_LAST] = { 0 };
		bool m_input_previous_action[GLFW_KEY_LAST] = { 0 };
	};
}