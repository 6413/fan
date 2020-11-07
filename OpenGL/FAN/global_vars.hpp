#include <FAN/types.h>

#ifdef REQUIRE_GRAPHICS
#include <GLFW/glfw3.h>

namespace fan {
	constexpr auto WINDOWSIZE = fan::vec2i(1280, 960);

	inline GLFWwindow* window;

	inline float delta_time;

	namespace input {
		inline bool action[GLFW_KEY_LAST] = { 0 };
		inline bool previous_action[GLFW_KEY_LAST] = { 0 };
	}

	inline bool is_colliding;

	namespace flags {
		constexpr bool decorated = true;
		constexpr bool disable_mouse = 1;
		constexpr bool resizeable = 1;
		constexpr bool antialising = 0;
		constexpr bool full_screen = false;
	}
}

#endif