#include <FAN/types.h>

#ifdef REQUIRE_GRAPHICS
#include <GLFW/glfw3.h>

namespace fan {
	constexpr auto WINDOWSIZE = fan::vec2i(1280, 960);

	inline GLFWwindow* window;

	inline float delta_time;

	inline bool is_colliding;

	namespace flags {
		constexpr bool decorated = true;
	#if defined(FAN_WINDOWS)
		constexpr bool disable_mouse = 1;
	#elif defined(FAN_UNIX)
		constexpr bool disable_mouse = 0;
	#endif
		constexpr bool resizeable = 1;
		constexpr bool antialising = 0;
		constexpr bool full_screen = false;
	}
}

#endif