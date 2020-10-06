#include <FAN/t.h>

#ifdef REQUIRE_GRAPHICS
#include <GLFW/glfw3.h>

constexpr vec2i WINDOWSIZE(1280, 960);

inline GLFWwindow* window;

inline float delta_time;

inline bool is_colliding;

#endif