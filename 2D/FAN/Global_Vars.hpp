#include <FAN/Vectors.hpp>

#ifdef REQUIRE_GRAPHICS
#include <GLFW/glfw3.h>

constexpr auto WINDOWSIZE = vec2i(1280, 960);

inline GLFWwindow* window;

inline float delta_time;

inline bool is_colliding;

#endif