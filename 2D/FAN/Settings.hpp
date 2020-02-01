#pragma once
#include <GLFW/glfw3.h>
#include <FAN/Vectors.hpp>

void GetFps();

constexpr auto gridSize = __Vec2<int>(5, 5);

namespace Settings {
	extern float deltaTime;
	static int blockSize = 900 / gridSize.x;
	extern GLFWwindow* window;
}
