#pragma once
#include <GLFW/glfw3.h>
#include <FAN/Vectors.hpp>

void GetFps(bool print = true);

static constexpr const auto WINDOWSIZE = __Vec2<int>(900, 900);

//constexpr auto gridSize = __Vec2<int>(10, 10);

namespace Settings {
	extern float deltaTime;
	static constexpr int blockSize = 64;
	extern GLFWwindow* window;
	static constexpr __Vec2<int> view(WINDOWSIZE.x / blockSize, WINDOWSIZE.y / blockSize);
}
