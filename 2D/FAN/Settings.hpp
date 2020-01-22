#pragma once
#include <GLFW/glfw3.h>
#include <FAN/Vectors.hpp>

void GetFps();

namespace Settings {
	extern float deltaTime;
	static int blockSize = 64;
	static GLFWwindow* window;
}