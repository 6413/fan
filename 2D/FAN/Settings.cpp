#include "Settings.hpp"
#include <cstdio>
#include <GLFW/glfw3.h>

namespace Settings {
	float delta_time = 0;
	GLFWwindow* window;
}

void GetFps(bool print) {
	static int fps = 0;
	static double start = glfwGetTime();
	float currentFrame = glfwGetTime();
	static float lastFrame = 0;
	Settings::delta_time = currentFrame - lastFrame;
	lastFrame = currentFrame;
	if ((glfwGetTime() - start) > 1.0) {
		if (print) {
			printf("%d\n", fps);
		}

		fps = 0;
		start = glfwGetTime();
	}
	fps++;
}