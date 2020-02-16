#include "Settings.hpp"
#include <cstdio>
#include <GLFW/glfw3.h>

namespace Settings {
	float deltaTime = 0;
	GLFWwindow* window;
}

namespace FanColors {
	Vec3 White = Vec3(1.0, 1.0, 1.0);
	Vec3 Red(1.0, 0.0, 0.0);
	Vec3 Green(0.0, 1.0, 0.0);
	Vec3 Blue(0.0, 0.0, 1.0);
}

void GetFps(bool print) {
	static int fps = 0;
	static double start = glfwGetTime();
	float currentFrame = glfwGetTime();
	static float lastFrame = 0;
	Settings::deltaTime = currentFrame - lastFrame;
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