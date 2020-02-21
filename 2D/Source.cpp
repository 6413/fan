#include <iostream>
#include <FAN/Graphics.hpp>
#include <FAN/DBT.hpp>
#include <vector>
#include <chrono>
#include <FAN/Alloc.hpp>

int main() {

	glfwSetErrorCallback(GlfwErrorCallback);
	if (!glfwInit()) {
		printf("GLFW ded\n");
		return 0;
	}
	WindowInit();
	Camera camera;
	glfwSetKeyCallback(window, KeyCallback);
	glfwSetCharCallback(window, CharacterCallback);
	glfwSetScrollCallback(window, ScrollCallback);
	glfwSetWindowUserPointer(window, &camera);
	glfwSetCursorPosCallback(window, CursorPositionCallback);
	glfwSetMouseButtonCallback(window, MouseButtonCallback);
	glfwSetFramebufferSizeCallback(window, FrameSizeCallback);

	Square line(vec2(windowSize / 2), vec2(100), Color(1, 0, 0));

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		line.draw();
		line.set_position(0, cursorPos - 50);

		if (KeyPress(GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, true);
		}

		GetFps();
		glfwSwapBuffers(window);
		KeysReset();
	}
}