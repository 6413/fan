#include <iostream>
#include <FAN/Graphics.hpp>
#include <FAN/DBT.hpp>
#include <vector>

int main() {
	glfwSetErrorCallback(GlfwErrorCallback);
	if (!glfwInit()) {
		printf("GLFW ded\n");
		return 0;
	}
	WindowInit();
	Main _Main;
	_Main.shader.Use();
	glfwSetKeyCallback(window, KeyCallback);
	glfwSetCharCallback(window, CharacterCallback);
	glfwSetScrollCallback(window, ScrollCallback);
	glfwSetWindowUserPointer(window, &_Main.camera);
	glfwSetCursorPosCallback(window, CursorPositionCallback);
	glfwSetMouseButtonCallback(window, MouseButtonCallback);
	glfwSetFramebufferSizeCallback(window, FrameSizeCallback);

	Particles particles(10000, Vec2(20), Vec2(100), 2000, Color(1, 0, 0), Color(1, 1, 0));

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		if (KeyPressA(GLFW_MOUSE_BUTTON_LEFT)) {
			particles.add(cursorPos);
		}

		particles.draw();

		if (KeyPress(GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, true);
		}

		GetFps();
		glfwSwapBuffers(window);
		KeysReset();
	}
}