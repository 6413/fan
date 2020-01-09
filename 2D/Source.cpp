#include <iostream>
#include <FAN/Graphics.hpp>
#include <FAN/DBT.hpp>

int main() {
	glfwSetErrorCallback(GlfwErrorCallback);
	if (!glfwInit()) {
		printf("GLFW ded\n");
		return 0;
	}
	GLFWwindow* window;
	WindowInit(window);
	Main _Main;
	_Main.shader.Use();
	glfwSetKeyCallback(window, KeyCallback);
	glfwSetCharCallback(window, CharacterCallback);
	glfwSetMouseButtonCallback(window, MouseButtonCallback);
	glfwSetFramebufferSizeCallback(window, FrameSizeCallback);
	glfwSetCursorPosCallback(window, CursorPositionCallback);

	Line line(&_Main.camera, Mat2x2(Vec2(), windowSize), Color(0, 1, 0, 1));
	line.push_back(Mat2x2(Vec2(windowSize.x, 0), Vec2(0, windowSize.y)));
	line.set_position(1, Mat2x2(Vec2(windowSize.x / 2, 0), Vec2(windowSize.x / 2, windowSize.y)));
	line.set_color(1, Color(0, 0, 1, 1));
	Mat2x2 x = line.get_position(1);

	Triangle triangle(&_Main.camera, Vec2(windowSize / 2), Vec2(100), Color(1, 1, 1, 1));

	triangle.push_back(Vec2(), Vec2(500));
	triangle.set_position(1, Vec2(windowSize));
	
	Square square(&_Main.camera, Vec2(windowSize / 2), Vec2(100), Color(1, 0, 0, 1));

	square.push_back(Vec2(100));
	square.set_position(0, Vec2(0));

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		line.draw();
		triangle.draw();
		square.draw();
		square.set_position(0, cursorPos);

		if (KeyPress(GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, true);
		}

		GetFps();
		glfwSwapBuffers(window);
		KeysReset();
	}
}