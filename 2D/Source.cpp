#include <iostream>
#include <FAN/Graphics.hpp>
#include <FAN/DBT.hpp>

size_t _2D1D() {
	return (int(cursorPos.x / blockSize)) + int(cursorPos.y / blockSize) * (windowSize.y / blockSize);
}

size_t _2D1D(Vec2 pos) {
	return (int(floor(pos.x / blockSize))) + int(floor(pos.y / blockSize)) * (windowSize.y / blockSize);
}

#define movement_speed 10000

#define lineLength 300 

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
	glfwSetScrollCallback(window, ScrollCallback);


	__Vec2<int> view(windowSize.x / blockSize, windowSize.y / blockSize);
	Square grid(&_Main.camera);

	Alloc<bool> walls;

	for (int y = 0; y < view.y; y++) {
		for (int x = 0; x < view.x; x++) {
			grid.push_back(Vec2(x * blockSize, y * blockSize), Vec2(blockSize), Color(0, 0, 0, 1));
		}
	}

	for (int y = 0; y < view.x * view.y; y++) {
		walls.push_back(false);
	}

	while (!glfwWindowShouldClose(window)) {  
		glfwPollEvents();
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (KeyPressA(GLFW_MOUSE_BUTTON_RIGHT)) {
			Color newColor;
			newColor.r = (unsigned int)grid.get_color(_2D1D()).r ^ (unsigned int)1;
			newColor.a = 1;
			walls[_2D1D()] = !walls[_2D1D()];
			grid.set_color(_2D1D(), newColor);
		}

		grid.draw();
		if (KeyPress(GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, true);
		}

		GetFps();
		glfwSwapBuffers(window);
		KeysReset();
	}
}