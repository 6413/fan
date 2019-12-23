#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <ctime>

#include "FAN/Bmp.hpp"
#include "FAN/Functions.hpp"
#include "FAN/Texture.hpp"

int main() {
	glfwSetErrorCallback(GlfwErrorCallback);
	if (!glfwInit()) {
		printf("GLFW ded\n");
		return 0;
	}
	GLFWwindow* window;
	WindowInit(window);
	float lastFrame = 0;
	srand(time(NULL));
	Main _Main;
	_Main.shader.Use();
	glfwSetKeyCallback(window, KeyCallback);
	glfwSetCharCallback(window, CharacterCallback);
	glfwSetMouseButtonCallback(window, MouseButtonCallback);
	glfwSetFramebufferSizeCallback(window, FrameSizeCallback);
	glfwSetCursorPosCallback(window, CursorPositionCallback);

	const float blockSize = 64;

	Line grid(&_Main.camera, Mat2x2(Vec2(), Vec2()), Color(1, 0, 0, 1));

	Square gridSquares(&_Main.camera, Vec2(blockSize), Color(0, 0, 0, 1));

	for (int rows = 0; rows < windowSize.x; rows += blockSize) {
		grid.push_back(Mat2x2(Vec2(0, rows), Vec2(windowSize.x, rows)), Color(1, 0, 0, 1));
	}
	for (int rows = 0; rows < windowSize.x; rows += blockSize) {
		grid.push_back(Mat2x2(Vec2(rows, 0), Vec2(rows, windowSize.y)), Color(1, 0, 0, 1));
	}

	std::vector<Vec2> coordinates = LoadMap(blockSize);

	for (auto i : coordinates) {
		gridSquares.push_back(i, Color(1, 1, 1, 1));
	}

	bool showGrid = true;
	 
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		Vec2 view = windowSize / blockSize;

		bool colliding = false;

		if (showGrid) {
			grid.draw();
		}

		if (KeyPressA(GLFW_MOUSE_BUTTON_RIGHT)) {
			gridSquares.GetColor(0);
			const size_t at = (int(cursorPos.x / blockSize)) + int(cursorPos.y / blockSize) * (windowSize.y / blockSize);
			gridSquares.SetColor(at, '^', 1);
			gridSquares.wall[at] ^= 1;
		}

		if (KeyPress(GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, true);
		}
		
		if (KeyPressA(GLFW_KEY_G)) {
			showGrid = !showGrid;
		}

		gridSquares.draw();

		GetFps();
		glfwSwapBuffers(window);
		KeysReset();
	}
	//gridSquares.SaveMap(blockSize);
	glfwTerminate();
	return 0;
}