#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <ctime>
#include "FAN/Bmp.hpp"

#include "FAN/Texture.hpp"

#define CORRECTPOSITION(x) (floor(x / BLOCKSIZE) * BLOCKSIZE)

Collision collision;

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

	Line line(&_Main.camera, Mat2x2(Vec2(0, 0), Vec2(0, 0)), Color(1, 0, 0, 1));

	Square square(&_Main.camera, Vec2(blockSize, blockSize), Color(0, 0, 0, 1));

	for (int rows = 0; rows < windowSize.x; rows += blockSize) {
		line.Add(Mat2x2(Vec2(0, rows), Vec2(windowSize.x, rows)));
	}
	for (int rows = 0; rows < windowSize.x; rows += blockSize) {
		line.Add(Mat2x2(Vec2(rows, 0), Vec2(rows, windowSize.y)));
	}
	for (int colums = 0; colums < windowSize.y; colums += blockSize) {
		for (int rows = 0; rows < windowSize.x; rows += blockSize) {
			square.Add(Vec2(blockSize / 2 + rows, blockSize / 2 + colums), Vec2(blockSize));
		}
	}

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		 
		glClearColor(0, 0, 0, 1); 
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		line.Draw();
	
		square.Draw();

		if (KeyPressA(GLFW_MOUSE_BUTTON_LEFT)) {
			square.SetColor(floor(cursorPos.x / blockSize) + floor(cursorPos.y / blockSize) * (windowSize.y / blockSize), '^', 1);
		}

		if (KeyPress(GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, true);
		}

		GetFps();
		glfwSwapBuffers(window);
	}

	glfwTerminate();
	return 0;
}