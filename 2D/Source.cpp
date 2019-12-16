#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <ctime>
#include <Windows.h>
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
	glEnable(GL_DEPTH_TEST);
	glfwSetKeyCallback(window, KeyCallback);
	glfwSetCharCallback(window, CharacterCallback);
	glfwSetMouseButtonCallback(window, MouseButtonCallback);
	glfwSetFramebufferSizeCallback(window, FrameSizeCallback);
	glfwSetCursorPosCallback(window, CursorPositionCallback);

	const float blockSize = 64;

	Square square(&_Main.camera, Vec2(16), Vec2(blockSize), Color(0, 0, 0, 255));

	for (int columns = 0; columns < windowSize.y; columns += blockSize) {
		for (int rows = 0; rows < windowSize.x; rows += blockSize) {
			square.Add(Vec2(blockSize / 2 + rows, blockSize / 2 + columns), Vec2(blockSize));
		}
	}

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		 
		glClearColor(1, 1, 1, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		square.Draw();
		//square.SetColor(Color(255, 0, 0, 255));

		if (KeyPress(GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, true);
		}

		GetFps();
		glfwSwapBuffers(window);
	}

	glfwTerminate();
	return 0;
}