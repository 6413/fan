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

	Line line(&_Main.camera, Mat2x2(Vec2(0, 0), Vec2(400, 400)), Color(1, 0, 0, 1));

	//for (int columns = 0; columns < windowSize.y; columns += blockSize) {
	//	for (int rows = 0; rows < windowSize.x; rows += blockSize) {
	//		square.Add(Vec2(blockSize / 2 + rows, blockSize / 2 + columns), Vec2(blockSize));
	//	}
	//}

	line.Add(Mat2x2(Vec2(0, 800), Vec2(400, 400)), Color(0, 1, 0, 0.5));


	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		 
		glClearColor(0, 0, 0, 1); 
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		line.Draw();

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