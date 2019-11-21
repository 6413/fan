#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <ctime>
#include "FAN/Bmp.hpp"

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
	glEnable(GL_DEPTH_TEST);
	glfwSetKeyCallback(window, KeyCallback);
	glfwSetCharCallback(window, CharacterCallback);
	glfwSetMouseButtonCallback(window, MouseButtonCallback);
	glfwSetFramebufferSizeCallback(window, FrameSizeCallback);
	glfwSetCursorPosCallback(window, CursorPositionCallback);

	Square grass(&_Main.camera, Vec2(windowSize.x / 2, windowSize.y - GRASSHEIGHT / 2), Vec2(windowSize.x, GRASSHEIGHT), Color(0x7c, 0xfc, 0, 255));

	Sprite player_left(&_Main.camera, "Pictures/guy_left.bmp");

	Entity player(&_Main.camera, "Pictures/guy_right.bmp", PLAYERSIZE, Vec2(windowSize.x / 2, windowSize.y - GRASSHEIGHT));

	player.SetImage(player_left);

	while (!glfwWindowShouldClose(window)) {

		glfwPollEvents();
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		glClearColor(0.0, 0.0, 0.1, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		player.Move();
		player.Draw();

		grass.Draw();

		grass.SetPosition(0, Vec2(windowSize.x / 2, windowSize.y - GRASSHEIGHT / 2));

		if (KeyPress(GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, true);
		}

		GetFps();

		glfwSwapBuffers(window);
	}

	glfwTerminate();
	return 0;
}