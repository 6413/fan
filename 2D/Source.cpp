#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <ctime>

#include "FAN/Bmp.hpp"
#include "FAN/Functions.hpp"
#include "FAN/Texture.hpp"


float LINELENGTH = 500;

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

	//Line grid(&_Main.camera, Mat2x2(Vec2(0, 0), Vec2(0, 0)), Color(1, 0, 0, 1));

	//Square gridSquares(&_Main.camera, Vec2(blockSize, blockSize), Color(0, 0, 0, 1));

	//Square light(&_Main.camera, Vec2(windowSize / 2), Vec2(16, 16), Color(1, 0, 1, 1));

	//Line rays(&_Main.camera, Mat2x2(light.GetPosition(0), light.GetPosition(0) + Vec2(100, 0)), Color(0, 1, 1, 1));

	//for (int rows = 0; rows < windowSize.x; rows += blockSize) {
	//	grid.Add(Mat2x2(Vec2(0, rows), Vec2(windowSize.x, rows)));
	//}
	//for (int rows = 0; rows < windowSize.x; rows += blockSize) {
	//	grid.Add(Mat2x2(Vec2(rows, 0), Vec2(rows, windowSize.y)));
	//}
	//for (int colums = 0; colums < windowSize.y; colums += blockSize) {
	//	for (int rows = 0; rows < windowSize.x; rows += blockSize) {
	//		gridSquares.Add(Vec2(blockSize / 2 + rows, blockSize / 2 + colums), Vec2(blockSize));
	//	}
	//}

	bool wall[196] = {};
	//rays.Add(Mat2x2(Vec2(), Vec2(800, 800)), Color(0, 0, 1, 1));

	//for (int iray = 0; iray < 1024; iray++) {
	//	float progress = (float)iray / (float)1024;
	//	Vec2 HDG = Vec2(sin(progress), cos(progress));
	//	Vec2 Start = light.GetPosition(0);
	//	rays.Add(Mat2x2(Start, HDG * 9999), Color(0, 0, 1, 1));
	//}

	Line line(&_Main.camera, Mat2x2(Vec2(), windowSize), Color(1, 0, 0, 1));

	//line.push_back(Mat2x2(Vec2(windowSize.x, 0), Vec2(0, windowSize.y)), Color(0, 1, 0, 1));

	//line.setPosition(1, Mat2x2(Vec2(0, 400), Vec2(800, 400)));

	//Allocate<float> vec(1);

	//vec[0] = 1;

	//vec.push_back(5);

	//std::vector<float> vec;

	//vec.push_back(23);

	//float& x = vec[0];

	//printf("%p\n", &x);

	//vec.push_back(32);

	//printf("%p\n", &x);

	line.push_back(Mat2x2(Vec2(0, windowSize.y / 2), Vec2(windowSize.x, windowSize.y / 2)), Color(1, 0, 0, 1));

	//line.setPosition(0, Mat2x2(Vec2(windowSize.x, 0), Vec2(0, windowSize.y)));

	//Alloc<VerCol> test;
	//Alloc<float> somethign;
	//somethign.push_back(5);

	//for (int i = 0; i < 100; i++) {
	//	test.push_back(VerCol(somethign, Color(1, 0, 0, 1)));
	//}

	//for (int i = 0; i < test.size(); i++) {
	//	printf("%f\n", test[i].colors[0]);
	//}

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		Vec2 view = windowSize / blockSize;

		bool colliding = false;

		line.draw();
		//for (int i = 0; i < LINELENGTH; i++) {
		//	Vec2 lightPos = light.GetPosition(0);
		//	if (wall[int(int((lightPos.x + i) / blockSize) + int(lightPos.y / blockSize) * view.y)]) {
		//		break;
		//	}
		//	rays.SetPosition(0, Mat2x2(lightPos, lightPos + Vec2(i, 0)));
		//}

		//for (int i = 0; i < 196; i++) {
		//	if (!wall[i]) {
		//		continue;
		//	}
		//	printf("%d\n", i);
		//	if (Trace(light.GetPosition(0), 0, Vec2((i % 14) * view.x - 16, (i / 14) * view.y - 16), Vec2((i % 14) * view.x - 16, (i / 14) * view.y + 16))) {
		//		printf("colliding\n");
		//	}
		//}


		//float distance = LINELENGTH / view.x;
		//for (float i = 0; distance >= 0 ? i < distance : i > distance; i += distance / accuracy) {
		//	if (wall[int(int((cursorPos.x + i * view.x) / blockSize) + int(cursorPos.y / blockSize) * view.y) % 196]) {
		//		if (KeyPress(GLFW_MOUSE_BUTTON_LEFT)) {
		//			rays.SetPosition(0, Mat2x2(light.GetPosition(0), Vec2(cursorPos.x + i * view.y, cursorPos.y)));
		//			break;
		//		}
		//	}
		//}

		if (KeyPressA(GLFW_MOUSE_BUTTON_MIDDLE)) {
			LINELENGTH = -LINELENGTH;
		}

		if (KeyPress(GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, true);
		}

		GetFps();
		glfwSwapBuffers(window);
		KeysReset();
	}

	glfwTerminate();
	return 0;
}