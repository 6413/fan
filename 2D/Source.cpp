#include <iostream>
#include <FAN/Graphics.hpp>
#include <FAN/DBT.hpp>
#include <vector>

size_t _2D1D() {
	return (int(cursorPos.x / blockSize)) + int(cursorPos.y / blockSize) * (windowSize.y / blockSize);
}

constexpr auto ray_amount = 100;

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

	__Vec2<int> view(windowSize.x / blockSize, windowSize.y / blockSize);
	Square squares;

	for (int y = 0; y < view.y; y++) {
		for (int x = 0; x < view.x; x++) {
			squares.push_back(Vec2(x * blockSize, y * blockSize), Vec2(blockSize), Color(0, 0, 0, 1), true);
		}
	}
	squares.break_queue();

	Line line(Mat2x2(), Color(1, 1, 1, 1));

	for (int i = 0; i < ray_amount; i++) {
		line.push_back(Mat2x2(), Color(1, 1, 1, 1), true);
	}

	line.break_queue();
	Alloc<Vec2> inter(ray_amount);
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		for (int i = 0; i < ray_amount; i++) {
			Mat2x2 linePos = line.get_position(i);
			float theta = 2.0f * 3.1415926f * float(i) / float(ray_amount);
			Vec2 direction(linePos[1] + Vec2(sin(theta) * 1000, cos(theta) * 1000));
			inter[i] = Raycast(squares, Mat2x2(linePos[0], direction), view.x * view.y);

			if (inter[i].x != -1) {
				line.set_position(i, Mat2x2(linePos[0], inter[i]), true);

			}
			else {
				line.set_position(i, Mat2x2(linePos[0], direction), true);
			}
		}
		line.break_queue();

		//grid.draw();
		line.draw();
		squares.draw();


		if (KeyPressA(GLFW_MOUSE_BUTTON_RIGHT)) {
			Color newColor;
			newColor.r = (unsigned int)squares.get_color(_2D1D()).r ^ (unsigned int)1;
			newColor.a = 1;
			squares.set_color(_2D1D(), newColor);
			auto it = std::find(ind.begin(), ind.end(), _2D1D());

			if (it != ind.end()) {
				ind.erase(it);
			}
			else {
				ind.push_back(_2D1D());
			}
		}

		if (KeyPress(GLFW_KEY_SPACE)) {
			for (int i = 0; i < ray_amount; i++) {
				line.set_position(i, Mat2x2(cursorPos, cursorPos), true);
			}
			line.break_queue();
		}

		if (KeyPress(GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, true);
		}

		GetFps();
		glfwSwapBuffers(window);
		KeysReset();
	}
}