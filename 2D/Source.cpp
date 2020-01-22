#include <iostream>
#include <FAN/Graphics.hpp>
#include <FAN/DBT.hpp>
#include <vector>

size_t _2D1D() {
	return (int(cursorPos.x / blockSize)) + int(cursorPos.y / blockSize) * (windowSize.y / blockSize);
}

std::vector<size_t> ind;

#define ray_amount 100

template <typename _Square, typename _Matrix, typename _Alloc, typename _ReturnType = Vec2>
constexpr _ReturnType Raycast(const _Square& grid, const _Matrix& direction, const _Alloc& walls, size_t gridSize, bool right) {
	for (auto&& i : ind) {
		if (direction[1].x >= direction[0].x) {
			Vec2 inter = IntersectionPoint(direction[0], direction[1], grid.get_corners(i)[0], grid.get_corners(i)[3]);
			if (inter.x != -1) {
				return inter;
			}
		}
		else {
			Vec2 inter = IntersectionPoint(direction[0], direction[1], grid.get_corners(i)[1], grid.get_corners(i)[2]);
			if (inter.x != -1) {
				return inter;
			}
		}
		if (direction[1].y <= direction[0].y) {
			Vec2 inter = IntersectionPoint(direction[0], direction[1], grid.get_corners(i)[2], grid.get_corners(i)[3]);
			if (inter.x != -1) {
				return inter;
			}
		}
		else {
			Vec2 inter = IntersectionPoint(direction[0], direction[1], grid.get_corners(i)[0], grid.get_corners(i)[1]);
			if (inter.x != -1) {
				return inter;
			}
		}
	}
	return Vec2(-1, -1);
}

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
	Square grid;

	Alloc<bool> walls;

	for (int y = 0; y < view.y; y++) {
		for (int x = 0; x < view.x; x++) {
			grid.push_back(Vec2(x * blockSize, y * blockSize), Vec2(blockSize), Color(0, 0, 0, 1));
		}
	}

	for (int y = 0; y < view.x * view.y; y++) {
		walls.push_back(false);
	}

	Line line(Mat2x2(), Color(1, 0, 1, 1));

	for (int i = 0; i < ray_amount; i++)
	line.push_back(Mat2x2(),Color(-1, -1, -1, -1), true);
	line.break_queue();
	Alloc<Vec2> inter(ray_amount);

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		for (int i = 0; i < ray_amount; i++) {
			float theta = 2.0f * 3.1415926f * float(i) / float(line_amount);
			Vec2 direction(line.get_position(i)[1] + Vec2(sin(theta) * 1000, cos(theta) * 1000));
			inter[i] = Raycast(grid, Mat2x2(line.get_position(0)[0], direction), walls, view.x * view.y, true);
			line.set_position_queue(i, Mat2x2(line.get_position(i)[0], inter[i].x != -1 ? inter[i] : direction));
		}
		line.break_queue();

		line.draw();
		grid.draw();

		if (KeyPressA(GLFW_MOUSE_BUTTON_RIGHT)) {
			Color newColor;
			newColor.r = (unsigned int)grid.get_color(_2D1D()).r ^ (unsigned int)1;
			newColor.a = 1;
			auto it = std::find(ind.begin(), ind.end(), _2D1D());
			walls[_2D1D()] = !walls[_2D1D()];
			grid.set_color(_2D1D(), newColor);
			if (it != ind.end()) {
				ind.erase(it);
			}
			else {
				ind.push_back(_2D1D());
			}
		}

		if (KeyPress(GLFW_KEY_SPACE)) {
			for (int i = 0; i < ray_amount; i++) {
				line.set_position_queue(i, Mat2x2(cursorPos, cursorPos));
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