#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <ctime>

#include "FAN/Bmp.hpp"
#include "FAN/Functions.hpp"
#include "FAN/Texture.hpp"

template<typename T1, typename T2>
constexpr auto LDF_HDG2(T1 M_src, T2  M_dst) {
	return (M_src - M_dst) / (abs((M_src - M_dst).x) > abs((M_src - M_dst).y) ? abs((M_src - M_dst).x) : abs((M_src - M_dst).y));
}

typedef struct {
	Vec2 HDG, src, dst;
	__Vec2<int> grid;
}LT_gridRaycast;

size_t F_WITCH_Math_gridRaycast(LT_gridRaycast& ret, int GridSize) {
	Vec2 pos(fmodf(ret.src.x, GridSize), fmodf(ret.src.y, GridSize));
	Vec2 closest = (Vec2(pos - GridSize / 2).Abs() + -GridSize / 2).Abs();
	pos.x = ((ret.HDG.x < 0) ? pos.x : GridSize - pos.x); if (closest.x == 0) pos.x = GridSize;
	pos.y = ((ret.HDG.y < 0) ? pos.y : GridSize - pos.y); if (closest.y == 0) pos.y = GridSize;

	pos = (pos - ret.HDG).Abs();
	ret.src = ret.HDG * (pos.x < pos.y ? pos.x : pos.y);
	ret.grid = ret.src / GridSize;
	//ret.grid.x -= ret
	return 0;
}

#define LDF_gridRaycast(M_oldPos, M_newPos, M_varName, M_gridSize) \
	LT_gridRaycast M_varName = LDC_WITCH(LT_gridRaycast){LDF_HDG2(M_newPos, M_oldPos), M_oldPos, M_newPos, DC_2si(0, 0)}; \
	if(!OP_2fE2f(M_varName.src, M_varName.dst)) \
	while(F_WITCH_Math_gridRaycast(&M_varName, M_gridSize))

int main() {
	glfwSetErrorCallback(GlfwErrorCallback);
	if (!glfwInit()) {
		printf("GLFW ded\n");
		return 0;
	}
	GLFWwindow* window;
	WindowInit(window);
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

	__Vec2<int> view = windowSize / blockSize;

	for (int rows = 0; rows < windowSize.x; rows += blockSize) {
		grid.push_back(Mat2x2(Vec2(0, rows), Vec2(windowSize.x, rows)), Color(1, 0, 0, 1));
	}
	for (int rows = 0; rows < windowSize.x; rows += blockSize) {
		grid.push_back(Mat2x2(Vec2(rows, 0), Vec2(rows, windowSize.y)), Color(1, 0, 0, 1));
	}
	for (int square = 0; square < view.x * view.y; square++) {
		gridSquares.push_back(Vec2(square % view.x * blockSize + blockSize / 2, square / view.y * blockSize + blockSize / 2), Color(0, 0, 0, 1));
	}

	Square light(&_Main.camera, windowSize / 2, Vec2(blockSize / 4), Color(1, 1, 1, 1));

	bool showGrid = true;

	Line line(&_Main.camera, Mat2x2(), Color(0, 1, 0, 1));

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		view = windowSize / blockSize;

		if (KeyPressA(GLFW_MOUSE_BUTTON_RIGHT)) {
			gridSquares.GetColor(0);
			const int at = (int(cursorPos.x / blockSize)) + int(cursorPos.y / blockSize) * (windowSize.y / blockSize);
			gridSquares.SetColor(at, '^', __Vec3<int>(0, 0, 1));
			//gridSquares.wall[at] ^= 1;
		}

		if (KeyPress(GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, true);
		}
		
		if (KeyPressA(GLFW_KEY_G)) {
			showGrid = !showGrid;
		}

		//light.draw();sec
		
		line.setPosition(0, Mat2x2(light.getPosition(0), Cast<float>(cursorPos)));
		//line.getPositionMat(0).vec[0];
		//line.getPositionMat(0).vec[1];
		line.draw();
		//Vec2 HDG = LDF_HDG2(line.getPositionMat(0).vec[1], line.getPositionMat(0).vec[0]);
		//for (int i = 0; i < 6; i++) {
		//	Vec2 NextDot = line.getPositionMat(0).vec[0] + HDG * Vec2(blockSize * i);
		//	point.setPosition(i, NextDot);
		//}
		//NextDot = Round(NextDot, blockSize);
		//point.draw();

		if (showGrid) {
			grid.draw();
		}

		gridSquares.draw();

		GetFps();
		glfwSwapBuffers(window);
		KeysReset();
	}
	glfwTerminate();
	return 0;
}