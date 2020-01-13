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

	for (int i = 0; i < view.x; i++) {
		Color newColor;
		newColor.r = (unsigned int)grid.get_color(i).r ^ (unsigned int)1;
		newColor.a = 1;
		grid.set_color(i, newColor);
		walls[i] = true;
	}

	for (int i = view.x * view.y - view.x; i < view.x * view.y; i++) {
		Color newColor;
		newColor.r = (unsigned int)grid.get_color(i).r ^ (unsigned int)1;
		newColor.a = 1;
		grid.set_color(i, newColor);
		walls[i] = true;
	}

	for (int i = 0; i < view.y; i++) {
		Color newColor;
		newColor.r = (unsigned int)grid.get_color(i * 14).r ^ (unsigned int)1;
		newColor.a = 1;
		grid.set_color(i * 14, newColor);
		walls[i * 14] = true;
	}

	for (int i = view.x - 1; i < view.y * 14; i+=14) {
		Color newColor;
		newColor.r = (unsigned int)grid.get_color(i).r ^ (unsigned int)1;
		newColor.a = 1;
		grid.set_color(i, newColor);
		walls[i] = true;
	}

	Square collidable(&_Main.camera, Vec2(), Vec2(blockSize), Color(.415, 0.05, .67, 1));

	Square square(&_Main.camera, Cast<float>(windowSize) / 2, Vec2(blockSize), Color(0, 1, 0, 1));

	Square dot(&_Main.camera, Vec2(), Vec2(16), Color(0, 0, 1, 1));

	collidable.set_position(0, windowSize / 2);
	//line.set_position(0, Mat2x2(Vec2(blockSize, windowSize.y / 3), Vec2(blockSize + lineLength, windowSize.y / 3)));

	const uint8_t movement[] = { GLFW_KEY_W, GLFW_KEY_A, GLFW_KEY_S, GLFW_KEY_D };

	

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

		Alloc<Vec2> collision;
		collision.resize(12);


		//a
		collision[0] = Raycast(grid, Mat2x2(
			Vec2(square.get_position(0).x + 1, square.get_position(0).y),
			Vec2(square.get_position(0).x - movement_speed * deltaTime, square.get_position(0).y)),
			walls, view.x * view.y, false);
		collision[1] = Raycast(grid, Mat2x2(
			Vec2(square.get_position(0).x + 1, square.get_position(0).y + square.get_size(0).y / 2),
			Vec2(square.get_position(0).x - movement_speed * deltaTime, square.get_position(0).y + square.get_size(0).y / 2)),
			walls, view.x * view.y, false);
		collision[2] = Raycast(grid, Mat2x2(
			Vec2(square.get_position(0).x + 1, square.get_position(0).y + square.get_size(0).y),
			Vec2(square.get_position(0).x - movement_speed * deltaTime, square.get_position(0).y + square.get_size(0).y)),
			walls, view.x * view.y, false);

		//d
		collision[3] = Raycast(grid, Mat2x2(
			Vec2(square.get_position(0).x + square.get_size(0).x - 1, square.get_position(0).y),
			Vec2(square.get_position(0).x + square.get_size(0).x + movement_speed * deltaTime, square.get_position(0).y)),
			walls, view.x * view.y, true);
		collision[4] = Raycast(grid, Mat2x2(
			Vec2(square.get_position(0).x + square.get_size(0).x - 1, square.get_position(0).y + square.get_size(0).y / 2),
			Vec2(square.get_position(0).x + square.get_size(0).x + movement_speed * deltaTime, square.get_position(0).y + square.get_size(0).y / 2)),
			walls, view.x * view.y, true);
		collision[5] = Raycast(grid, Mat2x2(
			Vec2(square.get_position(0).x + square.get_size(0).x - 1, square.get_position(0).y + square.get_size(0).y),
			Vec2(square.get_position(0).x + square.get_size(0).x + movement_speed * deltaTime, square.get_position(0).y + square.get_size(0).y)),
			walls, view.x * view.y, true);

		//s
		collision[6] = Raycast(grid, Mat2x2(
			Vec2(square.get_position(0).x, square.get_position(0).y + square.get_size(0).y - 1),
			Vec2(square.get_position(0).x, square.get_position(0).y + square.get_size(0).y + movement_speed * deltaTime)),
			walls, view.x * view.y, true);
		collision[7] = Raycast(grid, Mat2x2(
			Vec2(square.get_position(0).x + square.get_size(0).x / 2, square.get_position(0).y + square.get_size(0).y - 1),
			Vec2(square.get_position(0).x + square.get_size(0).x / 2, square.get_position(0).y + square.get_size(0).y + movement_speed * deltaTime)),
			walls, view.x * view.y, true);
		collision[8] = Raycast(grid, Mat2x2(
			Vec2(square.get_position(0).x + square.get_size(0).x, square.get_position(0).y + square.get_size(0).y - 1),
			Vec2(square.get_position(0).x + square.get_size(0).x, square.get_position(0).y + square.get_size(0).y + movement_speed * deltaTime)),
			walls, view.x * view.y, true);

		//w
		collision[9] = Raycast(grid, Mat2x2(
			Vec2(square.get_position(0).x, square.get_position(0).y + 1),
			Vec2(square.get_position(0).x, square.get_position(0).y - movement_speed * deltaTime)),
			walls, view.x * view.y, true);
		collision[10] = Raycast(grid, Mat2x2(
			Vec2(square.get_position(0).x + square.get_size(0).x / 2, square.get_position(0).y + 1),
			Vec2(square.get_position(0).x + square.get_size(0).x / 2, square.get_position(0).y - movement_speed * deltaTime)),
			walls, view.x * view.y, true);
		collision[11] = Raycast(grid, Mat2x2(
			Vec2(square.get_position(0).x + square.get_size(0).x, square.get_position(0).y + 1),
			Vec2(square.get_position(0).x + square.get_size(0).x, square.get_position(0).y - movement_speed * deltaTime)),
			walls, view.x * view.y, true);
		
		if (KeyPress(GLFW_MOUSE_BUTTON_LEFT)) {
			square.set_position(0, Cast<float>(cursorPos) - square.get_size(0) / 2);
		}

		if (KeyPress(GLFW_KEY_A)) {
			int index = -1;
			float smallest = -INFINITY;
			for (int i = 0; i < 3; i++) {
				if (collision[i].x == -1) {
					continue;
				}
				if (collision[i].x > smallest) {
					smallest = collision[i].x;
					index = i;
				}
			}
			if (index != -1) {
				square.set_position(0, Vec2(collision[index].x, square.get_position(0).y));
			}
			else {
				square.set_position(0, Vec2(square.get_position(0).x - movement_speed * deltaTime, square.get_position(0).y));
			}
		}
		if (KeyPress(GLFW_KEY_D)) {
			int index = -1;
			float smallest = INFINITY;
			for (int i = 3; i < 6; i++) {
				if (collision[i].x == -1) {
					continue;
				}
				if (collision[i].x < smallest) {
					smallest = collision[i].x;
					index = i;
				}
			}
			if (index != -1) {
				square.set_position(0, Vec2(collision[index].x - square.get_size(0).x, square.get_position(0).y));
			}
			else {
				square.set_position(0, Vec2(square.get_position(0).x + movement_speed * deltaTime, square.get_position(0).y));
			}
		}
		if (KeyPress(GLFW_KEY_S)) {
			int index = -1;
			float smallest = INFINITY;
			for (int i = 6; i < 9; i++) {
				if (collision[i].x == -1) {
					continue;
				}
				if (collision[i].y < smallest) {
					smallest = collision[i].y;
					index = i;
				}
			}
			if (index != -1) {
				square.set_position(0, Vec2(square.get_position(0).x, collision[index].y - square.get_size(0).y));
			}
			else {
				square.set_position(0, Vec2(square.get_position(0).x, square.get_position(0).y + movement_speed * deltaTime));
			}
		}
		if (KeyPress(GLFW_KEY_W)) {
			int index = -1;
			float smallest = -INFINITY;
			for (int i = 9; i < 12; i++) {
				if (collision[i].y == -1) {
					continue;
				}
				if (collision[i].y > smallest) {
					smallest = collision[i].y;
					index = i;
				}
			}
			if (index != -1) {
				square.set_position(0, Vec2(square.get_position(0).x, collision[index].y));
			}
			else {
				square.set_position(0, Vec2(square.get_position(0).x, square.get_position(0).y - movement_speed * deltaTime));
			}
		}

		//line.draw();
		square.draw();
		grid.draw();
		if (KeyPress(GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, true);
		}

		GetFps();
		glfwSwapBuffers(window);
		KeysReset();
	}
}