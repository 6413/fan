#include <iostream>
#include <FAN/Graphics.hpp>
#include <FAN/DBT.hpp>
#include <vector>

size_t _2D1D() {
	return (int(cursorPos.x / blockSize)) + int(cursorPos.y / blockSize) * (windowSize.y / blockSize);
}

constexpr auto movement_speed = 400;

bool walls[view.x][view.y];

size_t find_closest(float start, float end, bool x, int x_or_y, bool direction) {
	if (x) {
		if (direction) {
			for (int i = start; i < end; i++) {
				if (walls[i][x_or_y]) {
					return i;
				}
			}
		}
		else {
			for (int i = start; i--; ) {
				if (walls[i][x_or_y]) {
					return i;
				}
			}
		}
	}
	else {
		if (direction) {
			for (int i = start; i < end; i++) {
				if (walls[x_or_y][i]) {
					return i;
				}
			}
		}
		else {
			for (int i = start; i--; ) {
				if (walls[x_or_y][i]) {
					return i;
				}
			}
		}
	}
	return -1;
}

#include <ctime>

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

	Square squares;

	for (int y = 0; y < view.y; y++) {
		for (int x = 0; x < view.x; x++) {
			squares.push_back(Vec2(x * blockSize, y * blockSize), Vec2(blockSize), Color(0, 0, 0, 1), true);
		}
	}
	squares.break_queue();


	Line line;
	for (int i = 1; i < view.x + 1; i++) {
		line.push_back(Mat2x2(Vec2(0, i * blockSize), Vec2(windowSize.x, i * blockSize)), Color(1, 0, 0), true);
	}
	for (int i = 1; i < view.y + 1; i++) {
		line.push_back(Mat2x2(Vec2(i * blockSize, 0), Vec2(i * blockSize, windowSize.y)), Color(1, 0, 0), true);
	}
	line.break_queue();

	Square player(Vec2(blockSize * 6, blockSize * 6), Vec2(blockSize), Color(1, 1, 1));

	for (int i = 0; i < view.x; i++) {
		for (int j = 0; j < view.y; j++) {
			walls[i][j] = false;
		}
	}

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		Vec2 player_pos(floor(player.get_position(0).x / blockSize), floor(player.get_position(0).y / blockSize));

		if (KeyPressA(GLFW_MOUSE_BUTTON_RIGHT)) {
			Color newColor;
			newColor.r = (unsigned int)squares.get_color(_2D1D()).r ^ (unsigned int)1;
			newColor.a = 1;
			squares.set_color(_2D1D(), newColor);
			walls[_2D1D() % view.x][_2D1D() / view.y] = !walls[_2D1D() % view.x][_2D1D() / view.y];
		}

		if (KeyPress(GLFW_KEY_A)) {
			int point = find_closest(player_pos.x, view.x, true, player_pos.y, false);
			int point2 = find_closest(player_pos.x, view.x, true, (player.get_position(0).y + blockSize - 1) / blockSize, false);
			if (int((player.get_position(0).x / blockSize) - (movement_speed / blockSize) * deltaTime) <= point && point != -1) {
				player.set_position(0, Vec2((point + 1) * blockSize, player.get_position(0).y));
			}
			else if (int((player.get_position(0).x / blockSize) - (movement_speed / blockSize) * deltaTime) <= point2 && point2 != -1) {
				player.set_position(0, Vec2((point2 + 1) * blockSize, player.get_position(0).y));
			}
			else {
				player.set_position(0, Vec2(player.get_position(0).x - (movement_speed * deltaTime), player.get_position(0).y));
			}
		}
		if (KeyPress(GLFW_KEY_D)) {
			int point = find_closest(player_pos.x, view.x, true, player_pos.y, true);
			int point2 = find_closest(player_pos.x, view.x, true, (player.get_position(0).y + blockSize - 1) / blockSize, true);
			if (int((player.get_position(0).x / blockSize + 1) + (movement_speed / blockSize) * deltaTime) >= point && point != -1) {
				player.set_position(0, Vec2((point - 1) * blockSize, player.get_position(0).y));
			}
			else if (int((player.get_position(0).x / blockSize + 1) + (movement_speed / blockSize) * deltaTime) >= point2 && point2 != -1) {
				player.set_position(0, Vec2((point2 - 1) * blockSize, player.get_position(0).y));
			}
			else {
				player.set_position(0, Vec2(player.get_position(0).x + (movement_speed * deltaTime), player.get_position(0).y));
			}
		}
		if (KeyPress(GLFW_KEY_W)) {
			int point = find_closest(player_pos.y, view.y, false, player_pos.x, false);
			int point2 = find_closest(player_pos.y, view.y, false, (player.get_position(0).x + blockSize - 1) / blockSize, false);
			if (int((player.get_position(0).y / blockSize) - (movement_speed / blockSize) * deltaTime) <= point && point != -1) {
				player.set_position(0, Vec2(player.get_position(0).x, (point + 1) * blockSize));
			}
			else if (int((player.get_position(0).y / blockSize) - (movement_speed / blockSize) * deltaTime) <= point2 && point2 != -1) {
				player.set_position(0, Vec2(player.get_position(0).x, (point2 + 1) * blockSize));
			}
			else {
				player.set_position(0, Vec2(player.get_position(0).x, player.get_position(0).y - (movement_speed * deltaTime)));
			}
		}
		if (KeyPress(GLFW_KEY_S)) {
			int point = find_closest(player_pos.y, view.y, false, player_pos.x, true);
			int point2 = find_closest(player_pos.y, view.y, false, (player.get_position(0).x + blockSize - 1) / blockSize, true);
			if (int((player.get_position(0).y / blockSize + 1) + (movement_speed / blockSize) * deltaTime) >= point && point != -1) {
				player.set_position(0, Vec2(player.get_position(0).x, (point - 1) * blockSize));
			}
			else if (int((player.get_position(0).y / blockSize + 1) + (movement_speed / blockSize) * deltaTime) >= point2 && point2 != -1) {
				player.set_position(0, Vec2(player.get_position(0).x, (point2 - 1) * blockSize));
			}
			else {
				player.set_position(0, Vec2(player.get_position(0).x, player.get_position(0).y + (movement_speed * deltaTime)));
			}
		}

		line.draw();
		player.draw();
		squares.draw();

		if (KeyPress(GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, true);
		}

		GetFps();
		glfwSwapBuffers(window);
		KeysReset();
	}
}