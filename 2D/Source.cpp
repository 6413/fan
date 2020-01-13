#include <iostream>
#include <FAN/Graphics.hpp>
#include <FAN/DBT.hpp>

size_t _2D1D() {
	return (int(cursorPos.x / blockSize)) + int(cursorPos.y / blockSize) * (windowSize.y / blockSize);
}

size_t _2D1D(Vec2 pos) {
	return (int(floor(pos.x / blockSize))) + int(floor(pos.y / blockSize)) * (windowSize.y / blockSize);
}

#define movement_speed 1000000000

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

	Line line(&_Main.camera, Mat2x2(), Color(0, 1, 0, 1));

	Square dot(&_Main.camera, Vec2(), Vec2(16), Color(0, 0, 1, 1));

	line.push_back(Mat2x2(Vec2(windowSize.x / 3, blockSize), Vec2(windowSize.x / 3, blockSize + lineLength)));

	collidable.set_position(0, windowSize / 2);
	line.set_position(0, Mat2x2(Vec2(blockSize, windowSize.y / 3), Vec2(blockSize + lineLength, windowSize.y / 3)));

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
		collision.resize(4);

		for (int i = 0; i < 4; i++) {
			if (i < 2) {
				collision[i] = Raycast(grid, Mat2x2(
					Vec2(line.get_position(i % 2)[0]), Vec2(line.get_position(i % 2)[1] +
						Vec2(!i ? movement_speed : 0, i == 1 ? movement_speed : 0))
				), walls, view.x * view.y, true);
			}
			else if ( i > 1) {
				collision[i] = Raycast(grid, Mat2x2(
					Vec2(line.get_position(i % 2)[1]),
					Vec2(line.get_position(i % 2)[0] + Vec2(i == 2 ? -movement_speed : 0, i == 3 ? -movement_speed : 0))
				), walls, view.x * view.y, false);
			}

		}
		
		if (KeyPress(GLFW_MOUSE_BUTTON_LEFT)) {
			line.set_position(0, Mat2x2(cursorPos, Cast<float>(cursorPos) + line.get_size(0)));
		}

		if (KeyPress(GLFW_MOUSE_BUTTON_MIDDLE)) {
			line.set_position(1, Mat2x2(cursorPos, Cast<float>(cursorPos) + line.get_size(1)));
		}

		if (KeyPress(GLFW_KEY_A)) {
			if (collision[2].x != -1) {
				
				line.set_position(0, Mat2x2(collision[2], Vec2(collision[2].x + line.get_size(0).x, line.get_position(0)[0].y)));
			}
			else {
				line.set_position(0, Mat2x2(Vec2(line.get_position(0)[0].x - movement_speed, line.get_position(0)[0].y), Vec2(line.get_position(0)[1].x - movement_speed, line.get_position(0)[1].y)));
			}
		}
		if (KeyPress(GLFW_KEY_D)) {
			if (collision[0].x != -1) {
				line.set_position(0, Mat2x2(Vec2(collision[0].x - line.get_size(0).x, line.get_position(0)[0].y), collision[0]));
			}
			else {
				line.set_position(0, Mat2x2(Vec2(line.get_position(0)[0].x + movement_speed, line.get_position(0)[0].y), Vec2(line.get_position(0)[1].x + movement_speed, line.get_position(0)[1].y)));
			}
		}
		if (KeyPress(GLFW_KEY_S)) {
			if (collision[1].x != -1) {
				line.set_position(1, Mat2x2(Vec2(line.get_position(1)[0].x, collision[1].y - line.get_size(1).y), collision[1]));
			}
			else {
				line.set_position(1, Mat2x2(Vec2(line.get_position(1)[0].x, line.get_position(1)[0].y + movement_speed), Vec2(line.get_position(1)[1].x, line.get_position(1)[1].y + movement_speed)));
			}
		}
		if (KeyPress(GLFW_KEY_W)) {
			if (collision[3].x != -1) {
				line.set_position(1, Mat2x2(collision[3], Vec2(line.get_position(1)[0].x, collision[3].y + line.get_size(1).y)));
			}
			else {
				line.set_position(1, Mat2x2(Vec2(line.get_position(1)[0].x, line.get_position(1)[0].y - movement_speed), Vec2(line.get_position(1)[1].x, line.get_position(1)[1].y - movement_speed)));
			}
		}

		line.draw();
		grid.draw();
		if (KeyPress(GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, true);
		}

		GetFps();
		glfwSwapBuffers(window);
		KeysReset();
		collision.free();
	}
}