#include <iostream>
#include <FAN/Graphics.hpp>

int main() {
	glfwSetErrorCallback(GlfwErrorCallback);
	if (!glfwInit()) {
		printf("GLFW ded\n");
		return 0;
	}
	WindowInit();
	Camera camera;
	glfwSetKeyCallback(window, KeyCallback);
	glfwSetCharCallback(window, CharacterCallback);
	glfwSetScrollCallback(window, ScrollCallback);
	glfwSetWindowUserPointer(window, &camera);
	glfwSetCursorPosCallback(window, CursorPositionCallback);
	glfwSetMouseButtonCallback(window, MouseButtonCallback);
	glfwSetFramebufferSizeCallback(window, FrameSizeCallback);

	Entity player("player.bmp", window_size / 2, vec2(block_size), 90);
	Line rays(mat2x2(player.get_position(), vec2(100)), Color(1, 1, 1));

	const float rayLength = 500;
	const std::size_t ray_amount = 100;

	for (int i = 0; i < ray_amount - 1; i++) {
		rays.push_back(mat2x2(), Color(1, 1, 1), true);
	}

	rays.break_queue();

	Square squares;

	bool map[grid_size.x][grid_size.y]{ false };

	for (int j = 0; j < grid_size.y; j++) {
		for (int i = 0; i < grid_size.x; i++) {
			squares.push_back(vec2(i * block_size, j * block_size), block_size, Color(), true);
		}
	}
	squares.break_queue();


	Timer timer;
	Alloc<vec2> ray(ray_amount);

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		vec2 a = player.get_position();
		vec2 b = cursor_position;
		vec2 c = b - a;

		vec2 end_point = a + (c / Distance(a, b) * rayLength);

		if (KeyPressA(GLFW_MOUSE_BUTTON_LEFT)) {
			std::size_t position(_2d_1d());
			Color color(squares.get_color(position));
			color ^= Color(1, 0, 0);
			squares.set_color(position, color);
			map[cursor_position.floored(block_size).x][cursor_position.floored(block_size).y] =
				!map[cursor_position.floored(block_size).x][cursor_position.floored(block_size).y];
		}

		for (int i = 0; i < ray_amount; i++) {
			float theta = float(i) / float(ray_amount);
			vec2 direction(vec2(sin(theta) * rayLength, cos(theta) * rayLength));
			direction = a + direction;
			ray[i] = Raycast(a, direction, squares, map);

			if (!on_hit(ray[i], [&rays, &ray, a, i]() { rays.set_position(i, mat2x2(a, ray[i]), true); })) {
				rays.set_position(i, mat2x2(a, direction), true);
			}
		}
		rays.break_queue();

		timer.restart();
		squares.draw();
		player.move();
		rays.draw();
		player.draw();

		if (KeyPress(GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, true);
		}

		GetFps();
		glfwSwapBuffers(window);
		KeysReset();
	}
}