#include <FAN/Graphics.hpp>

auto lambda_update_crosshair = [&](float crosshair_size) {
	return mat2x4(
		vec2(
			window_size.x / 2 - crosshair_size,
			window_size.y / 2
		),
		vec2(
			window_size.x / 2 + crosshair_size,
			window_size.y / 2
		),
		vec2(
				window_size.x / 2,
				window_size.y / 2 - crosshair_size
			),
		vec2(
			window_size.x / 2,
			window_size.y / 2 + crosshair_size
		)
	);
};

void create_crosshair(LineVector& crosshair, float crosshair_size) {
	auto position = lambda_update_crosshair(crosshair_size);
    crosshair.push_back(mat2x2(position[0], position[1]));
	crosshair.push_back(mat2x2(position[2], position[3]));
}

void update_crosshair(LineVector& crosshair, float crosshair_size) {
	auto position = lambda_update_crosshair(crosshair_size);
	crosshair.set_position(0, mat2x2(position[0], position[1]));
	crosshair.set_position(1, mat2x2(position[0], position[1]));
}

int main() {
	bool noclip = true;
	vec3& position = camera3d.position;
	key_callback.add(GLFW_KEY_LEFT_CONTROL, true, [&] {
		noclip = !noclip;
		});

	key_callback.add(GLFW_KEY_ESCAPE, true, [&] {
		glfwSetWindowShouldClose(window, true);
		});

	float crosshair_size = 10;
	LineVector crosshair;
	create_crosshair(crosshair, crosshair_size);

	window_resize_callback.add([&] {
		update_crosshair(crosshair, crosshair_size);
	});

	cursor_move_callback.add(rotate_camera);

	SquareVector3D grass("sides_05.png");

	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 100; j++) {
			grass.push_back(vec3(i, 0, j), vec3(1), vec2());
		}
	}

	while (!glfwWindowShouldClose(window)) {
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		move_camera(noclip, 200);

		crosshair.draw();

		grass.draw();

        GetFps();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}