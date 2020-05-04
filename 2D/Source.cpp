#include <iostream>
#include <FAN/Graphics.hpp>

int main() {
	glfwSetErrorCallback(GlfwErrorCallback);
	if (!glfwInit()) {
		printf("GLFW ded\n");
		system("pause");
		return 0;
	}
	WindowInit();
	glEnable(GL_DEPTH_TEST);
	Camera2D cam;
	glfwSetKeyCallback(window, KeyCallback);
	glewExperimental = GL_TRUE;
	glfwSetWindowUserPointer(window, &cam);
	//glfwSetScrollCallback(window, ScrollCallback);
	glfwSetCharCallback(window, CharacterCallback);
	glfwSetWindowFocusCallback(window, FocusCallback);
	glfwSetMouseButtonCallback(window, MouseButtonCallback);
	glfwSetCursorPosCallback(window, CursorPositionCallback);
	glfwSetFramebufferSizeCallback(window, FrameSizeCallback);
	//glfwSetDropCallback(window, DropCallback);

	vec3 pos(-1, 0, 0);

	bool firstMouse = true;

	vec2 last, current;

	float lastX, lastY;

	bool fullscreen = false;

	key_callback.add(GLFW_KEY_F11, true, [&] {
		fullscreen = !fullscreen;
		if (fullscreen) {
			const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
			glfwSetWindowMonitor(
				window,
				glfwGetPrimaryMonitor(),
				0,
				0,
				mode->width,
				mode->height,
				mode->refreshRate
			);
			float pos = glfwGetVideoMode(glfwGetPrimaryMonitor())->width / 2 -
						glfwGetVideoMode(glfwGetPrimaryMonitor())->height / 2;
			glfwSetWindowMonitor(
				window, 
				nullptr, 
				0, 
				0, 
				glfwGetVideoMode(
					glfwGetPrimaryMonitor())->width, 
				glfwGetVideoMode(glfwGetPrimaryMonitor())->height, 0
			);
		}
		else
		{
			float pos = glfwGetVideoMode(glfwGetPrimaryMonitor())->width / 2 -
				glfwGetVideoMode(glfwGetPrimaryMonitor())->height / 2;
			glfwSetWindowMonitor(window, nullptr, pos, pos, WINDOWSIZE.x, WINDOWSIZE.y, 0);
		}

		});

	key_callback.add(GLFW_KEY_ESCAPE, true, [&] {
		glfwSetWindowShouldClose(window, true);
	});

	cursor_move_callback.add([&] {
		float xpos = cursor_position.x;
		float ypos = cursor_position.y;

		float& yaw = camera3d.yaw;
		float& pitch = camera3d.pitch;

		if (firstMouse)
		{
			lastX = xpos;
			lastY = ypos;
			firstMouse = false;
		}

		float xoffset = xpos - lastX;
		float yoffset = lastY - ypos;
		lastX = xpos;
		lastY = ypos;

		float sensitivity = 0.05f;
		xoffset *= sensitivity;
		yoffset *= sensitivity;

		yaw += xoffset;
		pitch += yoffset;

		if (pitch > 89.0f)
			pitch = 89.0f;
		if (pitch < -89.0f)
			pitch = -89.0f;
		});

	window_resize_callback.add([&] {
		firstMouse = true;
	});

	vec3 new_block_position;
	vec3 block_pos;

	SquareVector3D grass("sides_05.png", vec2(0, 0));

	vec2 texture(0, 0);

	key_callback.add(GLFW_KEY_1, true, [&] {
		texture = vec2(0, 0);
	});

	key_callback.add(GLFW_KEY_2, true, [&] {
		texture = vec2(0, 1);
	});

	key_callback.add(GLFW_KEY_3, true, [&] {
		texture = vec2(1, 0);
	});

	key_callback.add(GLFW_KEY_4, true, [&] {
		texture = vec2(1, 1);
	});

	auto world_size = 400;
	std::unordered_map<vec3, bool, hash_vector_operators> blocks;

	float crosshair_size = 10;
	LineVector crosshair;
	crosshair.push_back(
		mat2x2(
			vec2(
				window_size.x / 2 - crosshair_size,
				window_size.y / 2
			),
			vec2(
				window_size.x / 2 + crosshair_size,
				window_size.y / 2
			)
		)
	);

	crosshair.push_back(
		mat2x2(
			vec2(
				window_size.x / 2,
				window_size.y / 2 - crosshair_size
			),
			vec2(
				window_size.x / 2,
				window_size.y / 2 + crosshair_size
			)
		)
	);


	window_resize_callback.add([&] {
		crosshair.set_position(0,
			mat2x2(
				vec2(
					window_size.x / 2 - crosshair_size,
					window_size.y / 2
				),
				vec2(
					window_size.x / 2 + crosshair_size,
					window_size.y / 2
				)
			));
		crosshair.set_position(1,
			mat2x2(
				vec2(
					window_size.x / 2,
					window_size.y / 2 - crosshair_size
				),
				vec2(
					window_size.x / 2,
					window_size.y / 2 + crosshair_size
				)
			)
		);
	});

	float yoff = 0;
	int index = 0;
	for (int i = 0; i < world_size; i++) {
		float xoff = 0;
		for (int j = 0; j < world_size; j++) {
			vec3 position(i, ceil(ValueNoise_2D(xoff, yoff) * 800), j);
			grass.push_back(position, vec3(1), vec2(0, 0), true);
			blocks[position] = 1;
			xoff += 0.1;
		}
		yoff += 0.1;
	}

	grass.break_queue();

	pos = grass.get_position(0);

	float ray_length = 5;

	auto raycast = [&] {
		bool ray_hits = false;

		vec3 point;
		int closest_side = -1;
		int j = 0;
		block_pos = vec3();
		float distance = INFINITY;
		constexpr float ray_length = 5;
		for (int j = 0; j < grass.amount(); j++) {
			int side = 0;
			if (ManhattanDistance(grass.get_position(j), pos) > ray_length) {
				continue;
			}
			for (auto i : e_cube_loop) {
				vec3 l_point = intersection_point3d(grass.get_position(j), grass.get_size(j), pos, i);
				if ((distance > g_distances[eti(i)]) && ray_hit(l_point)) {
					block_pos = grass.get_position(j);
					distance = g_distances[eti(i)];
					closest_side = side;
					point = l_point;
					goto skip;
				}
				side++;
			}
		}
	skip:
		new_block_position = vec3();
		switch (closest_side) {
		case 0: {
			new_block_position = block_pos - vec3(0, 0, 1);
			break;
		}
		case 1: {

			new_block_position = block_pos + vec3(0, 0, 1);
			break;
		}
		case 2: {
			new_block_position = block_pos - vec3(1, 0, 0);
			break;
		}
		case 3: {

			new_block_position = block_pos + vec3(1, 0, 0);
			break;
		}
		case 4: {
			new_block_position = block_pos - vec3(0, 1, 0);
			break;
		}
		case 5: {
			new_block_position = block_pos + vec3(0, 1, 0);
			break;
		}
		default: {
			new_block_position = vec3(INFINITY);
		}
		}
	};

	key_callback.add(GLFW_MOUSE_BUTTON_LEFT, true, [&] {
		raycast();
		if (new_block_position != INFINITY) {
			auto found = std::find_if(
				grass.get_positions().begin(),
				grass.get_positions().end(),
				[&](const vec3& a) -> bool {
					return new_block_position == a;
				}
			);
			if (found == grass.get_positions().end()) {
				blocks[new_block_position] = 1;
				grass.push_back(new_block_position, vec3(1), texture);
			}
		}
	});

	key_callback.add(GLFW_MOUSE_BUTTON_RIGHT, true, [&] {
		raycast();
		if (block_pos != INFINITY) {
			auto found = std::find_if(
				grass.get_positions().begin(),
				grass.get_positions().end(),
				[&](const vec3& a) -> bool {
					return block_pos == a;
				}
			);
			if (found != grass.get_positions().end()) {
				blocks[block_pos] = 0;
				grass.erase(std::distance(grass.get_positions().begin(), found));
			}
		}
	});

	while (!glfwWindowShouldClose(window)) {
		glClearColor(135.f / 255.f, 206.f / 255.f, 235.f / 255.f, 1.0f);
		//glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		move_camera(pos, 1, 15);

		grass.draw();

		crosshair.draw();

		GetFps();
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}