#include <iostream>
#include <FAN/Graphics.hpp>
#include <unordered_map>

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
	glBlendEquation(GL_FUNC_ADD);
	vec3 position;

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
			float position = 
				glfwGetVideoMode(
					glfwGetPrimaryMonitor()
				)->width / 2 -
				glfwGetVideoMode(
					glfwGetPrimaryMonitor()
				)->height / 2;
			glfwSetWindowMonitor(
				window, 
				nullptr, 
				0, 
				0, 
				glfwGetVideoMode(glfwGetPrimaryMonitor())->width, 
				glfwGetVideoMode(glfwGetPrimaryMonitor())->height, 
				0
			);
		}
		else
		{
			float position = glfwGetVideoMode(glfwGetPrimaryMonitor())->width / 2 -
				glfwGetVideoMode(glfwGetPrimaryMonitor())->height / 2;
			glfwSetWindowMonitor(window, nullptr, position, position, WINDOWSIZE.x, WINDOWSIZE.y, 0);
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

	key_callback.add(GLFW_KEY_6, true, [&] {
		texture = vec2(1,2);
	});

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

	SquareVector3D trees("sides_05.png", vec2(0, 2));

	map_t blocks(
		world_size, 
		std::vector<std::vector<bool>>(
			world_size, 
			std::vector<bool>(world_size)
		)
	);

	float yoff = 0;
	auto assign_block = [&](const vec3& position) -> decltype(auto) {
		if (position.x < 0 || position.y < 0 || position.z < 0) {
			puts("WARNING block was placed outside map");
			return blocks[0][0][0];
		}
		return blocks[(uint64_t)position.x][(uint64_t)position.y][(uint64_t)position.z];
	};

	for (int i = 0; i < world_size; i++) {
		float xoff = 0;
		for (int j = 0; j < world_size; j++) {
			vec3 position(i, ceil(ValueNoise_2D(xoff, yoff) * 800), j);
			grass.push_back(position, vec3(1), vec2(0, 0), true);
			vec3 grass_position(position);
			assign_block(position) = 1;
			xoff += 0.1;
			if (!random(0, 1000)) {
				for (int tree = 1; tree < 7; tree++) {
					assign_block(grass_position + vec3(0, tree, 0)) = 1;
					trees.push_back(grass_position + vec3(0, tree, 0), vec3(1), vec2(0, 1), true);
				}
				for (int level = 0; level < 2; level++) {
					assign_block(grass_position + vec3(-1, level + 6, 0)) = 1; // yeah will be fixed soon
					assign_block(grass_position + vec3(1, level + 6, 0)) = 1;  // yeah will be fixed soon
					assign_block(grass_position + vec3(0, level + 6, -1)) = 1; // yeah will be fixed soon
					assign_block(grass_position + vec3(0, level + 6, 1)) = 1;  // yeah will be fixed soon
					trees.push_back(grass_position + vec3(-1, level + 6, 0), vec3(1), vec2(0, 2), true); // yeah will be fixed soon
					trees.push_back(grass_position + vec3(1, level + 6, 0), vec3(1), vec2(0, 2), true);  // yeah will be fixed soon
					trees.push_back(grass_position + vec3(0, level + 6, -1), vec3(1), vec2(0, 2), true); // yeah will be fixed soon
					trees.push_back(grass_position + vec3(0, level + 6, 1), vec3(1), vec2(0, 2), true);  // yeah will be fixed soon
				}
				assign_block(grass_position + vec3(0, 7, 0)) = 1;
				assign_block(grass_position + vec3(0, 8, 0)) = 1;
				trees.push_back(grass_position + vec3(0, 7, 0), vec3(1), vec2(0, 2), true);
				trees.push_back(grass_position + vec3(0, 8, 0), vec3(1), vec2(0, 2), true);
			}
		}
		yoff += 0.1;
	}

	LineVector3D lines(matrix<3, 2>(vec3(), vec3()), Color(1, 0, 0));

	key_callback.add(GLFW_MOUSE_BUTTON_RIGHT, true, [&] {
		vec3 view = DirectionVector(camera3d.yaw, camera3d.pitch) * 1000;
		lines.set_position(0, matrix<3, 2>(vec3(position.x, position.y, position.z), view + vec3(position.x, position.y, position.z)));
		int i = 0;
		vec3i grid = grid_raycast(position, position + view, blocks, 1);
		if (!ray_hit(grid)) {
			return;
		}

		if (assign_block(grid)) {
			float distance = INFINITY;
			int side = 0;
			int closest_side = e_cube_loop.size();
			for (auto i : e_cube_loop) {
				vec3 l_point = intersection_point3d(Cast<vec3::type>(grid) + 0.5, vec3(1), position, i); // probably not the best idea
				if ((distance > g_distances[eti(i)]) && ray_hit(l_point)) {
					distance = g_distances[eti(i)];
					closest_side = side;
				}
				side++;
			}
			static const vec3 directions[] = {
				-vec3(0, 0, 1), vec3(0, 0, 1),
				-vec3(1, 0, 0), vec3(1, 0, 0),
				-vec3(0, 1, 0), vec3(0, 1, 0),
				vec3(INFINITY)
			};
			const vec3 new_block = grid + directions[closest_side];
			if (directions[closest_side] != INFINITY) {
				grass.push_back(new_block, vec3(1), texture, true, true);
				assign_block(new_block) = 1;
			}
		}
	});

	key_callback.add(GLFW_MOUSE_BUTTON_LEFT, true, [&] {
		vec3 view = DirectionVector(camera3d.yaw, camera3d.pitch) * 1000;
		lines.set_position(0, matrix<3, 2>(vec3(position.x, position.y, position.z), view + vec3(position.x, position.y, position.z)));
		vec3i grid = grid_raycast(position, position + view, blocks, 1);
		if (!ray_hit(grid)) {
			return;
		}
		if (assign_block(grid)) {
			std::vector<vec3> positions = grass.get_positions(); // can't erase trees yet
			auto found = std::find_if(
				positions.begin(), 
				positions.end(), 
				[&] (const vec3& p) {
					return p == grid; 
				}
			);
			if (found != positions.end()) {
				auto at = std::distance(positions.begin(), found);
				assign_block(grid) = 0;
				grass.erase(at);
			}
		}
	});

	position = grass.get_position(world_size * world_size / 2);

	while (!glfwWindowShouldClose(window)) {
		glClearColor(135.f / 255.f, 206.f / 255.f, 235.f / 255.f, 1.0f);
		//glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		move_camera(position, 1, 10);

		grass.draw();
		trees.draw();
		crosshair.draw();
		lines.draw();

		GetFps();
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}