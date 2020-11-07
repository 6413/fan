#define REQUIRE_GRAPHICS
#include <FAN/input.hpp>
#include <FAN/global_vars.hpp>

fan::vec2i fan::cursor_position;
fan::vec2i fan::window_size;

fan::KeyCallback fan::callback::key;
fan::KeyCallback fan::callback::key_release;
fan::KeyCallback fan::callback::scroll;
fan::default_callback fan::callback::window_resize;
fan::default_callback fan::callback::cursor_move;
fan::default_callback fan::callback::character;
fan::default_callback fan::callback::drop;

auto& fan::default_callback::get_function(uint64_t i)
{
	return functions[i];
}

uint64_t fan::default_callback::size() const
{
	return functions.size();
}

void fan::callback::glfwErrorCallback(int id, const char* error) {
	printf("GLFW Error %d : %s\n", id, error);
}

void fan::callback::glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS) {
		fan::input::action[key] = action;
	}
	for (uint_t i = 0; i < callback::key.size(); i++) {
		if (key != callback::key.get_key(i)) {
			continue;
		}
		if (callback::key.get_action(i) != action && callback::key.get_action(i)) {
			continue;
		}
		callback::key.get_function(i)();
	}

	static bool release = 0;

	if (release) {
		for (uint_t i = 0; i < callback::key_release.size(); i++) {
			if (key != callback::key_release.get_key(i)) {
				continue;
			}
			if (callback::key_release.get_action(i) != GLFW_RELEASE) {
				continue;
			}
			callback::key_release.get_function(i)();
		}
	}
	
	release = !release;
}

void fan::callback::MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	fan::input::action[button] = action;
	for (uint_t i = 0; i < callback::key.size(); i++) {
		if (button != callback::key.get_key(i)) {
			continue;
		}
		if (callback::key.get_action(i) && callback::key.get_action(i) != action) {
			continue;
		}
		callback::key.get_function(i)();
	}

	static bool release = 0;
	if (release) {
		for (uint_t i = 0; i < callback::key_release.size(); i++) {
			if (button != callback::key_release.get_key(i)) {
				continue;
			}
			if (callback::key_release.get_action(i) != GLFW_RELEASE) {
				continue;
			}
			callback::key_release.get_function(i)();
		}
	}
	release = !release;
}

void fan::callback::CursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
	cursor_position = fan::vec2i(xpos, ypos);
	for (uint_t i = 0; i < callback::cursor_move.size(); i++) {
		callback::cursor_move.get_function(i)();
	}
}

void fan::callback::ScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
	for (uint_t i = 0; i < callback::scroll.size(); i++) {
		if (callback::scroll.get_key(i) != GLFW_MOUSE_SCROLL_UP && yoffset == 1) {
			continue;
		}
		else if (callback::scroll.get_key(i) != GLFW_MOUSE_SCROLL_DOWN && yoffset == -1) {
			continue;
		}
		callback::scroll.get_function(i)();
	}
}

void fan::callback::CharacterCallback(GLFWwindow* window, unsigned int key) {
	for (uint_t i = 0; i < callback::character.size(); i++) {
		(*(std::function<void(int, int)>*)(callback::character.get_function_ptr(i)))(std::any_cast<int>(callback::character.get_parameter(i, 0)), key);
	}
}

void fan::callback::FrameSizeCallback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
	window_size = fan::vec2i(width, height);
	for (uint_t i = 0; i < callback::window_resize.size(); i++) {
		callback::window_resize.get_function(i)();
	}
}

void fan::callback::DropCallback(GLFWwindow* window, int path_count, const char* paths[]) {
	callback::drop_info.paths.clear();
	callback::drop_info.path_count = path_count;
	for (int i = 0; i < path_count; i++) {
		callback::drop_info.paths.push_back(paths[i]);
	}
	for (uint_t i = 0; i < callback::drop.size(); i++) {
		callback::drop.get_function(i)();
	}
}

int fan::input::key_press(int key, bool action) {
	if (!fan::input::previous_action[key] && fan::input::action[key]) {
		fan::input::previous_action[key] = true;
	}
	else {
		fan::input::action[key] = false;
		fan::input::previous_action[key] = false;
	}
	if (key <= GLFW_MOUSE_BUTTON_8) {
		return glfwGetMouseButton(window, key) && (action ? fan::input::action[key] : 1);
	}
	return glfwGetKey(window, key) && (action ? fan::input::action[key] : 1);
}

void fan::get_fps(bool, bool);

bool fan::initialize_window() {
	static bool initialized = false;
	if (initialized) {
		return 1;
	}
	glfwSetErrorCallback(callback::glfwErrorCallback);
	if (!glfwInit()) {
		printf("GLFW ded\n");
		static_cast<void>(system("pause") + 1);
		return 0;
	}
	fan::window_size = WINDOWSIZE;
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	//	glfwWindowHint(GLFW_RESIZABLE, false);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_DECORATED, flags::decorated);
	glfwWindowHint(GLFW_RESIZABLE, flags::resizeable);
	if constexpr (flags::antialising) {
		glfwWindowHint(GLFW_SAMPLES, 32);
		glEnable(GL_MULTISAMPLE);
	}
	if (flags::full_screen) {
		window_size = fan::vec2(glfwGetVideoMode(glfwGetPrimaryMonitor())->width, glfwGetVideoMode(glfwGetPrimaryMonitor())->height);
	}
	fan::window = glfwCreateWindow(window_size.x, window_size.y, "FPS: ", flags::full_screen ? glfwGetPrimaryMonitor() : NULL, NULL);

	if (!fan::window) {
		printf("Window ded\n");
		static_cast<void>(system("pause") + 1);
		glfwTerminate();
		exit(EXIT_SUCCESS);
	}

	glfwMakeContextCurrent(fan::window);
	if (GLEW_OK != glewInit())
	{
		std::cout << "Failed to initialize GLEW" << std::endl;
		static_cast<void>(system("pause") + 1);
		glfwTerminate();
		exit(EXIT_SUCCESS);
	}
	if constexpr (fan::flags::disable_mouse) {
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}
	glViewport(0, 0, fan::window_size.x, fan::window_size.y);
	//glEnable(GL_DEPTH_TEST);

	glfwSetKeyCallback(fan::window, fan::callback::glfwKeyCallback);
	glewExperimental = GL_TRUE;
	glfwSetCharCallback(fan::window, fan::callback::CharacterCallback);
	glfwSetScrollCallback(fan::window, fan::callback::ScrollCallback);
	glfwSetMouseButtonCallback(fan::window, fan::callback::MouseButtonCallback);
	glfwSetCursorPosCallback(fan::window, fan::callback::CursorPositionCallback);
	glfwSetFramebufferSizeCallback(fan::window, fan::callback::FrameSizeCallback);
	glfwSetDropCallback(window, fan::callback::DropCallback);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	fan::callback::key.add(GLFW_KEY_ESCAPE, true, [&] {
		glfwSetWindowShouldClose(window, true);
	});

	// initialization for delta time
	fan::get_fps(true, true);
	glfwSwapBuffers(window);
	glfwPollEvents();
	initialized = true;

	return 1;
}

bool window_init = fan::initialize_window();
