#define REQUIRE_GRAPHICS
#include <FAN/Input.hpp>
#include <FAN/Global_Vars.hpp>

vec2i cursor_position;
vec2i window_size;

KeyCallback callbacks::key_callback;
KeyCallback callbacks::key_release_callback;
KeyCallback callbacks::scroll_callback;
default_callback callbacks::window_resize_callback;
default_callback callbacks::cursor_move_callback;
default_callback callbacks::character_callback;
default_callback callbacks::drop_callback;

auto& default_callback::get_function(uint64_t i)
{
	return functions[i];
}

uint64_t default_callback::size() const
{
	return functions.size();
}

void callbacks::glfwErrorCallback(int id, const char* error) {
	printf("GLFW Error %d : %s\n", id, error);
}

void callbacks::glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	for (int i = 0; i < callbacks::key_callback.size(); i++) {
		if (key == callbacks::key_callback.get_key(i)) {
			if (callbacks::key_callback.get_action(i)) {
				if (callbacks::key_callback.get_action(i) == static_cast<bool>(action)) {
					callbacks::key_callback.get_function(i)();
				}
			}
			else {
				callbacks::key_callback.get_function(i)();
			}
		}
	}

	static int release = 0;
	if (release) {
		for (int i = 0; i < callbacks::key_release_callback.size(); i++) {
			if (key == callbacks::key_release_callback.get_key(i)) {
				if (callbacks::key_release_callback.get_action(i) == GLFW_RELEASE) {
					callbacks::key_release_callback.get_function(i)();
				}
			}
		}
		release = -1;
	}
	release++;
}

void callbacks::MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	for (int i = 0; i < callbacks::key_callback.size(); i++) {
		if (button == callbacks::key_callback.get_key(i)) {
			if (callbacks::key_callback.get_action(i)) {
				if (callbacks::key_callback.get_action(i) == static_cast<bool>(action)) {
					callbacks::key_callback.get_function(i)();
				}
			}
			else {
				callbacks::key_callback.get_function(i)();
			}
		}
	}

	static int release = 0;
	if (release) {
		for (int i = 0; i < callbacks::key_release_callback.size(); i++) {
			if (button == callbacks::key_release_callback.get_key(i)) {
				if (callbacks::key_release_callback.get_action(i) == GLFW_RELEASE) {
					callbacks::key_release_callback.get_function(i)();
				}
			}
		}
		release = -1;
	}
	release++;
}

void callbacks::CursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
	cursor_position = vec2(xpos, ypos);
	for (int i = 0; i < callbacks::cursor_move_callback.size(); i++) {
		callbacks::cursor_move_callback.get_function(i)();
	}
}

void callbacks::ScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
	for (int i = 0; i < callbacks::scroll_callback.size(); i++) {
		if (callbacks::scroll_callback.get_key(i) == GLFW_MOUSE_SCROLL_UP && yoffset == 1) {
			callbacks::scroll_callback.get_function(i)();
		}
		else if (callbacks::scroll_callback.get_key(i) == GLFW_MOUSE_SCROLL_DOWN && yoffset == -1) {
			callbacks::scroll_callback.get_function(i)();
		}
	}
}

void callbacks::CharacterCallback(GLFWwindow* window, unsigned int key) {
	for (int i = 0; i < callbacks::character_callback.size(); i++) {
		(*(std::function<void(int, int)>*)(callbacks::character_callback.get_function_ptr(i)))(std::any_cast<int>(callbacks::character_callback.get_parameter(i, 0)), key);
	}
}

void callbacks::FrameSizeCallback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
	window_size = vec2(width, height);
	for (int i = 0; i < callbacks::window_resize_callback.size(); i++) {
		callbacks::window_resize_callback.get_function(i)();
	}
}

void callbacks::DropCallback(GLFWwindow* window, int path_count, const char* paths[]) {
	for (int i = 0; i < callbacks::drop_callback.size(); i++) {
		callbacks::drop_callback.get_function(i)();
	}
}

bool key_press(int key)
{
	if (key <= GLFW_MOUSE_BUTTON_8) {
		return glfwGetMouseButton(window, key);
	}
	return glfwGetKey(window, key);
}

void GetFps(bool, bool);

bool WindowInit() {
	glfwSetErrorCallback(callbacks::glfwErrorCallback);
	if (!glfwInit()) {
		printf("GLFW ded\n");
		static_cast<void>(system("pause") + 1);
		return 0;
	}
	window_size = WINDOWSIZE;
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//	glfwWindowHint(GLFW_RESIZABLE, false);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#ifdef FAN_CUSTOM_WINDOW
	glfwWindowHint(GLFW_DECORATED, false);
#endif
	//glfwWindowHint(GLFW_SAMPLES, 32);
	//glEnable(GL_MULTISAMPLE);
	if (fullScreen) {
		window_size = vec2(glfwGetVideoMode(glfwGetPrimaryMonitor())->width, glfwGetVideoMode(glfwGetPrimaryMonitor())->height);
	}
	window = glfwCreateWindow(window_size.x, window_size.y, "FPS: ", fullScreen ? glfwGetPrimaryMonitor() : NULL, NULL);

	if (!window) {
		printf("Window ded\n");
		static_cast<void>(system("pause") + 1);
		glfwTerminate();
		exit(EXIT_SUCCESS);
	}

	glfwMakeContextCurrent(window);
	if (GLEW_OK != glewInit())
	{
		std::cout << "Failed to initialize GLEW" << std::endl;
		static_cast<void>(system("pause") + 1);
		glfwTerminate();
		exit(EXIT_SUCCESS);
	}
	//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glViewport(0, 0, window_size.x, window_size.y);
	glEnable(GL_DEPTH_TEST);

	glfwSetKeyCallback(window, callbacks::glfwKeyCallback);
	glewExperimental = GL_TRUE;
	glfwSetCharCallback(window, callbacks::CharacterCallback);
	glfwSetMouseButtonCallback(window, callbacks::MouseButtonCallback);
	glfwSetCursorPosCallback(window, callbacks::CursorPositionCallback);
	glfwSetFramebufferSizeCallback(window, callbacks::FrameSizeCallback);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	callbacks::key_callback.add(GLFW_KEY_ESCAPE, true, [&] {
		glfwSetWindowShouldClose(window, true);
	});

	// INITIALIZATION FOR DELTA TIME
	GetFps(true, true);
	glfwSwapBuffers(window);
	glfwPollEvents();

	return 1;
}

bool window_init = WindowInit();