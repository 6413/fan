#include "Input.hpp"
#include <FAN/Graphics.hpp>

_vec2<int> cursor_position;
_vec2<int> window_size;

KeyCallback key_callback;
KeyCallback key_release_callback;
KeyCallback scroll_callback;
default_callback window_resize_callback;
default_callback cursor_move_callback;
default_callback character_callback;
default_callback drop_callback;

auto& default_callback::get_function(uint64_t i)
{
	return functions[i];
}

uint64_t default_callback::size() const
{
	return functions.size();
}

void glfwErrorCallback(int id, const char* error) {
	printf("GLFW Error %d : %s\n", id, error);
}

void glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	for (int i = 0; i < key_callback.size(); i++) {
		if (key == key_callback.get_key(i)) {
			if (key_callback.get_action(i)) {
				if (key_callback.get_action(i) == static_cast<bool>(action)) {
					key_callback.get_function(i)();
				}
			}
			else {
				key_callback.get_function(i)();
			}
		}
	}

	static int release = 0;
	if (release) {
		for (int i = 0; i < key_release_callback.size(); i++) {
			if (key == key_release_callback.get_key(i)) {
				if (key_release_callback.get_action(i) == GLFW_RELEASE) {
					key_release_callback.get_function(i)();
				}
			}
		}
		release = -1;
	}
	release++;
}

void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	for (int i = 0; i < key_callback.size(); i++) {
		if (button == key_callback.get_key(i)) {
			if (key_callback.get_action(i)) {
				if (key_callback.get_action(i) == static_cast<bool>(action)) {
					key_callback.get_function(i)();
				}
			}
			else {
				key_callback.get_function(i)();
			}
		}
	}

	static int release = 0;
	if (release) {
		for (int i = 0; i < key_release_callback.size(); i++) {
			if (button == key_release_callback.get_key(i)) {
				if (key_release_callback.get_action(i) == GLFW_RELEASE) {
					key_release_callback.get_function(i)();
				}
			}
		}
		release = -1;
	}
	release++;
}

void CursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
	cursor_position = vec2(xpos, ypos);
	for (int i = 0; i < cursor_move_callback.size(); i++) {
		cursor_move_callback.get_function(i)();
	}
}

void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
	for (int i = 0; i < scroll_callback.size(); i++) {
		if (scroll_callback.get_key(i) == GLFW_MOUSE_SCROLL_UP && yoffset == 1) {
			scroll_callback.get_function(i)();
		}
		else if (scroll_callback.get_key(i) == GLFW_MOUSE_SCROLL_DOWN && yoffset == -1) {
			scroll_callback.get_function(i)();
		}
	}
}

void CharacterCallback(GLFWwindow* window, unsigned int key) {
	for (int i = 0; i < character_callback.size(); i++) {
		(*(std::function<void(int, int)>*)(character_callback.get_function_ptr(i)))(std::any_cast<int>(character_callback.get_parameter(i, 0)), key);
	}
}

void FrameSizeCallback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
	window_size = vec2(width, height);
	for (int i = 0; i < window_resize_callback.size(); i++) {
		window_resize_callback.get_function(i)();
	}
}

void DropCallback(GLFWwindow* window, int path_count, const char* paths[]) {
	for (int i = 0; i < drop_callback.size(); i++) {
		drop_callback.get_function(i)();
	}
}

bool key_press(int key)
{
	if (key <= GLFW_MOUSE_BUTTON_8) {
		return glfwGetMouseButton(window, key);
	}
	return glfwGetKey(window, key);
}

bool WindowInit() {
	glfwSetErrorCallback(glfwErrorCallback);
	if (!glfwInit()) {
		printf("GLFW ded\n");
		static_cast<void>(system("pause") + 1);
		return 0;
	}
	window_size = WINDOWSIZE;
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//	glfwWindowHint(GLFW_RESIZABLE, false);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	//glfwWindowHint(GLFW_SAMPLES, 32);
	//glEnable(GL_MULTISAMPLE);
	//window_size = vec2(glfwGetVideoMode(glfwGetPrimaryMonitor())->width, glfwGetVideoMode(glfwGetPrimaryMonitor())->height);
	window = glfwCreateWindow(window_size.x, window_size.y, "FPS: ", fullScreen ? glfwGetPrimaryMonitor() : NULL, NULL);
	glfwSetWindowMonitor(window, NULL, window_size.x / 2, window_size.y / 2, window_size.x, window_size.y, 0);

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

	glfwSetKeyCallback(window, glfwKeyCallback);
	glewExperimental = GL_TRUE;
	glfwSetCharCallback(window, CharacterCallback);
	glfwSetMouseButtonCallback(window, MouseButtonCallback);
	glfwSetCursorPosCallback(window, CursorPositionCallback);
	glfwSetFramebufferSizeCallback(window, FrameSizeCallback);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	return 1;
}