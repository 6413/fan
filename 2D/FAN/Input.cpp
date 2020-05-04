#include "Input.hpp"
#include <FAN/Settings.hpp>
#include <string>

_vec2<int> cursor_position;
_vec2<int> window_size;

class KeyCallback key_callback;
struct default_callback<void()> window_resize_callback;
struct default_callback<void()> cursor_move_callback;
struct default_callback<void(int key)> character_callback;
class KeyCallback key_release_callback;

void GlfwErrorCallback(int id, const char* error) {
	printf("GLFW Error %d : %s\n", id, error);
}

void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	for (int i = 0; i < key_callback.size(); i++) {
		if (key == key_callback.get_key(i)) {
			if (key_callback.get_action(i)) {
				if (key_callback.get_action(i) == action) {
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
				if (key_callback.get_action(i) == action) {
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
	/*if (yoffset == 1) {
		Input::action[GLFW_MOUSE_SCROLL_UP] = true;
	}
	else if (yoffset == -1) {
		Input::action[GLFW_MOUSE_SCROLL_DOWN] = true;
	}*/
}

void CharacterCallback(GLFWwindow* window, unsigned int key) {
	for (int i = 0; i < character_callback.size(); i++) {
		character_callback.get_function(i)(key);
	}
}

void FrameSizeCallback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
	window_size = vec2(width, height);
	for (int i = 0; i < window_resize_callback.size(); i++) {
		window_resize_callback.get_function(i)();
	}
}

//void CursorEnterCallback(GLFWwindow* window, int entered) {
//
//}

void WindowInit() {
	window_size = WINDOWSIZE;
	glfwWindowHint(GLFW_DECORATED, false);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_RESIZABLE, true);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_SAMPLES, 16);
	glEnable(GL_MULTISAMPLE);
	//window_size = vec2(glfwGetVideoMode(glfwGetPrimaryMonitor())->width, glfwGetVideoMode(glfwGetPrimaryMonitor())->height);
	window = glfwCreateWindow(window_size.x, window_size.y, "Server", fullScreen ? glfwGetPrimaryMonitor() : NULL, NULL);
	if (!window) {
		printf("Window ded\n");
		glfwTerminate();
		exit(EXIT_SUCCESS);
	}
	glfwMakeContextCurrent(window);
	glewInit();
	glViewport(0, 0, window_size.x, window_size.y);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

//bool cursor_inside_window() {
//	return Input::cursor_inside_window;
//}