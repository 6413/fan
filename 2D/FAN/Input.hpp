#pragma once
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <FAN/Vectors.hpp>
#include <FAN/Settings.hpp>
#include <string>

using namespace Settings;

constexpr bool fullScreen = false;

namespace CursorNamespace{
	static __Vec2<int> cursorPos;
}

namespace WindowNamespace {
	static __Vec2<int> windowSize;
}

namespace Input {
	static bool key[1024];
	static bool readchar;
	static std::string characters;
	static bool action[348];
	static bool released[1024];
	static double* ptr;
}

template <typename _Ty, typename _Ty2>
constexpr void GlfwErrorCallback(_Ty Num, _Ty2 Desc) {
	printf("GLFW Error %d : %s\n", Num, Desc);
}

static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (!action) {
		if (key < 0) {
			return;
		}
		Input::key[key % 1024] = false;
		return;
	}

	Input::key[key] = true;

	for (int i = 32; i < 162; i++) {
		if (key == i && action == 1) {
			Input::action[key] = true;
		}
	}

	for (int i = 256; i < 384; i++) {
		if (key == i && action == 1) {
			Input::action[key] = true;
		}
	}
}

static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if ((button == GLFW_MOUSE_BUTTON_LEFT ||
		button == GLFW_MOUSE_BUTTON_RIGHT || 
		button == GLFW_MOUSE_BUTTON_MIDDLE)
		&& action == GLFW_PRESS) {
		Input::key[button] = true;
	}
	else {
		Input::key[button] = false;
	}
	for (int i = 0; i < GLFW_MOUSE_BUTTON_8; i++) {
		if (i == button && action == 1) {
			Input::action[button] = true;
		}
	}
}

static void CursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {

	CursorNamespace::cursorPos = Vec2(xpos, ypos);
}

#define GLFW_MOUSE_SCROLL_UP 200
#define GLFW_MOUSE_SCROLL_DOWN 201

static void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
	if (yoffset == 1) {
		Input::action[GLFW_MOUSE_SCROLL_UP] = true;
	}
	else if (yoffset == -1) {
		Input::action[GLFW_MOUSE_SCROLL_DOWN] = true;
	}
}

template <typename _Ty>
constexpr void CharacterCallback(_Ty window, unsigned int key) {
	if (Input::readchar) {
		Input::characters.push_back(key);
	}
}

template <typename _Ty>
constexpr void FrameSizeCallback(_Ty window, int width, int height) {
	using namespace WindowNamespace;
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	windowSize = Vec2(width, height);
//	view = windowSize / blockSize;
}

static void WindowInit() {
	using namespace WindowNamespace;
	windowSize = WINDOWSIZE;
	glfwWindowHint(GLFW_RESIZABLE, true);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	//windowSize = Vec2(glfwGetVideoMode(glfwGetPrimaryMonitor())->width, glfwGetVideoMode(glfwGetPrimaryMonitor())->height);
	window = glfwCreateWindow(windowSize.x, windowSize.y, "Window", fullScreen ? glfwGetPrimaryMonitor() : NULL, NULL);
	if (!window) {
		printf("Window ded\n");
		glfwTerminate();
		exit(EXIT_SUCCESS);
	}
	//view = windowSize / blockSize;
	//glfwWindowHint(GLFW_DOUBLEBUFFER, 1);
	glfwMakeContextCurrent(window);
	glewInit();
	glViewport(0, 0, windowSize.x, windowSize.y);
	//glfwWindowHint(GLFW_SAMPLES, 4);
//	glEnable(GL_MULTISAMPLE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_ALPHA);
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
}

inline bool KeyPress(int key) {
	if (Input::key[key % 1024]) {
		return true;
	}
	return false;
}

static bool KeyPressA(int key) {
	if (Input::action[key % 348]) {
		return true;
	}
	return false;
}

inline void KeysReset() {
	for (int i = 0; i < 348; i++) {
		Input::action[i] = false;
	}
}