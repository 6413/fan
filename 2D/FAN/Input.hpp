#pragma once
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "Vectors.hpp"
#include <string>

constexpr bool fullScreen = false;

namespace CursorNamespace{
	extern __Vec2<int> cursorPos;
}

namespace WindowNamespace {
	extern __Vec2<int> windowSize;
}

namespace Input {
	extern bool key[1024];
	extern bool readchar;
	extern std::string characters;
	extern bool action[348];
	extern bool released[1024];
}

template <typename _Ty, typename _Ty2>
constexpr void GlfwErrorCallback(_Ty Num, _Ty2 Desc) {
	printf("GLFW Error %d : %s\n", Num, Desc);
}

void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

constexpr void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
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
}

template <typename _Ty>
constexpr void WindowInit(_Ty& window) {
	using namespace WindowNamespace;
	windowSize = Vec2(896, 896);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	//windowSize = Vec2(glfwGetVideoMode(glfwGetPrimaryMonitor())->width, glfwGetVideoMode(glfwGetPrimaryMonitor())->height);
	window = glfwCreateWindow(windowSize.x, windowSize.y, "Window", fullScreen ? glfwGetPrimaryMonitor() : NULL, NULL);
	if (!window) {
		printf("Window ded\n");
		glfwTerminate();
		exit(EXIT_SUCCESS);
	}
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
	if (Input::key[key]) {
		return true;
	}
	return false;
}

inline bool KeyPressA(int key) {
	if (Input::action[key]) {
		return true;
	}
	return false;
}

inline void KeysReset() {
	for (int i = 0; i < 348; i++) {
		Input::action[i] = false;
	}
}