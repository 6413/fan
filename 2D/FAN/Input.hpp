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
	extern bool action;
}

template <typename _Ty, typename _Ty2>
constexpr void GlfwErrorCallback(_Ty Num, _Ty2 Desc) {
	printf("GLFW Error %d : %s\n", Num, Desc);
}

constexpr void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (!action) {
		Input::key[key] = false;
		return;
	}
	Input::key[key] = true;
}

template <typename _Ty>
constexpr void MouseButtonCallback(_Ty window, int button, int action, int mods) {
	if (Input::key[button]) {
		return;
	}
	if ((button == GLFW_MOUSE_BUTTON_LEFT ||
		button == GLFW_MOUSE_BUTTON_RIGHT)
		&& action == GLFW_PRESS) {
		Input::key[button] = true;
	}
	else {
		Input::key[button] = false;
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

template <typename _Ty, typename _Ty2 = Mat4x4>
constexpr void FrameSizeCallback(_Ty window, int width, int height) {
	using namespace WindowNamespace;
	_Ty2 projection(1);
	GLuint projLoc = glGetUniformLocation(3, "projection");
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection.vec[0].x);
	glViewport(0, 0, width, height);
	windowSize = Vec2(width, height);
}

template <typename _Ty>
constexpr void WindowInit(_Ty& window) {
	using namespace WindowNamespace;
	//windowSize = Vec2(1080, 1080);
	windowSize = Vec2(glfwGetVideoMode(glfwGetPrimaryMonitor())->width, glfwGetVideoMode(glfwGetPrimaryMonitor())->height);
	window = glfwCreateWindow(windowSize.x, windowSize.y, "Window", fullScreen ? glfwGetPrimaryMonitor() : NULL, NULL);
	if (!window) {
		printf("Window ded\n");
		glfwTerminate();
		exit(EXIT_SUCCESS);
	}
	glfwWindowHint(GLFW_DOUBLEBUFFER, 1);
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_SAMPLES, 4);
	glEnable(GL_MULTISAMPLE);
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
	glViewport(0, 0, windowSize.x, windowSize.y);
	glfwMakeContextCurrent(window);
	glewInit();
}

inline bool KeyPress(int key) {
	if (Input::key[key]) {
		return true;
	}
	return false;
}

inline bool KeyPressA(int key) {
	bool localkey = KeyPress(key);
	if (localkey && !Input::action) {
		Input::action = true;
		return true;
	}
	else if (!localkey) {
		Input::action = false;
	}
	return false;
}

constexpr void ReadCharacters(bool condition) {
	Input::readchar = condition;
}