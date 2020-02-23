#pragma once
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <FAN/Vectors.hpp>
#include <string>

constexpr bool fullScreen = false;

#define GLFW_MOUSE_SCROLL_UP 200
#define GLFW_MOUSE_SCROLL_DOWN 201

namespace CursorNamespace{
	extern _vec2<int> cursor_position;
}

namespace WindowNamespace {
	extern _vec2<int> window_size;
}

namespace Input {
	extern bool key[1024];
	extern bool action[348];
	extern bool released[1024];
}

void GlfwErrorCallback(int id, const char* error);
void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void CursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
void CharacterCallback(GLFWwindow* window, unsigned int key);
void FrameSizeCallback(GLFWwindow* window, int width, int height);

void WindowInit();

bool KeyPress(int key);
bool KeyPressA(int key);
void KeysReset();