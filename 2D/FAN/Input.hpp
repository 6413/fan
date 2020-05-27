#pragma once
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <FAN/Vectors.hpp>
#include <string>
#include <functional>
#include <any>
#include <vector>

constexpr bool fullScreen = false;

#define GLFW_MOUSE_SCROLL_UP 200
#define GLFW_MOUSE_SCROLL_DOWN 201

extern _vec2<int> cursor_position;
extern _vec2<int> window_size;

template <typename T>
struct default_callback {
public:
	default_callback() : functions() {}
	void add(const std::function<T>& function) {
		functions.push_back(function);
	}
	inline auto get_function(uint64_t ind) const {
		return functions[ind];
	}
	inline uint64_t size() const {
		return functions.size();
	}
protected:
	std::vector<std::function<T>> functions;
};

class KeyCallback : public default_callback<void()> {
public:
	KeyCallback() : action(), key() {}
	void add(int key, int action, const std::function<void()>& function) {
		this->action.push_back(action);

		this->key.push_back(key);
		functions.push_back(function);
	}
	inline bool get_action(uint64_t ind) const {
		return action[ind];
	}
	inline int get_key(uint64_t ind) const {
		return key[ind];
	}
private:
	using default_callback::add;
	std::vector<int> action;
	std::vector<int> key;
};

extern class KeyCallback key_callback;
extern struct default_callback<void()> window_resize_callback;
extern struct default_callback<void()> cursor_move_callback;
extern struct default_callback<void(int key)> character_callback;
extern class KeyCallback key_release_callback;
extern class KeyCallback scroll_callback;
extern struct default_callback<void(int file_count, const char** path)> drop_callback;

void GlfwErrorCallback(int id, const char* error);
void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void CursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
void CharacterCallback(GLFWwindow* window, unsigned int key);
void FrameSizeCallback(GLFWwindow* window, int width, int height);
void FocusCallback(GLFWwindow* window, int focused);
void DropCallback(GLFWwindow* window, int path_count, const char* paths[]);

bool WindowInit();

bool cursor_inside_window();
bool window_focused();