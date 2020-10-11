#pragma once
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <FAN/types.h>

#include <functional>
#include <vector>
#include <any>

constexpr bool fullScreen = false;

#define GLFW_MOUSE_SCROLL_UP 200
#define GLFW_MOUSE_SCROLL_DOWN 201

extern fan::vec2i cursor_position;
extern fan::vec2i window_size;

template <typename T>
void variadic_vector_emplace(std::vector<T>&) {}

template <typename T, typename First, typename... Args>
void variadic_vector_emplace(std::vector<T>& vector, First&& first, Args&&... args)
{
	vector.emplace_back(std::forward<First>(first));
	variadic_vector_emplace(vector, std::forward<Args>(args)...);
}

class default_callback {
public:

	~default_callback() {
		for (auto i : function_pointers) {
			delete (char*)i;
		}
	}

	template <typename ...Args, typename T>
	void add(const T& function, Args... args) {
		functions.emplace_back(std::bind(function, args...));

		T* function_adder = new T{ function };
		function_pointers.emplace_back((void*)&*function_adder);

		parameters.resize(parameters.size() + 1);

		variadic_vector_emplace(parameters[parameters.size() - 1], args...);
	}


	std::any get_parameter(uint64_t i, uint64_t parameter) {
		return parameters[i][parameter];
	}

	void* get_function_ptr(uint64_t i) {
		return function_pointers[i];
	}

	auto& get_function(uint64_t i);

	uint64_t size() const;

private:
	std::vector<std::function<void()>> functions;
	std::vector<void*> function_pointers;
	std::vector<std::vector<std::any>> parameters;
};

class KeyCallback {
public:

	template <typename ...Args, typename T>
	void add(int key, int action, const T& function, Args... args) {
		this->action.push_back(action);
		this->key.push_back(key);
		functions.push_back(std::bind(function, args...));
	}

	bool get_action(uint64_t i) const {
		return action[i];
	}

	int get_key(uint64_t i) const {
		return key[i];
	}

	uint64_t size() const {
		return functions.size();
	}

	auto get_function(uint64_t i) const
	{
		return functions[i];
	}

private:
	std::vector<std::function<void()>> functions;
	std::vector<int> action;
	std::vector<int> key;
};

namespace callback {
	extern KeyCallback key;
	extern KeyCallback key_release;
	extern KeyCallback scroll;
	extern default_callback window_resize;
	extern default_callback cursor_move;
	extern default_callback character;
	extern default_callback drop;

	void glfwErrorCallback(int id, const char* error);
	void glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
	void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
	void CursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
	void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
	void CharacterCallback(GLFWwindow* window, unsigned int key);

	void FrameSizeCallback(GLFWwindow* window, int width, int height);
	void FocusCallback(GLFWwindow* window, int focused);
	void DropCallback(GLFWwindow* window, int path_count, const char* paths[]);

	typedef struct{
		int path_count;
		std::vector<std::string> paths;
	}drop_info_t;

	inline callback::drop_info_t drop_info;
}

bool key_press(int key);

bool WindowInit();