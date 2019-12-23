#include "Input.hpp"

namespace CursorNamespace {
	__Vec2<int> cursorPos;
}

namespace WindowNamespace {
	__Vec2<int> windowSize;
}

namespace Input {
	bool key[1024];
	bool readchar;
	std::string characters;
	bool action[348];
	bool released[1024];
}

void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
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