#include <iostream>
#include <Windows.h>
#include <FAN/Graphics.hpp>

typedef struct {
	__Vec2<int> Size;
	GLubyte* Data;
}LT_GLFont;

typedef struct {
	unsigned int Size;
	unsigned char* Data;
}LT_File;

#define LDC_File LDC_WITCH(LT_File){0, 0}

#define LDF_FileReadAll(M_Path) LF_FileReadAll((T_ptr)M_Path)
LT_File LF_FileReadAll(const char* Path) {
	FILE* in = fopen((const char*)Path, "rb");
	if (!in)
		return LT_File();
	LT_File RET = LT_File();
	fseek(in, 0, SEEK_END);
	RET.Size = ftell(in);
	fseek(in, 0, SEEK_SET);
	RET.Data = (unsigned char*)malloc(RET.Size);
	fread(RET.Data, RET.Size, 1, in);
	fclose(in);
	return RET;
}

void LF_GLCharacter(LT_GLFont Font, Color Color, Vec2 Coordinate, char Character) {
	glColor3d(Color.r, Color.g, Color.b);
	glRasterPos2i(Coordinate.x, Coordinate.y);
	glBitmap(Font.Size.x, Font.Size.y, 0, 0, 0, 0, (GLubyte*)Font.Data + (Character * (((Font.Size.x / 8) + ((Font.Size.x % 8) > 0))* Font.Size.y)));
}

class Button : public Square {
public:
	Button(const Vec2& position, const Vec2& size, const Color& color) : Square(position, size, color), count(1) {};
	template <typename _Vec2 = Vec2, typename _Color = Color>
	constexpr void add(const _Vec2& _Position = Vec2(), _Vec2 _Length = Vec2(), _Color color = Color(-1, -1, -1, -1), bool queue = false) {
		this->push_back(_Position, _Length, color, queue);
		count++;
	}

	bool pressed(size_t index) {
		return cursorPos.x >= get_position(index).x && cursorPos.x <= get_position(index).x + get_length(index).x &&
			   cursorPos.y >= get_position(index).y && cursorPos.y <= get_position(index).y + get_length(index).y && 
			   KeyPressA(GLFW_MOUSE_BUTTON_LEFT);
	}
	Vec2 get_size() const {
		return this->_Size;
	}

	size_t size() const {
		return count;
	}


private:
	using Square::push_back;
	size_t count;
	Vec2 _Size;
};

int main() {
	glfwSetErrorCallback(GlfwErrorCallback);
	if (!glfwInit()) {
		printf("GLFW ded\n");
		return 0;
	}
	WindowInit();
	Main _Main;
	glfwSetKeyCallback(window, KeyCallback);
	glfwSetCharCallback(window, CharacterCallback);
	glfwSetScrollCallback(window, ScrollCallback);
	glfwSetWindowUserPointer(window, &_Main.camera);
	glfwSetCursorPosCallback(window, CursorPositionCallback);
	glfwSetMouseButtonCallback(window, MouseButtonCallback);
	glfwSetFramebufferSizeCallback(window, FrameSizeCallback);

	Button button(Vec2(windowSize / 2), Vec2(100, 50), Color(1, 0.4, 0, 1));

	button.add();

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	

		button.draw();

		for (int i = 0; i < button.size(); i++) {
			if (button.pressed(i)) {
				printf("pressed button: %d\n", i);
			}
		}
		if (KeyPress(GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, true);
		}
		GetFps();
		glfwSwapBuffers(window);
		KeysReset();
	}
}