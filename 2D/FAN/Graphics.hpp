#pragma once

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Alloc.hpp"
#include "Camera.hpp"
#include "Input.hpp"
#include "Math.hpp"
#include "Settings.hpp"
#include "Shader.h"
#include <vector>

#ifdef _MSC_VER
#pragma warning (disable : 26495)
#endif

#define BackgroundSize 1500
#define COLORSIZE 4
#define COORDSIZE 2

class Sprite;
class Main;
class Entity;
class Square;
struct ImageData;
enum class GroupId;

using namespace Settings;
using namespace WindowNamespace;
using namespace CursorNamespace;

struct Texture {
public:
	unsigned int texture;
	int width, height;
	unsigned int VBO, VAO, EBO;
	Texture() : texture(0), width(0), height(0), VBO(0), VAO(0), EBO(0) { }
};

//class Texture {
//public:
//	Texture() : vertices{ 0 } { }
//	void IntializeImage(Texture& texture);
//	float vertices[30];
//};

class Sprite {
public:
	Sprite() : texture()/*,texture()*/, position(0) {}

	Sprite(const Sprite& info);

	Sprite(Camera* camera, const char* path, Vec2 size = Vec2(), Vec2 position = Vec2(), Shader shader = Shader("GLSL/core.vs", "GLSL/core.frag"), float angle = 0);

	void SetPosition(const Vec2& position);

	void Draw();

	Vec2 Size() const {
		return this->size;
	}

	//Texture GetTexture() const {
	//	return this->texture;
	//}

	//Texture GetObject() const {
	//	return this->object;
	//}

protected:
	Camera* camera;
	Shader shader;
	Texture texture;
	//Texture texture;
	Vec2 position;
	Vec2 size;
	float angle;
};

class DefaultShape {
public:
	template <typename _Color = Color>
	constexpr _Color get_color(size_t _Index) const {
		return Color(
			_Colors[_Index * COLORSIZE * (_PointSize / 2)],
			_Colors[_Index * COLORSIZE * (_PointSize / 2) + 1],
			_Colors[_Index * COLORSIZE * (_PointSize / 2) + 2],
			_Colors[_Index * COLORSIZE * (_PointSize / 2) + 3]
		);
	}
	constexpr auto& get_color_ptr() {
		return _Colors;
	}
	template <typename _Color>
	constexpr void set_color(size_t _Index, const _Color& color) {
		for (int i = 0; i < COLORSIZE * (_PointSize / 2); i++) {
			_Colors[_Index * (COLORSIZE * (_PointSize / 2)) + i] = color[i % 4];
		}
		write(false, true);
	}
	template <typename _Matrix = Mat4x4>
	void draw() {
		if (_Vertices.empty()) {
			return;
		}
		_Shader.Use();
		_Matrix view(1);
		_Matrix projection(1);
		view = _Camera->GetViewMatrix(Translate(view, Vec3(windowSize.x / 2, windowSize.y / 2, -700.0f)));
		projection = Ortho(windowSize.x / 2, windowSize.x + windowSize.x * 0.5, windowSize.y + windowSize.y * 0.5f, windowSize.y / 2, 0.1f, 1000.0f);
		int projLoc = glGetUniformLocation(_Shader.ID, "projection");
		int viewLoc = glGetUniformLocation(_Shader.ID, "view");
		glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection.vec[0].x);
		glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view.vec[0].x);
		glBindVertexArray(_ShapeBuffer.VAO);
		glBindBuffer(GL_ARRAY_BUFFER, _VerticeBuffer.VBO);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, _ColorBuffer.VBO);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glDrawArrays(_Mode, 0, _Points);
		glBindVertexArray(0);
	}
protected:
	void init() {
		this->_Camera = (Camera*)glfwGetWindowUserPointer(window);
		this->_Shader = Shader("GLSL/shapes.vs", "GLSL/shapes.frag");
		glGenBuffers(1, &_VerticeBuffer.VBO);
		glBindBuffer(GL_ARRAY_BUFFER, _VerticeBuffer.VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(_Vertices[0]) * _Vertices.size(), _Vertices.data(), GL_STATIC_DRAW);
		glGenBuffers(1, &_ColorBuffer.VBO);
		glBindBuffer(GL_ARRAY_BUFFER, _ColorBuffer.VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(_Colors[0]) * _Colors.size(), _Colors.data(), GL_STATIC_DRAW);
		glGenVertexArrays(1, &_ShapeBuffer.VAO);
		glBindVertexArray(_ShapeBuffer.VAO);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}
	void write(bool _EditVertices, bool _EditColor) {
		if (_EditVertices) {
			glBindBuffer(GL_ARRAY_BUFFER, _VerticeBuffer.VBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(_Vertices[0]) * _Vertices.size(), _Vertices.data(), GL_STATIC_DRAW);
		}
		if (_EditColor) {
			glBindBuffer(GL_ARRAY_BUFFER, _ColorBuffer.VBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(_Colors[0]) * _Colors.size(), _Colors.data(), GL_STATIC_DRAW);
		}
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	Texture _VerticeBuffer;
	Texture _ColorBuffer;
	Texture _ShapeBuffer;
	Shader _Shader;
	Camera* _Camera;
	std::vector<float> _Vertices;
	std::vector<float> _Colors;
	unsigned int _Mode;
	int _Points;
	size_t _PointSize;
};

class Line : public DefaultShape {
public:
	Line() {
		_Mode = GL_LINES;
		_Points = 0;
		_PointSize = 2 * 2;
		init();
	}

	template <typename _Matrix, typename _Color>
	constexpr Line(const _Matrix& _M, const _Color& color) {
		_Mode = GL_LINES;
		_Points = 2;
		_PointSize = _Points * 2;
		_Length.push_back(Vec2(_M[1] - _M[0]));
		for (int i = 0; i < 4; i++) {
			_Vertices.push_back(_M[(i & 2) >> 1][i & 1]);
		}
		for (int i = 0; i < COLORSIZE * _PointSize; i++) {
			_Colors.push_back(color[i % 4]);
		}
		init();
	}
	template <typename _Matrix = Mat2x2>
	constexpr _Matrix get_position(size_t _Index) const {
		return Mat2x2(
			Vec2(_Vertices[_Index * _PointSize], _Vertices[_Index * _PointSize + 1]),
			Vec2(_Vertices[_Index * _PointSize + 2], _Vertices[_Index * _PointSize + 3])
		);
	}
	template <typename _Matrix>
	constexpr void set_position_queue(size_t _Index, const _Matrix& _M) {
		for (int i = 0; i < 4; i++) {
			_Vertices[_Index * _PointSize + i] = _M[(i & 2) >> 1][i & 1];
		}
		_Length[_Index] = Vec2(_M[1] - _M[0]);
	}
	void break_queue() {
		write(true, true);
	}
	template <typename _Matrix>
	constexpr void set_position(size_t _Index, const _Matrix& _M) {
		for (int i = 0; i < 4; i++) {
			_Vertices[_Index * _PointSize + i] = _M[(i & 2) >> 1][i & 1];
		}
		write(true, false);
		_Length[_Index] = Vec2(_M[1] - _M[0]);
	}
	template <typename _Matrix, typename _Color = Color>
	constexpr void push_back(const _Matrix& _M, _Color color = Color(-1, -1, -1, -1), bool queue = false) {
		_Length.push_back(Vec2(_M[1] - _M[0]));
		for (int i = 0; i < 4; i++) {
			_Vertices.push_back(_M[(i & 2) >> 1][i & 1]);
		}
		if (color.r == -1) {
			if (_Colors.size() > COLORSIZE) {
				color = Color(_Colors[0], _Colors[1], _Colors[2], _Colors[3]);
			}
			else {
				color = Color(1, 1, 1, 1);
			}
			for (int i = 0; i < COLORSIZE * 2; i++) {
				_Colors.push_back(color[i % 4]);
			}
		}
		else {
			for (int i = 0; i < COLORSIZE * 2; i++) {
				_Colors.push_back(color[i % 4]);
			}
		}
		_Points += 2;
		if (!queue) {
			write(true, true);
		}
	}
	template <typename _Vec2 = Vec2>
	constexpr _Vec2 get_length(size_t _Index) {
		return _Length[_Index];
	}
private:
	Alloc<Vec2> _Length;
};

class Triangle : public DefaultShape {
public:
	template <typename _Camera, typename _Vec2, typename _Color>
	constexpr Triangle(_Camera camera, const _Vec2& _Position, const _Vec2& _Length, const _Color& color) {
		_Mode = GL_TRIANGLES;
		_Points = 3;
		_PointSize = _Points * 2;
		this->_Length.push_back(_Length);
		this->_Position.push_back(_Position);
		_Vertices.push_back(_Position.x - (_Length.x / 2));
		_Vertices.push_back(_Position.y + (_Length.y / 2));
		_Vertices.push_back(_Position.x + (_Length.x / 2));
		_Vertices.push_back(_Position.y + (_Length.y / 2));
		_Vertices.push_back(_Position.x);
		_Vertices.push_back(_Position.y - (_Length.y / 2));
		for (int i = 0; i < COLORSIZE * _PointSize; i++) {
			_Colors.push_back(color[i % 4]);
		}
		init(camera);
	}
	~Triangle() {}
	template <typename _Vec2, typename _Color = Color>
	constexpr void push_back(const _Vec2 _Position, _Vec2 _Length = Vec2(), _Color color = Color(-1, -1, -1, -1)) {
		if (!_Length.x) {
			_Length = this->_Length[0];
		}
		this->_Length.push_back(_Length);
		this->_Position.push_back(_Position);
		_Vertices.push_back(_Position.x - (_Length.x / 2));
		_Vertices.push_back(_Position.y + (_Length.y / 2));
		_Vertices.push_back(_Position.x + (_Length.x / 2));
		_Vertices.push_back(_Position.y + (_Length.y / 2));
		_Vertices.push_back(_Position.x);
		_Vertices.push_back(_Position.y - (_Length.y / 2));
		if (color.r == -1) {
			if (_Colors.size() > COLORSIZE) {
				color = Color(_Colors[0], _Colors[1], _Colors[2], _Colors[3]);
			}
			else {
				color = Color(1, 1, 1, 1);
			}
			for (int i = 0; i < COLORSIZE * 2; i++) {
				_Colors.push_back(color[i % 4]);
			}
		}
		else {
			for (int i = 0; i < COLORSIZE * 2; i++) {
				_Colors.push_back(color[i % 4]);
			}
		}
		_Points += 3;
		write(true, true);
	}
	template <typename _Vec2>
	constexpr void set_position(size_t _Index, const _Vec2& _Position) {
		_Vertices[_Index * _PointSize] = (_Position.x - (_Length[_Index].x / 2));
		_Vertices[_Index * _PointSize + 1] = (_Position.y + (_Length[_Index].y / 2));
		_Vertices[_Index * _PointSize + 2] = (_Position.x + (_Length[_Index].x / 2));
		_Vertices[_Index * _PointSize + 3] = (_Position.y + (_Length[_Index].y / 2));
		_Vertices[_Index * _PointSize + 4] = (_Position.x);
		_Vertices[_Index * _PointSize + 5] = (_Position.y - (_Length[_Index].y / 2));
		write(true, false);
	}
	template <typename _Vec2>
	constexpr _Vec2 get_position(size_t _Index) {
		return _Position[_Index];
	}
private:
	Alloc<Vec2> _Position;
	Alloc<Vec2> _Length;
};

class Square : public DefaultShape {
public:
	Square() {
		_Mode = GL_QUADS;
		_Points = 0;
		_PointSize = 4 * 2;
		init();
	}

	template <typename _Vec2, typename _Color>
	constexpr Square(const _Vec2& _Position, const _Vec2& _Length, const _Color& color) {
		_Mode = GL_QUADS;
		_Points = 4;
		_PointSize = _Points * 2;
		this->_Length.push_back(_Length);
		this->_Position.push_back(_Position);
		_Vertices.push_back(_Position.x);
		_Vertices.push_back(_Position.y);
		_Vertices.push_back(_Position.x + _Length.x);
		_Vertices.push_back(_Position.y);
		_Vertices.push_back(_Position.x + _Length.x);
		_Vertices.push_back(_Position.y + _Length.y);
		_Vertices.push_back(_Position.x);
		_Vertices.push_back(_Position.y + _Length.y);

		for (int i = 0; i < COLORSIZE * _PointSize; i++) {
			_Colors.push_back(color[i % 4]);
		}
		init();
	}
	template <typename _Vec2, typename _Color = Color>
	constexpr void push_back(const _Vec2& _Position, _Vec2 _Length = Vec2(), _Color color = Color(-1, -1, -1, -1), bool queue = false) {
		if (!_Length.x) {
			_Length = this->_Length[0];
		}
		this->_Length.push_back(_Length);
		this->_Position.push_back(_Position - _Length / 2);
		_Vertices.push_back(_Position.x);
		_Vertices.push_back(_Position.y);
		_Vertices.push_back(_Position.x + _Length.x);
		_Vertices.push_back(_Position.y);
		_Vertices.push_back(_Position.x + _Length.x);
		_Vertices.push_back(_Position.y + _Length.y);
		_Vertices.push_back(_Position.x);
		_Vertices.push_back(_Position.y + _Length.y);
		if (color.r == -1) {
			if (_Colors.size() > COLORSIZE) {
				color = Color(_Colors[0], _Colors[1], _Colors[2], _Colors[3]);
			}
			else {
				color = Color(1, 1, 1, 1);
			}
			for (int i = 0; i < COLORSIZE * 2; i++) {
				_Colors.push_back(color[i % 4]);
			}
		}
		else {
			for (int i = 0; i < COLORSIZE * _PointSize; i++) {
				_Colors.push_back(color[i % 4]);
			}
		}
		_Points += 4;
		if (!queue) {
			write(true, true);
		}
	}
	void break_queue() {
		write(true, true);
	}
	template <typename _Vec2>
	constexpr void set_position(size_t _Index, const _Vec2& _Position) {
		_Vertices[_Index * _PointSize] = _Position.x;
		_Vertices[_Index * _PointSize + 1] = _Position.y;
		_Vertices[_Index * _PointSize + 2] = _Position.x + _Length[_Index].x;
		_Vertices[_Index * _PointSize + 3] = _Position.y;
		_Vertices[_Index * _PointSize + 4] = _Position.x + _Length[_Index].x;
		_Vertices[_Index * _PointSize + 5] = _Position.y + _Length[_Index].y;
		_Vertices[_Index * _PointSize + 6] = _Position.x;
		_Vertices[_Index * _PointSize + 7] = _Position.y + _Length[_Index].y;
		write(true, false);
		this->_Position[_Index] = _Position;
	}
	inline Mat2x4 get_corners(size_t _Index) const {
		return Mat2x4(
			Vec2(_Vertices[_Index * _PointSize], _Vertices[_Index * _PointSize + 1]),
			Vec2(_Vertices[_Index * _PointSize + 2], _Vertices[_Index * _PointSize + 3]),
			Vec2(_Vertices[_Index * _PointSize + 4], _Vertices[_Index * _PointSize + 5]),
			Vec2(_Vertices[_Index * _PointSize + 6], _Vertices[_Index * _PointSize + 7])
		);
	}
	template <typename _Vec2 = Vec2>
	constexpr _Vec2 get_position(size_t _Index) {
		return _Position[_Index];
	}
	template <typename _Vec2 = Vec2>
	constexpr _Vec2 get_length(size_t _Index) {
		return _Length[_Index];
	}
	void erase(size_t _Index) {
		//_Index += 1;
		for (int i = 0; i < 8; i++) {
			_Vertices.erase(_Vertices.begin() + _Index * _PointSize);
		}
		for (int i = 0; i < COLORSIZE * _PointSize; i++) {
			_Colors.erase(_Vertices.begin() + _Index * _PointSize);
		}
		write(true, true);
	}
private:
	Alloc<Vec2> _Position;
	Alloc<Vec2> _Length;
};

static std::vector<size_t> ind;

template <typename _Square, typename _Matrix, typename _Alloc, typename _ReturnType = Vec2>
constexpr _ReturnType Raycast(const _Square& grid, const _Matrix& direction, const _Alloc& walls, size_t gridSize, bool right) {
	//_ReturnType best = 
	if (right) {
		for (auto i = ind.begin(); i != ind.end(); ++i) {
			if (direction[1].x >= direction[0].x) {
				Vec2 inter = IntersectionPoint(direction[0], direction[1], grid.get_corners(*i)[0], grid.get_corners(*i)[3]);
				if (inter.x != -1) {
					return inter;
				}
			}
			else {
				Vec2 inter = IntersectionPoint(direction[0], direction[1], grid.get_corners(*i)[1], grid.get_corners(*i)[2]);
				if (inter.x != -1) {
					return inter;
				}
			}
			if (direction[1].y <= direction[0].y) {
				Vec2 inter = IntersectionPoint(direction[0], direction[1], grid.get_corners(*i)[2], grid.get_corners(*i)[3]);
				if (inter.x != -1) {
					return inter;
				}
			}
			else {
				Vec2 inter = IntersectionPoint(direction[0], direction[1], grid.get_corners(*i)[0], grid.get_corners(*i)[1]);
				if (inter.x != -1) {
					return inter;
				}
			}
		}
	}
	else {
		for (auto i = ind.rbegin(); i != ind.rend(); ++i) {
			if (direction[1].x >= direction[0].x) {
				Vec2 inter = IntersectionPoint(direction[0], direction[1], grid.get_corners(*i)[0], grid.get_corners(*i)[3]);
				if (inter.x != -1) {
					return inter;
				}
			}
			else {
				Vec2 inter = IntersectionPoint(direction[0], direction[1], grid.get_corners(*i)[1], grid.get_corners(*i)[2]);
				if (inter.x != -1) {
					return inter;
				}
			}
			if (direction[1].y <= direction[0].y) {
				Vec2 inter = IntersectionPoint(direction[0], direction[1], grid.get_corners(*i)[2], grid.get_corners(*i)[3]);
				if (inter.x != -1) {
					return inter;
				}
			}
			else {
				Vec2 inter = IntersectionPoint(direction[0], direction[1], grid.get_corners(*i)[0], grid.get_corners(*i)[1]);
				if (inter.x != -1) {
					return inter;
				}
			}
		}
	}
	
	return Vec2(-1, -1);
}


//enum class GroupId {
//	NotAssigned = -1,
//	LocalPlayer = 0,
//	Enemy = 1
//};

//class Entity : public Sprite {
//public:
//	Entity() : groupId(GroupId::NotAssigned) {}
//
//	Entity(const char* path, GroupId _groupId);
//
//	Entity(const Object& object) {
//		this->object = object;
//	}
//
//	Entity(Camera* camera, const char* path, const Vec2& size, Vec2 position = Vec2(), GroupId _groupId = GroupId::LocalPlayer) : 
//		velocity(0), groupId(_groupId), Sprite(camera, path, size, position) {
//		this->objects.push_back(this->object);
////		this->textures.push_back(this->texture);
//	}
//
//	GroupId GetGroupId();
//
//	void SetGroupId(GroupId groupId);
//	void Move();
//
//	Vec2 GetPosition() const {
//		return this->position;
//	}
//
//	void SetImage(const Sprite& sprite) {
//		this->objects.push_back(sprite.GetObject());
////		this->textures.push_back(sprite.GetTexture());
//	}
//
//private:
//	std::vector<Object> objects;
//	std::vector<Texture> textures;
//	GroupId groupId;
//	Vec2 velocity;
//	float health;
//	float gravity;
//	const float movementSpeed = 10000;
//	const float friction = 10;
//	const float jumpForce = 800;
//};

class Main {
public:
	//std::vector<Entity> entity;
	Shader shader = Shader("GLSL/core.vs", "GLSL/core.frag");
	Camera camera;
	GLFWwindow* window;
	Main() : camera(Vec3()) {}
};