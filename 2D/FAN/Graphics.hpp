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
	constexpr void set_color(size_t _Index, const _Color& color, bool queue = false) {
		for (int i = 0; i < COLORSIZE * (_PointSize / 2); i++) {
			_Colors[_Index * (COLORSIZE * (_PointSize / 2)) + i] = color[i % 4];
		}
		if (!queue) {
			write(false, true);
		}
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
	void break_queue() {
		write(true, true);
	}
protected:
	void init() {
		this->_Camera = (Camera*)glfwGetWindowUserPointer(window);
		this->_Shader = Shader("GLSL/shapes.vs", "GLSL/shapes.frag");
		glGenBuffers(1, &_VerticeBuffer.VBO);
		glBindBuffer(GL_ARRAY_BUFFER, _VerticeBuffer.VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(_Vertices[0]) * _Vertices.current(), _Vertices.data(), GL_STATIC_DRAW);
		glGenBuffers(1, &_ColorBuffer.VBO);
		glBindBuffer(GL_ARRAY_BUFFER, _ColorBuffer.VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(_Colors[0]) * _Colors.current(), _Colors.data(), GL_STATIC_DRAW);
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
			glBufferData(GL_ARRAY_BUFFER, sizeof(_Vertices[0]) * _Vertices.current(), _Vertices.data(), GL_STATIC_DRAW);
		}
		if (_EditColor) {
			glBindBuffer(GL_ARRAY_BUFFER, _ColorBuffer.VBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(_Colors[0]) * _Colors.current(), _Colors.data(), GL_STATIC_DRAW);
		}
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	Texture _VerticeBuffer;
	Texture _ColorBuffer;
	Texture _ShapeBuffer;
	Shader _Shader;
	Camera* _Camera;
	Alloc<float> _Vertices;
	Alloc<float> _Colors;
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
	constexpr void set_position(size_t _Index, const _Matrix& _M, bool _Queue = false) {
		for (int i = 0; i < 4; i++) {
			_Vertices[_Index * _PointSize + i] = _M[(i & 2) >> 1][i & 1];
		}
		if (!_Queue) {
			write(true, false);
		}
		_Length[_Index] = Vec2(_M[1] - _M[0]);
	}
	template <typename _Matrix, typename _Color = Color>
	constexpr void push_back(const _Matrix& _M, _Color color = Color(-1, -1, -1, -1), bool _Queue = false) {
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
		if (!_Queue) {
			write(true, true);
		}
	}
	//template <typename _Color = Color>
	//constexpr void copy_push_back(std::size_t _Index, bool _Queue = false) {
	//	//_Length.push_back(Vec2(_Vertices[_M[1] - _M[0]));
	//	for (int i = 0; i < 4; i++) {
	//		_Vertices.push_back(_Vertices[_Index * _PointSize][(i & 2) >> 1][i & 1]);
	//	}
	//	if (color.r == -1) {
	//		if (_Colors.size() > COLORSIZE) {
	//			color = Color(_Colors[0], _Colors[1], _Colors[2], _Colors[3]);
	//		}
	//		else {
	//			color = Color(1, 1, 1, 1);
	//		}
	//		for (int i = 0; i < COLORSIZE * 2; i++) {
	//			_Colors.push_back(color[i % 4]);
	//		}
	//	}
	//	else {
	//		for (int i = 0; i < COLORSIZE * 2; i++) {
	//			_Colors.push_back(color[i % 4]);
	//		}
	//	}
	//	_Points += 2;
	//	if (!_Queue) {
	//		write(true, true);
	//	}
	//}
	template <typename _Vec2 = Vec2>
	constexpr _Vec2 get_length(size_t _Index) {
		return _Length[_Index];
	}
private:
	Alloc<Vec2> _Length;
};

class Triangle : public DefaultShape {
public:
	Triangle() {
		_Mode = GL_TRIANGLES;
		_Points = 0;
		_PointSize = 3 * 2;
		init();
	}

	template <typename _Vec2, typename _Color>
	constexpr Triangle(const _Vec2& _Position, const _Vec2& _Length, const _Color& color) {
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
		init();
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
		if (!_Length.x || !_Position.x && !_Position.y) {
			_Length = this->_Length[0];
		}
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
	template <typename _Matrix = Mat2x4>
	constexpr _Matrix get_corners(size_t _Index) const {
		size_t _Multiplier = _Index * _PointSize;
		return _Matrix(
			Vec2(_Vertices[_Multiplier    ], _Vertices[_Multiplier + 1]),
			Vec2(_Vertices[_Multiplier + 2], _Vertices[_Multiplier + 3]),
			Vec2(_Vertices[_Multiplier + 4], _Vertices[_Multiplier + 5]),
			Vec2(_Vertices[_Multiplier + 6], _Vertices[_Multiplier + 7])
		);
	}
	template <typename _Vec2 = Vec2>
	constexpr _Vec2 get_position(size_t _Index) const {
		return _Position[_Index];
	}
	template <typename _Vec2 = Vec2>
	constexpr _Vec2 get_length(size_t _Index) const {
		return _Length[_Index];
	}
	constexpr std::size_t amount() const {
		return _Points / 4;
	}
	//void erase(size_t _Index) {
	//	//_Index += 1;
	//	for (int i = 0; i < 8; i++) {
	//		_Vertices.erase(_Vertices.begin() + _Index * _PointSize);
	//	}
	//	for (int i = 0; i < COLORSIZE * _PointSize; i++) {
	//		_Colors.erase(_Vertices.begin() + _Index * _PointSize);
	//	}
	//	write(true, true);
	//}
private:
	Alloc<Vec2> _Position;
	Alloc<Vec2> _Length;
};

class Circle : public DefaultShape {
public:
	Circle(std::size_t _Number_Of_Points, float _Radius) {
		_Mode = GL_LINES;
		_Points = 0;
		_PointSize = _Number_Of_Points * 2;
		this->_Radius.push_back(_Radius);
		init();
	}
	Circle(Vec2 _Position, float _Radius, std::size_t _Number_Of_Points, Color _Color) {
		_Mode = GL_LINES;
		_Points = _Number_Of_Points;
		_PointSize = _Number_Of_Points * 2;
		this->_Position.push_back(_Position);
		this->_Radius.push_back(_Radius);
		for (int ii = 0; ii < _Points; ii++) {
			float theta = 2.0f * 3.1415926f * float(ii) / float(_Points);

			float x = _Radius * cosf(theta);
			float y = _Radius * sinf(theta);

			_Vertices.push_back(_Position.x + x);
			_Vertices.push_back(_Position.y + y);
		}
		
		for (int i = 0; i < COLORSIZE * _PointSize; i++) {
			_Colors.push_back(_Color[i % 4]);
		}
		init();
	}
	void push_back(Vec2 _Position, float _Radius, Color _Color, bool _Queue = false) {
		const std::size_t _LPoints = _PointSize / 2;
		this->_Position.push_back(_Position);
		this->_Radius.push_back(_Radius);
		for (int ii = 0; ii < _Points; ii++) {
			float theta = 2.0f * 3.1415926f * float(ii) / float(_Points);

			float x = _Radius * cosf(theta);
			float y = _Radius * sinf(theta);

			_Vertices.push_back(_Position.x + x);
			_Vertices.push_back(_Position.y + y);
		}
		for (int i = 0; i < COLORSIZE * _PointSize; i++) {
			_Colors.push_back(_Color[i % 4]);
		}
		_Points += _PointSize / 2;
		if (_Queue) {
			write(true, true);
		}
	}
	template <typename _Vec2>
	constexpr void set_position(size_t _Index, const _Vec2& _Position) {
		int x = 0;
		for (int ii = 0; ii < _PointSize; ii+= 2) {
			float theta = _Double_Pi * float(ii) / float(_PointSize);

			float x = _Radius[_Index] * cosf(theta);
			float y = _Radius[_Index] * sinf(theta);
			_Vertices[_Index * _PointSize + ii] = _Position.x + x;
			_Vertices[_Index * _PointSize + ii + 1] = _Position.y + y;
		}
		this->_Position[_Index] = _Position;
		write(true, false);
	}
private:
	Alloc<float> things;
	Alloc<Vec2> _Position;
	Alloc<float> _Radius;
	const float _Double_Pi = 2.0f * PI;
	const float diameter = windowSize.x / 3;
};

class Sprite {
public:
	Sprite() : texture()/*,texture()*/, position(0) {}

	Sprite(const char* path, Vec2 position = Vec2(), Vec2 size = Vec2(), float angle = 0, Shader shader = Shader("GLSL/core.vs", "GLSL/core.frag")) : shader(shader), angle(angle), position(0), texture() {
		this->camera = (Camera*)glfwGetWindowUserPointer(window);
		init_image();
		LoadImg(path, texture);
		this->size = Vec2(size.x, size.y);
		this->position = Vec2(position.x - this->size.x / 2, position.y - this->size.y / 2);
	}
	void SetPosition(const Vec2& position);

	void LoadImg(const char* path, Texture& object);

	void Draw();

	void init_image();

	Vec2 Size() const {
		return this->size;
	}

	Vec2 get_position() const {
		return this->position;
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
	Vec2 position;
	Vec2 size;
	float angle;
	Alloc<float> _Vertices;
};

static Alloc<size_t> ind;

template <typename _Square, typename _Matrix, typename _ReturnType = Vec2>
constexpr _ReturnType Raycast(_Square& grid, const _Matrix& direction, size_t gridSize) {
	Vec2 best(-1, -1);
	for (int i = 0; i < ind.current(); i++) {
		const Mat2x4 corners = grid.get_corners(ind[i]);
		if (direction[0].x < direction[1].x) {
			const Vec2 inter = IntersectionPoint(direction[0], direction[1], corners[0], corners[3]);
			if (inter.x != -1) {
				if (best.x != -1) {
					if (ManhattanDistance(direction[0], inter) < ManhattanDistance(direction[0], best)) {
						best = inter;
					}
				}
				else {
					best = inter;
				}
			}
		}
		else {
			const Vec2 inter = IntersectionPoint(direction[0], direction[1], corners[1], corners[2]);
			if (inter.x != -1) {
				if (best.x != -1) {
					if (ManhattanDistance(direction[0], inter) < ManhattanDistance(direction[0], best)) {
						best = inter;
					}
				}
				else {
					best = inter;
				}
			}
		}
		if (direction[1].y < direction[0].y) {
			const Vec2 inter = IntersectionPoint(direction[0], direction[1], corners[2], corners[3]);
			if (inter.x != -1) {
				if (best.x != -1) {
					if (ManhattanDistance(direction[0], inter) < ManhattanDistance(direction[0], best)) {
						best = inter;
					}
				}
				else {
					best = inter;
				}
			}
		}
		else {
			const Vec2 inter = IntersectionPoint(direction[0], direction[1], corners[0], corners[1]);
			if (inter.x != -1) {
				if (best.x != -1) {
					if (ManhattanDistance(direction[0], inter) < ManhattanDistance(direction[0], best)) {
						best = inter;
					}
				}
				else {
					best = inter;
				}
			}
		}
	}
	return best;
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