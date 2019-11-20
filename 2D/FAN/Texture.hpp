#pragma once
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "../SOIL2/SOIL2.h"
#include <vector>
#include "../SOIL2/stb_image.h"

#include "Input.hpp"
#include "Settings.hpp"
#include "Math.hpp"
#include "Shader.h"
#include "Camera.hpp"
#include "Color.hpp"

#ifdef _MSC_VER
#pragma warning (disable : 26495)
#endif

#define BackgroundSize 1500

class Object;
class Texture;
class Sprite;
class Main;
class Entity;
enum class GroupId;

using namespace Settings;
using namespace WindowNamespace;
using namespace CursorNamespace;

#define LINEVERT 4
#define TRIANGLEVERT 6
#define SQUAREVERT 8
#define GRASSHEIGHT 100
#define PLAYERSIZE Vec2(6, 6)

constexpr float triangle_vertices[TRIANGLEVERT] = {
	-0.433, 0.25,
	 0.433, 0.25,
	 0.0,  -0.5f
};

constexpr float square_vertices[SQUAREVERT] = {
	0.5f, 0.5f,
	0.5f, -0.5f,
   -0.5f, -0.5f,
   -0.5f, 0.5f
};

void LoadImg(const char* path, Object& object, Texture& texture);

class Object {
public:
	unsigned int texture;
	int width, height;
	unsigned int VBO, VAO, EBO;
	Object() : texture(0), width(0), height(0), VBO(0), VAO(0), EBO(0) { }
};

class Texture {
public:
	Texture() : vertices{ 0 } { }
	void IntializeImage(Texture& texture);
	float vertices[30];
};

class Physics {
protected:
	const float gravity = 3000;
};

class Sprite {
public:
	Sprite() : object(), texture(), position(0) {}

	Sprite(const Sprite& info);

	Sprite(Camera* camera, const char* path, Vec2 size = Vec2(), Vec2 position = Vec2(), float angle = 0);

	void SetPosition(const Vec2& position);

	void Draw();

	Texture GetTexture() const {
		return this->texture;
	}

	Object GetObject() const {
		return this->object;
	}

protected:
	Camera* camera;
	Shader shader;
	Object object;
	Texture texture;
	Vec2 position;
	Vec2 size;
	float angle;
};

#define MIDDLE 0xDEADBEEF

class Shape {
public:
	Shape() {}

	Shape(Camera* camera, const Vec2& position, const Vec2& pixelSize, const Color& color, std::vector<float> vec);

	~Shape();

	void Draw(Entity& player);

	void SetColor(Color color);

	void Rotatef(float angle, Vec2 point = Vec2(MIDDLE));

	Vec2 Size() const;

	Color GetColor(bool div) {
		if (div) {
			return this->color / 0xff;
		}
		return this->color;
	}

protected:
	std::vector<float> vertices;
	Object object;
	Shader shader;
	Camera* camera;
	Vec2 position;
	Vec2 size;
	Color color;
	Vec3 rotatexyz;
	float angle;
	long long type;
	uint64_t points;
	uint64_t vertSize;
};

class Triangle : public Shape {
public:
	Triangle(Camera* camera, const Vec2& position, const Vec2& size, const Color& color) :
		Shape(camera, position, size, color, std::vector<float> {
		position.x - (size.x / 2), position.y + (size.y / 2),
		position.x + (size.x / 2), position.y + (size.y / 2),
		position.x, position.y - (size.y / 2)
	}) {
		this->size = size;
		vertSize = TRIANGLEVERT;
		type = GL_TRIANGLES;
		points = 3;
	}
	void Add(const Vec2& position, Vec2 size = Vec2());

	void SetPosition(size_t _Where, const Vec2& position);
};

class Square : public Shape {
public:
	Square(Camera* camera, const Vec2& position, const Vec2& size, const Color& color) :
		Shape(camera, position, size, color, std::vector<float> {
		position.x - (size.x / 2), position.y - (size.y / 2),
		position.x + (size.x / 2), position.y - (size.y / 2),
		position.x + (size.x / 2), position.y + (size.y / 2),
		position.x - (size.x / 2), position.y + (size.y / 2)
	}) {
		this->size = size;
		vertSize = SQUAREVERT;
		type = GL_QUADS;
		points = 4;
	}
	
	void Add(const Vec2& position, const Vec2& size);

	void SetPosition(size_t _Where, const Vec2& position);
};

class Line : public Shape {
public:
	Line() {};
	/*Line(Camera* camera, const Vec2& size, const Color& color = Color(255, 0, 0, 255)) : Shape(camera, Vec2(), Vec2(), color, std::vector<float>{}) {
		this->size = size;
		this->vertSize = LINEVERT;
		type = GL_LINES;
		points = 0;
	};*/
	Line(Camera* camera, const Mat2x2& begin_end, const Color& color) : 
		Shape(camera, Vec2(), Vec2(), color, std::vector<float> {
		begin_end.vec[0].x, begin_end.vec[0].y,
		begin_end.vec[1].x, begin_end.vec[1].y

	}){
		this->size = Vec2(abs(begin_end.vec[0].x - begin_end.vec[1].x), abs(begin_end.vec[0].y - begin_end.vec[1].y));
		this->vertSize = LINEVERT;
		this->type = GL_LINES;
		this->points = 2;
	}

	void Add(const Mat2x2& begin_end);

	void SetPosition(size_t _Where, const Mat2x2& begin_end);
	void SetPosition(size_t _Where, const Vec2& position);
};

enum class GroupId {
	NotAssigned = -1,
	LocalPlayer = 0,
	Enemy = 1
};

class Entity : public Sprite, public Physics {
public:
	Entity() : groupId(GroupId::NotAssigned) {}

	Entity(const char* path, GroupId _groupId);

	Entity(const Object& object) {
		this->object = object;
	}

	Entity(Camera* camera, const char* path, const Vec2& size, Vec2 position = Vec2(), GroupId _groupId = GroupId::LocalPlayer) : 
		velocity(0), groupId(_groupId), Sprite(camera, path, size, position) {
		this->objects.push_back(this->object);
		this->textures.push_back(this->texture);
	}

	GroupId GetGroupId();

	void SetGroupId(GroupId groupId);
	void Move();

	Vec2 GetPosition() const {
		return this->position;
	}

	void SetImage(const Sprite& sprite) {
		this->objects.push_back(sprite.GetObject());
		this->textures.push_back(sprite.GetTexture());
	}

private:
	std::vector<Object> objects;
	std::vector<Texture> textures;
	GroupId groupId;
	Vec2 velocity;
	float health;
	const float movementSpeed = 2000;
	const float friction = 5;
	const float jumpForce = 10000;
};

class Main {
public:
	std::vector<Entity> entity;
	Shader shader = Shader("GLSL/core.vs", "GLSL/core.frag");
	Camera camera;
	GLFWwindow* window;
	Main() : camera(Vec3()) {}
};