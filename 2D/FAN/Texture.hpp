#pragma once
#define GLFW_INCLUDE_VULKAN
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>

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
class Square;
struct ImageData;
enum class GroupId;

using namespace Settings;
using namespace WindowNamespace;
using namespace CursorNamespace;

#define COLORSIZE 4
#define LINEVERT 4
#define TRIANGLEVERT 6
#define SQUAREVERT 8
#define BLOCKAMOUNT 4
#define BLOCKSIZE 64
#define PLAYERSIZE Vec2(90, 90)

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

class Collision {
public:
	void AddCollidable(const Vec2& position, size_t _id) {

		Vec2 object(floor((position.x - BLOCKSIZE / 2) / BLOCKSIZE), floor((position.y - BLOCKSIZE) / BLOCKSIZE));
		this->collidable.push_back(object);
		object = Vec2(floor((position.x + BLOCKSIZE / 2) / BLOCKSIZE), floor((position.y + BLOCKSIZE) / BLOCKSIZE));
		this->collidable.push_back(object);
		this->id.push_back(_id);
	}

	void AddCollidableCustom(const Vec2& position, const Vec2& size, size_t _id) {
		Vec2 object(floor((position.x - BLOCKSIZE / 2 - size.x) / BLOCKSIZE), floor((position.y - BLOCKSIZE) / BLOCKSIZE));
		this->collidable.push_back(object);
		object = Vec2(floor((position.x + BLOCKSIZE / 2 - size.x) / BLOCKSIZE), floor((position.y + BLOCKSIZE) / BLOCKSIZE));
		this->collidable.push_back(object);
		this->id.push_back(_id);
	}

	void DeleteCollidable(size_t _id) {
		this->collidable.erase(collidable.begin() + _id - 1);
		this->collidable.erase(collidable.begin() + _id);
		this->id.erase(id.begin() + _id);
	}

	bool IsColliding(const Vec2& position) const {
		Vec2 formatted(floor((position.x) / BLOCKSIZE), floor((position.y + BLOCKSIZE / 2) / BLOCKSIZE));
		for (auto i : collidable) {
			if (formatted.x == i.x && formatted.y == i.y) {
				return true;
			}
		}
		return false;
	}

	bool isCollidingCustom(const Vec2& position, const Vec2& old) const {
		Vec2 formatted(floor((position.x) / BLOCKSIZE), floor((position.y + BLOCKSIZE / 2) / BLOCKSIZE));
		Vec2 formatOld(floor((old.x) / BLOCKSIZE), floor((old.y + BLOCKSIZE / 2) / BLOCKSIZE));
		for (auto i : collidable) {
			if (formatted.x < i.x && formatted.y == i.y && formatOld.x > i.x) {
				return true;
			}
		}
		return false;
	}
private:
	std::vector<Vec2> collidable;
	std::vector<size_t> id;
};

extern Collision collision;

class Object {
public:
	unsigned int texture;
	int width, height;
	unsigned int VBO, VAO, EBO;
	Object() : texture(0), width(0), height(0), VBO(0), VAO(0), EBO(0) { }
};

struct ImageData {
	ImageData() : image(0), object() {}
	ImageData(const unsigned char* image, Object& object) : image(image), object(object) {}
	const unsigned char* image;
	Object object;
};

class Texture {
public:
	Texture() : vertices{ 0 } { }
	void IntializeImage(Texture& texture);
	float vertices[30];
};

class Physics {
protected:
	const float gravity = 100;
};

class Sprite {
public:
	Sprite() : object(), texture(), position(0) {}

	Sprite(const Sprite& info);

	Sprite(Camera* camera, const char* path, Vec2 size = Vec2(), Vec2 position = Vec2(), Shader shader = Shader("GLSL/core.vs", "GLSL/core.frag"), float angle = 0);

	void SetPosition(const Vec2& position);

	void Draw();

	Vec2 Size() const {
		return this->size;
	}

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

	Shape(Camera* camera, const Vec2& position, const Vec2& pixelSize, const Color& color, size_t _vertSize, std::vector<float> vec, 
		Shader shader = Shader("GLSL/shapes.vs", "GLSL/shapes.frag"));

	~Shape();

	void Draw();
	
	void DrawLight(Square& light);

	void SetColor(size_t _Where, Color color);

	void SetColor(size_t _Where, char _operator, int value);

	void Rotatef(float angle, Vec2 point = Vec2(MIDDLE));

	Vec2 Size() const;

	Vec2 GetPosition(size_t _Where) const;

	Color GetColor(size_t _Where) const {
		return Color(this->vertices[_Where * this->vertSize + 2], 
			this->vertices[_Where * this->vertSize + 3], 
			this->vertices[_Where * this->vertSize + 4], 
			this->vertices[_Where * this->vertSize + 5]
		);
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
		Shape(camera, position, size, color, TRIANGLEVERT, std::vector<float> {
		position.x - (size.x / 2), position.y + (size.y / 2), color.r, color.g, color.b, color.a,
		position.x + (size.x / 2), position.y + (size.y / 2), color.r, color.g, color.b, color.a,
		position.x, position.y - (size.y / 2), color.r, color.g, color.b, color.a
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
	Square() {}

	Square(Camera* camera, const Vec2& size, const Color& color) : Shape(camera, Vec2(), size, color, SQUAREVERT, std::vector<float>{}, Shader("GLSL/shapes.vs", "GLSL/shapes.frag")) {
		this->size = size;
		vertSize = SQUAREVERT;
		type = GL_QUADS;
		points = 4;
	}

	Square(Camera* camera, const Vec2& position, const Vec2& size, const Color& color, Shader shader = Shader("GLSL/shapes.vs", "GLSL/shapes.frag")) :
		Shape(camera, position, size, color, SQUAREVERT, std::vector<float> {
		position.x - (size.x / 2), position.y - (size.y / 2), color.r, color.g, color.b, color.a,
		position.x + (size.x / 2), position.y - (size.y / 2), color.r, color.g, color.b, color.a,
		position.x + (size.x / 2), position.y + (size.y / 2), color.r, color.g, color.b, color.a,
		position.x - (size.x / 2), position.y + (size.y / 2), color.r, color.g, color.b, color.a
	}, shader) {
		this->size = size;
		vertSize = SQUAREVERT;
		type = GL_QUADS;
		points = 4;
	}
	
	void Add(const Vec2& position, const Vec2& size, const Color& color = Color(-1, -1, -1, -1));

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
		Shape(camera, Vec2(), Vec2(), color, LINEVERT, std::vector<float> {
		begin_end.vec[0].x, begin_end.vec[0].y, color.r, color.g, color.b, color.a,
		begin_end.vec[1].x, begin_end.vec[1].y, color.r, color.g, color.b, color.a

	}){
		this->size = Vec2(abs(begin_end.vec[0].x - begin_end.vec[1].x), abs(begin_end.vec[0].y - begin_end.vec[1].y));
		this->vertSize = LINEVERT;
		this->type = GL_LINES;
		this->points = 2;
	}

	void Add(const Mat2x2& begin_end, const Color& color = Color(-1, -1, -1, -1));

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
	const float movementSpeed = 10000;
	const float friction = 10;
	const float jumpForce = 800;
};

class Main {
public:
	std::vector<Entity> entity;
	Shader shader = Shader("GLSL/core.vs", "GLSL/core.frag");
	Camera camera;
	GLFWwindow* window;
	Main() : camera(Vec3()) {}
};