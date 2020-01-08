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


#ifdef _MSC_VER
#pragma warning (disable : 26495)
#endif

#define BackgroundSize 1500
#define COLORSIZE 4
#define COORDSIZE 2

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

struct lineVertices {
	lineVertices() : v(), color() {}
	lineVertices(float* vec, size_t vecSize, float* c) {
		for (size_t i = 0; i < vecSize; i++) {
			v[i] = vec[i];
		}
		for (int i = 0; i < 4; i++) {
			color[i] = c[i];
		}
	}
	float v[4];
	float color[4];
};

#include <vector>

template <typename Shape>
class DefaultShape {
public:

	void draw() {
		shader.Use();
		Mat4x4 view(1);
		Mat4x4 projection(1);

		view = camera->GetViewMatrix(Translate(view, Vec3(windowSize.x / 2, windowSize.y / 2, -700.0f)));
		projection = Ortho(windowSize.x / 2, windowSize.x + windowSize.x * 0.5, windowSize.y + windowSize.y * 0.5f, windowSize.y / 2, 0.1f, 1000.0f);
		int projLoc = glGetUniformLocation(shader.ID, "projection");
		int viewLoc = glGetUniformLocation(shader.ID, "view");
		glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection.vec[0].x);
		glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view.vec[0].x);
		glBindVertexArray(shapeBuffer.VAO);
		glDrawArrays(mode, 0, points);
		glBindVertexArray(0);
	}

	~DefaultShape() {
		//vertices.free();
		//colors.free();
	}
protected:
	void init(Camera* camera) {
		this->camera = camera;
		this->shader = Shader("GLSL/shapes.vs", "GLSL/shapes.frag");
		glGenVertexArrays(1, &shapeBuffer.VAO);
		glBindVertexArray(shapeBuffer.VAO);

		glGenBuffers(1, &verticeBuffer.VBO);
		glBindBuffer(GL_ARRAY_BUFFER, verticeBuffer.VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices[0]) * vertices.current(), vertices.data(), GL_STATIC_DRAW);
		// (sizeof(float) * 3
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);


		glGenBuffers(1, &colorBuffer.VBO);
		glBindBuffer(GL_ARRAY_BUFFER, colorBuffer.VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(colors[0]) * colors.size(), colors.data(), GL_STATIC_DRAW);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(vertices[0])* vertices.current() * sizeof(float), (void*)(sizeof(vertices[0]) * vertices.current() * sizeof(float)));
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

	void write() {
	/*	glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices[0]) * vertices.current(), vertices.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);*/
	}

	Object verticeBuffer;
	Object colorBuffer;
	Object shapeBuffer;
	Shader shader;
	Camera* camera;
	Alloc<float> vertices;
	Alloc<float> colors;
	size_t mode;
	size_t points;
};

class Line : public DefaultShape<lineVertices> {
public:
	template <typename Cam, typename Mat, typename Col>
	constexpr Line(Cam camera, const Mat& m, const Col& color) {
		vertices.push_back(m[0].x);
		vertices.push_back(m[0].y);
		vertices.push_back(m[1].x);
		vertices.push_back(m[1].y);
		colors.push_back(color.r);
		colors.push_back(color.g);
		colors.push_back(color.b);
		colors.push_back(color.a);
		mode = GL_LINES;
		points = 2;
		this->init(camera);
		write();
	}

};

enum class GroupId {
	NotAssigned = -1,
	LocalPlayer = 0,
	Enemy = 1
};

class Entity : public Sprite {
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
	float gravity;
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