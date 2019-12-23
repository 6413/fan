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

struct VerCol {
	VerCol() : vertices(), colors(){}
	VerCol(const std::vector<float>& _Vec, const Color& _Col) {
		for (int i = 0; i < _Vec.size(); i++) {
			this->vertices.push_back(_Vec[i]);
		}
		this->colors = _Col;
	}
	std::vector<float> vertices;
	Color colors;
};

std::vector<Vec2> LoadMap(float blockSize);

class ShapeDefault {
public:
	void draw() {
		this->shader.Use();
		Mat4x4 view(1);

		view = camera->GetViewMatrix(Translate(view, Vec3(windowSize.x / 2, windowSize.y / 2, -700.0f)));

		Mat4x4 projection(1);
		projection = Ortho(windowSize.x / 2, windowSize.x + windowSize.x * 0.5, windowSize.y + windowSize.y * 0.5f, windowSize.y / 2, 0.1f, 1000.0f);
		int projLoc = glGetUniformLocation(shader.ID, "projection");
		int viewLoc = glGetUniformLocation(shader.ID, "view");
		glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view.vec[0].x);
		glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection.vec[0].x);

		glBindVertexArray(this->object.VAO);
		glDrawArrays(this->type, 0, this->points);
		glBindVertexArray(0);
	}

	Color GetColor(size_t _Index) const {
		return Color(this->data[_Index * this->pointSize + 2],
			this->data[_Index * this->pointSize + 3],
			this->data[_Index * this->pointSize + 4],
			this->data[_Index * this->pointSize + 5]
		);
	}

	void SetColor(size_t _Index, char _operator, int value) {
		for (int i = 0; i < pointSize * COLORSIZE + COLORSIZE; i += (COORDSIZE + COLORSIZE)) {
			switch (_operator) {
			case '^': {
				//this->data[i + _Index] 
				this->data[(i + _Index * (24)) + 2] = (int)data[(i + _Index * (24)) + 2] ^ value;
				this->data[(i + _Index * (24)) + 3] = (int)data[(i + _Index * (24)) + 3] ^ value;
				this->data[(i + _Index * (24)) + 4] = (int)data[(i + _Index * (24)) + 4] ^ value;
				break;
			}
			}

		}
		writeToGpu(data);
	}

	void SaveMap(float blockSize) {
		std::ofstream file;
		file.open("data");
		std::vector<int> blocks;
		for (int i = 0; i < objectAmount; i++) {
			if (wall[i]) {
				blocks.push_back(i);
			}
		}

		for (int i = 0; i < blocks.size(); i++) {
			file << blocks[i] << std::endl;
		}

		file.close();
	}

	bool wall[14 * 14];
protected:
	void init(Camera* camera) {
		this->camera = camera;
		glGenVertexArrays(1, &this->object.VAO);
		glGenBuffers(1, &this->object.VBO);
		glBindVertexArray(this->object.VAO);

		if (objectAmount) {
			merge();
		}
		
		glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(data[0]) * data.size(), data.data(), GL_STATIC_DRAW);

		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(2 * sizeof(float)));
		glEnableVertexAttribArray(1);


		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

	void writeToGpu(const std::vector<float>& v) {
		glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(v[0]) * v.size(), v.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void merge() {
		size_t at = verCol.size() - 1;
		int vertIndex = 0;
		for (int i = 0; i < this->pointSize; i++) {
			for (int vertices = 0; vertices < 2; vertices++) {
				data.push_back(verCol[at].vertices[vertIndex]);
				verticeMemory.push_back(vertIndex + objectAmount * verCol[at].vertices.size());
				vertIndex++;
			}
			for (int color = 0; color < COLORSIZE; color++) {
				data.push_back(verCol[at].colors[color]);
			}
		}
	}

	void edit(size_t _Where, const std::vector<float>& v, const Color& color = Color(-1, 0, 0, 0)) {
		if (color.r != -1) {
			
		}
		else {
			for (int i = 0; i < v.size(); i++) {
				data[verticeMemory[i + _Where * v.size()]] = v[i];
			}
		}
		writeToGpu(data);
	}
	Vec2 size;
	size_t type;
	Object object;
	Camera* camera;
	Shader shader = Shader("GLSL/shapes.vs", "GLSL/shapes.frag");
	std::vector<VerCol> verCol;
	std::vector<float> data;
	std::vector<size_t> verticeMemory;
	size_t points;
	size_t pointSize;
	size_t objectAmount;
};

class Line : public ShapeDefault {
public:
	Line(Camera* camera, const Mat2x2& position, const Color& color) {
		this->type = GL_LINES;
		this->points = 2;
		this->pointSize = this->points;
		std::vector<float> _Temp;
		for (int i = 0; i < pointSize * 2; i++) {
			_Temp.push_back(position.vec[!!(i & 2)][i & 1]);
		}
		this->verCol.push_back(VerCol(_Temp, color));
		this->init(camera);
		objectAmount++;
	}
	void push_back(const Mat2x2& position, const Color& color = Color(-1, 0, 0, 0)) {
		this->verCol.push_back(VerCol(std::vector<float>{position.vec[0].x, position.vec[0].y, position.vec[1].x, position.vec[1].y} , color.r != -1 ? color : GetColor(0)));
		merge();
		this->writeToGpu(data);
		this->points += pointSize;
		objectAmount++;
	}

	Vec2 getPosition(size_t _Index) const {
		return Vec2(verCol[_Index].vertices[2] - verCol[_Index].vertices[0], verCol[_Index].vertices[3] - verCol[_Index].vertices[1]);
	}

	void setPosition(size_t _Index, const Mat2x2& position) {
		std::vector<float> vec;

		for (int i = 0; i < position.size() * 0.5; i++) {
			for (int j = 0; j < position.size() * 0.5; j++) {
				vec.push_back(position.vec[i][j]);
			}
		}
		edit(_Index, vec);
	}
};

class Triangle {
public:
	void setPos(Vec2 x) {}
};

class Square : public ShapeDefault {
public:
	Square(Camera* camera, const Vec2& position, const Vec2& size, const Color& color) {
		this->size = size;
		points = 4;
		pointSize = points;
		type = GL_QUADS;
		verCol.push_back(VerCol(std::vector<float>{
			position.x - (size.x / 2), position.y - (size.y / 2),
			position.x + (size.x / 2), position.y - (size.y / 2),
			position.x + (size.x / 2), position.y + (size.y / 2),
			position.x - (size.x / 2), position.y + (size.y / 2)

		}, color));
		objectAmount++;
		this->init(camera);
	}

	Square(Camera* camera, const Vec2& size, const Color& color) {
		this->size = size;
		type = GL_QUADS;
		points = 4;
		pointSize = points;
		objectAmount = 0;
		init(camera);
	}

	void push_back(const Vec2& position, const Color& color = Color(-1, 0, 0, 0)) {
		this->verCol.push_back(VerCol(std::vector<float> {
			position.x - (size.x / 2), position.y - (size.y / 2),
			position.x + (size.x / 2), position.y - (size.y / 2),
			position.x + (size.x / 2), position.y + (size.y / 2),
			position.x - (size.x / 2), position.y + (size.y / 2)
		}, color.r != -1 ? color : GetColor(0)));
		merge();
		this->writeToGpu(data);
		this->points += pointSize;
		objectAmount++;
	}

	Vec2 getPosition(size_t _Index) const {
		return Vec2(verCol[_Index].vertices[2] - verCol[_Index].vertices[0], verCol[_Index].vertices[3] - verCol[_Index].vertices[1]);
	}

	void setPosition(size_t _Index, const Vec2& position) {
		edit(_Index, std::vector<float> {
				position.x - (size.x / 2), position.y - (size.y / 2),
				position.x + (size.x / 2), position.y - (size.y / 2),
				position.x + (size.x / 2), position.y + (size.y / 2),
				position.x - (size.x / 2), position.y + (size.y / 2)
		});
	}
};

//template <typename shape_t>
//class Shape : public shape_t {
//	Shape(Camera* camera, const Mat2x2& position, const Vec2 size, const Color& color) : shape_t(camera, position, size, color) {}
//};

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