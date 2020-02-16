#pragma once

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

//#include <ft2build.h>
//#include FT_FREETYPE_H  

#include "Alloc.hpp"
#include "Camera.hpp"
#include "Input.hpp"
#include "Math.hpp"
#include "Settings.hpp"
#include "Shader.h"

#include <map>
#include <vector>
#include <chrono>

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
	Texture();

	unsigned int texture;
	int width, height;
	unsigned int VBO, VAO, EBO;
};

class DefaultShape {
public:
	Color get_color(std::size_t _Index) const;
	auto& get_color_ptr() const;
	void set_color(std::size_t _Index, const Color& color, bool queue = false);

	void draw();

	void break_queue();

protected:
	void init();

	void write(bool _EditVertices, bool _EditColor);

	Texture _VerticeBuffer;
	Texture _ColorBuffer;
	Texture _ShapeBuffer;
	Shader _Shader;
	Camera* _Camera;
	Alloc<float> _Vertices;
	Alloc<float> _Colors;
	unsigned int _Mode;
	int _Points;
	std::size_t _PointSize;
};

class Line : public DefaultShape {
public:
	Line();
	Line(const Mat2x2& _M, const Color& color);

	Mat2x2 get_position(std::size_t _Index) const;
	void set_position(std::size_t _Index, const Mat2x2& _M, bool _Queue = false);

	void push_back(const Mat2x2& _M, Color _Color = Color(-1, -1, -1, -1), bool _Queue = false);

	Vec2 get_length(std::size_t _Index) const;

private:
	Alloc<Vec2> _Length;
};

class Triangle : public DefaultShape {
public:
	Triangle();
	Triangle(const Vec2& _Position, const Vec2& _Length, const Color& _Color);
	~Triangle() {}

	void set_position(std::size_t _Index, const Vec2& _Position);
	Vec2 get_position(std::size_t _Index) const;

	void push_back(const Vec2 _Position, Vec2 _Length = Vec2(), Color _Color = Color(-1, -1, -1, -1));

private:
	Alloc<Vec2> _Position;
	Alloc<Vec2> _Length;
};
static std::size_t counter = 0;
class Square : public DefaultShape {
public:
	Square();
	Square(const Vec2& _Position, const Vec2& _Length, const Color& color);
	Square(std::size_t _Reserve, const Vec2& _Position, const Vec2& _Length, const Color& color);

	std::size_t amount() const;

	void free_to_max();

	void erase(std::size_t _Index);

	Vec2 get_length(std::size_t _Index) const;
	Mat2x4 get_corners(std::size_t _Index) const;

	Vec2 get_position(std::size_t _Index) const;
	void set_position(std::size_t _Index, const Vec2& _Position, bool _Queue = false);

	void push_back(const Vec2& _Position, Vec2 _Length = Vec2(), Color color = Color(-1, -1, -1, -1), bool _Queue = false);

	void rotate(std::size_t _Index, double _Angle, bool queue = false);

private:
	Alloc<Vec2> _Position;
	Alloc<Vec2> _Length;
};

class Circle : public DefaultShape {
public:
	Circle(std::size_t _Number_Of_Points, float _Radius);
	Circle(const Vec2& _Position, float _Radius, std::size_t _Number_Of_Points, const Color& _Color);

	void set_position(std::size_t _Index, const Vec2& _Position);

	void push_back(Vec2 _Position, float _Radius, Color _Color, bool _Queue = false);

private:
	Alloc<Vec2> _Position;
	Alloc<float> _Radius;
	const float _Double_Pi = 2.0f * PI;
};

class Sprite {
public:
	Sprite();
	Sprite(const char* path, Vec2 position = Vec2(), Vec2 size = Vec2(), float angle = 0, Shader shader = Shader("GLSL/core.vs", "GLSL/core.frag"));

	void draw();
	void init_image();
	void load_image(const char* path, Texture& object);

	Texture& get_texture();
	Vec2 get_size() const;

	Vec2 get_position() const;
	void set_position(const Vec2& position);

protected:
	Camera* camera;
	Shader shader;
	Texture texture;
	Vec2 position;
	Vec2 size;
	float angle;
	Alloc<float> _Vertices;
};

using namespace std::chrono;

class Timer {
public:
	Timer();
	Timer(const decltype(high_resolution_clock::now())& timer, std::size_t time);

	void start(int time);
	void restart();

	bool finished();
	std::size_t passed();

private:
	decltype(high_resolution_clock::now()) timer;
	std::size_t time;
};

struct Particle {
	float life_time;
	Timer time;
	bool display;
	Vec2 particle_speed;
};

class Particles {
public:
	std::size_t particles_per_second = 1000;

	Particles(std::size_t particles_amount, Vec2 particle_size, Vec2 particle_speed, float life_time, Color begin, Color end);

	void add(Vec2 position);

	void draw();

private:
	int64_t particleIndex;
	Square particles;
	Alloc<Particle> particle;
	Color begin;
	Color end;
	float life_time;
};

extern Alloc<std::size_t> blocks;

Vec2 Raycast(Square& grid, const Mat2x2& direction, std::size_t gridSize);

#ifdef FT_FREETYPE_H
struct Character {
	GLuint TextureID;   // ID handle of the glyph texture
	__Vec2<int> Size;    // Size of glyph
	__Vec2<int> Bearing;  // Offset from baseline to left/top of glyph
	GLuint Advance;    // Horizontal offset to advance to next glyph
};

class TextRenderer {
public:
	TextRenderer() : shader(Shader("GLSL/text.vs", "GLSL/text.frag")) {
		shader.Use();
		Mat4x4 projection = Ortho(0, windowSize.x, windowSize.y, 0);
		glUniformMatrix4fv(glGetUniformLocation(shader.ID, "projection"), 1, GL_FALSE, &projection.vec[0][0]);
		// FreeType
		FT_Library ft;
		// All functions return a value different than 0 whenever an error occurred
		if (FT_Init_FreeType(&ft))
			std::cout << "ERROR::FREETYPE: Could not init FreeType Library" << std::endl;

		// Load font as face
		FT_Face face;
		if (FT_New_Face(ft, "fonts/arial.ttf", 0, &face))
			std::cout << "ERROR::FREETYPE: Failed to load font" << std::endl;

		// Set size to load glyphs as
		FT_Set_Pixel_Sizes(face, 0, 48);

		// Disable byte-alignment restriction
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		// 246 = � in unicode
		for (GLubyte c = 0; c < 247; c++)
		{
			// Load character glyph 
			if (FT_Load_Char(face, c, FT_LOAD_RENDER))
			{
				std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
				continue;
			}
			// Generate texture
			GLuint texture;
			glGenTextures(1, &texture);
			glBindTexture(GL_TEXTURE_2D, texture);
			glTexImage2D(
				GL_TEXTURE_2D,
				0,
				GL_RED,
				face->glyph->bitmap.width,
				face->glyph->bitmap.rows,
				0,
				GL_RED,
				GL_UNSIGNED_BYTE,
				face->glyph->bitmap.buffer
			);
			// Set texture options
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			// Now store character for later use
			Character character = {
				texture,
				__Vec2<int>(face->glyph->bitmap.width, face->glyph->bitmap.rows),
				__Vec2<int>(face->glyph->bitmap_left, face->glyph->bitmap_top),
				face->glyph->advance.x
			};
			Characters.insert(std::pair<GLchar, Character>(c, character));
		}
		glBindTexture(GL_TEXTURE_2D, 0);
		// Destroy FreeType once we're finished
		FT_Done_Face(face);
		FT_Done_FreeType(ft);


		// Configure VAO/VBO for texture quads
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}
	void render(std::string text, Vec2 position, float scale, const Color& color) {
		Mat4x4 projection(1);
		projection = Ortho(0, windowSize.x, windowSize.y, 0);
		//printf("%d\n", windowSize.x);
		glUniformMatrix4fv(glGetUniformLocation(shader.ID, "projection"), 1, GL_FALSE, &projection.vec[0][0]);
		shader.Use();
		glUniform4f(glGetUniformLocation(shader.ID, "textColor"), color.r, color.g, color.b, color.a);
		glActiveTexture(GL_TEXTURE0);
		glBindVertexArray(VAO);
		// Iterate through all characters
		float biggest = 0;
		for (std::string::const_iterator c = text.begin(); c != text.end(); c++) {
			biggest = std::max(Characters[*c].Size.y * scale, biggest);
		}
		position.y += biggest;
		std::string::const_iterator c;

		Alloc<float(*)[4]> vert;
		for (c = text.begin(); c != text.end(); c++)
		{
			Character ch = Characters[*c];



			GLfloat w = ch.Size.x * scale;
			GLfloat h = ch.Size.y * scale;

			GLfloat xpos = position.x + ch.Bearing.x * scale;
			GLfloat ypos = position.y + (ch.Size.y - ch.Bearing.y) * scale;

			// Update VBO for each character
			GLfloat vertices[6][4] = {
				{ xpos,     ypos - h,   0.0, 0.0 },
				{ xpos,     ypos,       0.0, 1.0 },
				{ xpos + w, ypos,       1.0, 1.0 },

				{ xpos,     ypos - h,   0.0, 0.0 },
				{ xpos + w, ypos,       1.0, 1.0 },
				{ xpos + w, ypos - h,   1.0, 0.0 }
			};
			vert.push_back(vertices);

			// Render glyph texture over quad
			glBindTexture(GL_TEXTURE_2D, ch.TextureID);
			// Update content of VBO memory
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices); // Be sure to use glBufferSubData and not glBufferData

			glBindBuffer(GL_ARRAY_BUFFER, 0);
			// Render quad
			glDrawArrays(GL_TRIANGLES, 0, 6);
			// Now advance cursors for next glyph (note that advance is number of 1/64 pixels)
			position.x += (ch.Advance >> 6)* scale; // Bitshift by 6 to get value in pixels (2^6 = 64 (divide amount of 1/64th pixels by 64 to get amount of pixels))
		}
		glBindVertexArray(0);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
private:
	std::map<GLchar, Character> Characters;
	Shader shader;
	unsigned int VAO, VBO;
};
#endif

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

//class Main {
//public:
//	Shader shader = Shader("GLSL/core.vs", "GLSL/core.frag");
//	Camera camera;
//	Main() : camera(Vec3()) {}
//};