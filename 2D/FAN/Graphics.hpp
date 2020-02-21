#pragma once

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

//#include <ft2build.h>
//#include FT_FREETYPE_H  

#include "Alloc.hpp"
#include "Input.hpp"
#include "Math.hpp"
#include "Settings.hpp"
#include "Shader.h"
#include "Vectors.hpp"
#include "Time.hpp"

#include <map>
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
	Texture();

	unsigned int texture;
	int width, height;
	unsigned int VBO, VAO, EBO;
};

namespace BMP_Offsets {
	constexpr::ptrdiff_t PIXELDATA = 0xA;
	constexpr::ptrdiff_t WIDTH = 0x12;
	constexpr::ptrdiff_t HEIGHT = 0x16;
}

unsigned char* LoadBMP(const char* path, Texture& texture);

class Camera {
public:
	vec3 position;

	Camera(vec3 position = vec3(0.0f, 0.0f, 0.0f), vec3 up = vec3(0.0f, 1.0f, 0.0f), float yaw = -90.0f, float pitch = 0.0f);

	constexpr matrix<4, 4> GetViewMatrix(matrix<4, 4> m) {
		return m * LookAt(this->position, (this->position + Round(this->front)), Round(this->up));
	}

	constexpr vec3 GetPosition() const {
		return this->position;
	}

	constexpr vec3 GetFront() const {
		return this->front;
	}

	GLfloat yaw;
	GLfloat pitch;
private:
	vec3 front;
	vec3 up;
	vec3 right;
	vec3 worldUp;

	void updateCameraVectors() {
		front.x = cos(Radians(this->yaw)) * cos(Radians(this->pitch));
		front.y = sin(Radians(this->pitch));
		front.z = sin(Radians(this->yaw)) * cos(Radians(this->pitch));
		this->front = Normalize(front);
		this->right = Normalize(Cross(this->front, this->worldUp));
		this->up = Normalize(Cross(this->right, this->front));
	}
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
	std::vector<float> _Vertices;
	std::vector<float> _Colors;
	unsigned int _Mode;
	int _Points;
	std::size_t _PointSize;
};

class Line : public DefaultShape {
public:
	Line();
	Line(const mat2x2& _M, const Color& color);

	mat2x2 get_position(std::size_t _Index) const;
	void set_position(std::size_t _Index, const mat2x2& _M, bool _Queue = false);

	void push_back(const mat2x2& _M, Color _Color = Color(-1, -1, -1, -1), bool _Queue = false);

	vec2 get_length(std::size_t _Index) const;

private:
	std::vector<vec2> _Length;
};

class Triangle : public DefaultShape {
public:
	Triangle();
	Triangle(const vec2& _Position, const vec2& _Length, const Color& _Color);
	~Triangle() {}

	void set_position(std::size_t _Index, const vec2& _Position);
	vec2 get_position(std::size_t _Index) const;

	void push_back(const vec2 _Position, vec2 _Length = vec2(), Color _Color = Color(-1, -1, -1, -1));

private:
	std::vector<vec2> _Position;
	std::vector<vec2> _Length;
};
static std::size_t counter = 0;
class Square : public DefaultShape {
public:
	Square();
	Square(const vec2& _Position, const vec2& _Length, const Color& color);
	Square(std::size_t _Reserve, const vec2& _Position, const vec2& _Length, const Color& color);

	std::size_t amount() const;

	void free_to_max();

	void erase(std::size_t _Index);

	vec2 get_length(std::size_t _Index) const;
	mat2x4 get_corners(std::size_t _Index) const;

	vec2 get_position(std::size_t _Index) const;
	void set_position(std::size_t _Index, const vec2& _Position, bool _Queue = false);

	void push_back(const vec2& _Position, vec2 _Length = vec2(), Color color = Color(-1, -1, -1, -1), bool _Queue = false);

	void rotate(std::size_t _Index, double _Angle, bool queue = false);

private:
	std::vector<vec2> _Position;
	std::vector<vec2> _Length;
};

class Circle : public DefaultShape {
public:
	Circle(std::size_t _Number_Of_Points, float _Radius);
	Circle(const vec2& _Position, float _Radius, std::size_t _Number_Of_Points, const Color& _Color);

	void set_position(std::size_t _Index, const vec2& _Position);

	void push_back(vec2 _Position, float _Radius, Color _Color, bool _Queue = false);

private:
	std::vector<vec2> _Position;
	std::vector<float> _Radius;
	const float _Double_Pi = 2.0f * PI;
};

class Sprite {
public:
	Sprite();
	Sprite(const char* path, vec2 position = vec2(), vec2 size = vec2(), float angle = 0, Shader shader = Shader("GLSL/core.vs", "GLSL/core.frag"));

	void draw();
	void init_image();
	void load_image(const char* path, Texture& object);

	Texture& get_texture();
	vec2 get_size() const;

	vec2 get_position() const;
	void set_position(const vec2& position);

protected:
	Camera* camera;
	Shader shader;
	Texture texture;
	vec2 position;
	vec2 size;
	float angle;
	std::vector<float> _Vertices;
};

struct Particle {
	float life_time;
	Timer time;
	bool display;
	vec2 particle_speed;
};

class Particles {
public:
	std::size_t particles_per_second = 1000;

	Particles(std::size_t particles_amount, vec2 particle_size, vec2 particle_speed, float life_time, Color begin, Color end);

	void add(vec2 position);

	void draw();

private:
	int64_t particleIndex;
	Square particles;
	std::vector<Particle> particle;
	Color begin;
	Color end;
	float life_time;
};

extern std::vector<std::size_t> blocks;

vec2 Raycast(Square& grid, const mat2x2& direction);

#ifdef FT_FREETYPE_H
struct Character {
	GLuint TextureID;   // ID handle of the glyph texture
	_vec2<int> Size;    // Size of glyph
	_vec2<int> Bearing;  // Offset from baseline to left/top of glyph
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
				_vec2<int>(face->glyph->bitmap.width, face->glyph->bitmap.rows),
				_vec2<int>(face->glyph->bitmap_left, face->glyph->bitmap_top),
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
	void render(std::string text, vec2 position, float scale, const Color& color) {
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

		std::vector<float(*)[4]> vert;
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
//	Entity(Camera* camera, const char* path, const vec2& size, vec2 position = vec2(), GroupId _groupId = GroupId::LocalPlayer) : 
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
//	vec2 GetPosition() const {
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
//	vec2 velocity;
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
//	Main() : camera(vec3()) {}
//};