#pragma once
//#ifndef __INTELLISENSE__ 

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

//#include <ft2build.h>
//#include FT_FREETYPE_H  

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

//#define FAN_PERFORMANCE
#define RAM_SAVER

#include <vector>
#include <array>

#include "Input.hpp"
#include "Math.hpp"
#include "Shader.h"
#include "Vectors.hpp"
#include "Time.hpp"
#include "Network.hpp"
#include <SOIL2/SOIL2.h>
#include <SOIL2/stb_image.h>

#include <map>
#include <vector>
#include <deque>

#ifdef _MSC_VER
#pragma warning (disable : 26495)
#endif

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#define BackgroundSize 1500
#define COLORSIZE 4
#define COORDSIZE 2

class Sprite;
class SquareVector;

void GetFps(bool print = true);

extern bool window_init;
constexpr auto WINDOWSIZE = _vec2<int>(1024, 1024);
extern float delta_time;
static constexpr int block_size = 50;
extern GLFWwindow* window;
constexpr auto grid_size = _vec2<int>(WINDOWSIZE.x / block_size, WINDOWSIZE.y / block_size);

typedef std::vector<std::vector<std::vector<bool>>> map_t;

struct bmp {
	unsigned char* data;
	unsigned char* image;
};

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

bmp LoadBMP(const char* path, Texture& texture);
uint64_t _2d_1d(vec2 position = cursor_position);

_vec2<int> window_position();

class Camera {
public:
	Camera(
		vec3 position = vec3(0.0f, 0.0f, 0.0f),
		vec3 up = vec3(0.0f, 1.0f, 0.0f),
		float yaw = 0,
		float pitch = 0.0f
	);

	mat4 get_view_matrix(mat4 m);
	mat4 get_view_matrix();

	vec3 get_position() const;
	void set_position(const vec3& position);

	vec3 position;
	vec3 front;

	float yaw;
	float pitch;
	vec3 right;
	vec3 up;
	vec3 velocity;
	static constexpr auto friction = 12;

	void updateCameraVectors();

private:

	vec3 worldUp;
};

class Camera2D {
public:
	Camera2D(
		vec3 position = vec3(0.0f, 0.0f, 0.0f),
		vec3 up = vec3(0.0f, 1.0f, 0.0f),
		float yaw = -90.f,
		float pitch = 0.0f
	);

	mat4 get_view_matrix(mat4 m);
	mat4 get_view_matrix();

	vec3 get_position() const;
	void set_position(const vec3& position);

	vec3 position;
	vec3 front;

	GLfloat yaw;
	GLfloat pitch;
	vec3 right;
	vec3 up;
	void updateCameraVectors();

private:

	vec3 worldUp;
};


enum class shapes {
	line,
	triangle,
	square,
	circle,
	size
};

template <shapes shape>
class vertice_handler {
public:

	~vertice_handler() {
		if (!this->vertices.empty()) {
			glDeleteVertexArrays(1, &vertice_buffer.VAO);
			glDeleteVertexArrays(1, &color_buffer.VAO);
			glDeleteVertexArrays(1, &shape_buffer.VAO);
			glDeleteBuffers(1, &vertice_buffer.VBO);
			glDeleteBuffers(1, &color_buffer.VBO);
			glDeleteBuffers(1, &shape_buffer.VBO);
		}
	}

	int draw_id = 0;

	void init(
		const std::vector<float>& l_vertices,
		const std::vector<float>& l_colors,
		bool queue = false
	);

	void write(bool _EditVertices, bool _EditColor);

	void draw(uint64_t shape_id = -1) const;

	std::vector<float> vertices;
	std::vector<float> colors;
private:
	Texture vertice_buffer;
	Texture color_buffer;
	Texture shape_buffer;
	unsigned int mode;
	int points;
	uint64_t point_size;
	Shader shader;
	Camera2D* camera;
};

static vertice_handler<shapes::line> line_handler;
static vertice_handler<shapes::square> square_handler;

static void draw_all() {
	line_handler.draw();
	square_handler.draw();
}

struct Line;
class Square;

#include <FAN/Alloc.hpp>

template <shapes shape>
class default_shape {
public:
	using shape_type = std::remove_const_t<decltype(shape)>;

	virtual void draw();

protected:
	unsigned int draw_id;
	Color color;
	vec2 position;
	vec2 size;
};

struct Line : public default_shape<shapes::line> {

	Line(const mat2x2& m, const Color& color);

	vec2 get_position() const;
	void set_position(const mat2x2& m);

	Color get_color() const;
	void set_color(const Color& color);

	vec2 get_size() const;
	void set_size(const vec2& size);
};

class Square : public default_shape<shapes::square> {
public:

	Square(const vec2& position, const vec2& size, const Color& color, bool queue = false);
	~Square();

	vec2 get_position() const;

	virtual void set_position(const vec2& position, bool queue = false);

	Color get_color() const;
	void set_color(const Color& color);

	vec2 get_size() const;
	void set_size(const vec2& size, bool queue = false);

	static void break_queue();
};

extern Shader shape_shader2d;

class basic_2dshape_vector {
public:

	basic_2dshape_vector();

	~basic_2dshape_vector();

	Color get_color(uint64_t index) const;
	void set_color(uint64_t index, const Color& color, bool queue);

	void free_queue(bool colors = true, bool matrices = true);

	void realloc_copy_data(unsigned int& buffer, uint64_t buffer_type, int size, GLenum usage, unsigned int& allocator);
	void copy_data(unsigned int& buffer, uint64_t buffer_type, int size, GLenum usage, unsigned int& allocator);
	void realloc_buffer(unsigned int& buffer, uint64_t buffer_type, int location, int size, GLenum usage, unsigned int& allocator);

	int size();
protected:

	void basic_shape_draw(unsigned int mode, uint64_t points);

	unsigned int shape_vao;
	unsigned int matrix_vbo;
	unsigned int vertex_vbo;
	unsigned int color_vbo;
	unsigned int color_allocator_vbo;
	unsigned int matrix_allocator_vbo;

	bool color_allocated;
	bool matrix_allocated;

	uint64_t shapes_size;
};

struct square_vector2d : basic_2dshape_vector {

	square_vector2d();
	square_vector2d(const vec2& position, const vec2& size, const Color& color);

	void draw();

	void erase(uint64_t first, uint64_t last = -1);

	void push_back(const vec2& position, const vec2& size, const Color& color, bool queue = false);

	vec2 get_position(uint64_t index) const;
	void set_position(uint64_t index, const vec2& position);

	vec2 get_size(uint64_t index) const;
	void set_size(uint64_t index, const vec2& size);

};

class DefaultShapeVector {
public:
	~DefaultShapeVector();

	virtual Color get_color(uint64_t _Index = 0) const;
	auto& get_color_ptr() const;
	void set_color(uint64_t _Index, const Color& color, bool queue = false);

	void draw(uint64_t first = 0, uint64_t last = 0) const;

	virtual void break_queue(bool vertices = true, bool color = true);

protected:
	virtual void init();

	void write(bool _EditVertices, bool _EditColor);

	Texture _VerticeBuffer;
	Texture _ColorBuffer;
	Texture _ShapeBuffer;
	Shader _Shader;
	Camera2D* _Camera;
#ifdef FAN_PERFORMANCE
	Alloc<float> _Vertices;
	Alloc<float> _Colors;
#else
	std::vector<float> _Vertices;
	std::vector<float> _Colors;
#endif
	unsigned int _Mode;
	int _Points;
	uint64_t _PointSize;
};

class LineVector : public DefaultShapeVector {
public:
	LineVector();
	LineVector(const mat2x2& _M, const Color& color);
	LineVector(const LineVector& line);

	mat2x2 get_position(uint64_t _Index = 0) const;
	void set_position(uint64_t _Index, const mat2x2& _M, bool _Queue = false);

	void push_back(const mat2x2& _M, Color _Color = Color(-1, -1, -1, -1), bool _Queue = false);

	vec2 get_length(uint64_t _Index = 0) const;

	uint64_t amount() const { return _Length.size(); }

private:
	std::vector<vec2> _Length;
};

class TriangleVector : public DefaultShapeVector {
public:
	TriangleVector();
	TriangleVector(const vec2& _Position, const vec2& _Length, const Color& _Color);
	~TriangleVector() {}

	void set_position(uint64_t _Index, const vec2& _Position);
	vec2 get_position(uint64_t _Index) const;

	void push_back(const vec2 _Position, vec2 _Length = vec2(), Color _Color = Color(-1, -1, -1, -1));

private:
	std::vector<vec2> _Position;
	std::vector<vec2> _Length;
};

class SquareVector : public DefaultShapeVector {
public:
	SquareVector();
	SquareVector(const SquareVector& square) {
		*this = square;
	}
	SquareVector(const vec2& _Position, const vec2& _Length, const Color& color);
	SquareVector(uint64_t _Reserve, const vec2& _Position, const vec2& _Length, const Color& color);

	uint64_t amount() const;
	bool empty() const;

	void erase(uint64_t _Index);
	void erase_all(uint64_t _Index);

	vec2 get_length(uint64_t _Index) const;
	mat2x4 get_corners(uint64_t _Index) const;

	vec2 get_position(uint64_t _Index = 0) const;
	void set_position(uint64_t _Index, const vec2& _Position, bool _Queue = false);

	vec2 get_size(uint64_t _Index = 0) const;
	void set_size(uint64_t _Index, const vec2& _Size, bool _Queue = false);

	void push_back(const SquareVector& square);
	void push_back(const vec2& _Position, vec2 _Length = vec2(), Color color = Color(-1, -1, -1, -1), bool _Queue = false);

	void rotate(uint64_t _Index, double _Angle, bool queue = false);

private:
	std::vector<vec2> _Position;
	std::vector<vec2> _Length;
};

class CircleVector : public DefaultShapeVector {
public:
	CircleVector(uint64_t _Number_Of_Points, float _Radius);
	CircleVector(const vec2& _Position, float _Radius, uint64_t _Number_Of_Points, const Color& _Color);

	void set_radius(float _Radius, uint64_t index);

	void set_position(uint64_t _Index, const vec2& _Position);

	void push_back(vec2 _Position, float _Radius, Color _Color, bool _Queue = false);

private:
	std::vector<vec2> _Position;
	std::vector<float> _Radius;
	const float _Double_Pi = 2.0f * PI;
};

class Sprite {
public:
	Sprite();
	Sprite(
		const char* path,
		vec2 position = vec2(),
		vec2 size = vec2(),
		float angle = 0,
		Shader shader = Shader("GLSL/core.vs", "GLSL/core.frag")
	);
	Sprite(
		unsigned char* pixels,
		const vec2& image_size,
		const vec2& position,
		const vec2& size,
		Shader shader = Shader("GLSL/core.vs", "GLSL/core.frag")
	);
	~Sprite() {
		glDeleteTextures(1, &texture.texture);
		glDeleteVertexArrays(1, &texture.VAO);
		glDeleteBuffers(1, &texture.VBO);
		glDeleteBuffers(1, &texture.EBO);
	}
	void draw();
	void init_image();
	void load_image(const char* path, Texture& object);
	void load_image(unsigned char* pixels, Texture& object);
	void reload_image(unsigned char* pixels);

	Texture& get_texture();
	vec2 get_size() const;

	vec2 get_position() const;
	void set_position(const vec2& position);

	float get_angle() const;
	void set_angle(float angle);
	Camera2D* camera;
protected:
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
	uint64_t particles_per_second = 1000;

	Particles(uint64_t particles_amount, vec2 particle_size, vec2 particle_speed, float life_time, Color begin, Color end);

	void add(vec2 position);

	void draw();

private:
	int64_t particleIndex;
	SquareVector particles;
	std::vector<Particle> particle;
	Color begin;
	Color end;
	float life_time;
};

template <typename shape>
class Entity : public shape {
public:
	template<
		typename T = shape,
		typename _Shader = Shader,
		std::enable_if_t<std::is_same<Sprite, T>::value>* = nullptr
	>
		constexpr Entity(
			const char* path,
			vec2 position,
			vec2 size,
			float angle,
			_Shader shader = _Shader("GLSL/core.frag", "GLSL/core.vs")
		) :
		Sprite(path, position, size, angle, shader), velocity(0) { }

	template<
		typename T = shape,
		std::enable_if_t<std::is_same<SquareVector, T>::value>* = nullptr
	>
		constexpr Entity(
			const vec2& _Position,
			const vec2& _Length,
			const Color& color
		) :
		SquareVector(_Position, _Length, color), velocity(0) { }

	constexpr vec2 get_velocity() const {
		return movement_speed;
	}

	constexpr void set_velocity(const vec2& new_velocity) {
		velocity = new_velocity;
	}

private:
	vec2 position = this->get_position();
	const float movement_speed = 2000;
	const float friction = 5;
	vec2 velocity;
};

class button : public SquareVector {
public:
	button() {}
	button(
		const vec2& position,
		const vec2& size,
		const Color& color,
		std::function<void()> lambda = std::function<void()>()
	);

	void add(
		const vec2& _Position,
		vec2 _Length = vec2(),
		Color color = Color(-1, -1, -1, -1),
		std::function<void()> lambda = std::function<void()>(),
		bool queue = false
	);
	void add(const button& button);

	void button_press_callback(uint64_t index = 0);

	bool inside(uint64_t index = 0) const;

	uint64_t amount() const;

private:
	using SquareVector::push_back;
	std::vector<std::function<void()>> callbacks;
	uint64_t count;
};

class button_single : public Square {
public:
	button_single() : Square(vec2(), vec2(), Color()) {}
	button_single(
		const vec2& position,
		const vec2& size,
		const Color& color,
		std::function<void()> lambda = std::function<void()>(),
		bool queue = false
	);

	void button_press_callback(uint64_t index = 0);

	bool inside() const;

private:
	std::function<void()> callback;
};

class Box {
public:
	Box(const vec2& position, const vec2& size, const Color& color);

	void set_position(uint64_t index, const vec2& position);

	void draw() const;

private:
	LineVector box_lines;
	std::vector<vec2> size;
};


#ifdef FAN_WINDOWS
static _vec2<int> cursor_screen_position() {
	POINT p;
	GetCursorPos(&p);
	return _vec2<int>(p.x, p.y);
}
#endif

vec2 Raycast(
	const vec2& start,
	const vec2& end,
	const SquareVector& squares,
	bool map[grid_size.x][grid_size.y]
);

#ifdef FT_FREETYPE_H
struct Character {
	GLuint TextureID;   // ID handle of the glyph texture
	_vec2<int> Size;    // Size of glyph
	_vec2<int> Bearing;  // Offset from baseline to left/top of glyph
	GLuint Advance;    // Horizontal offset to advance to next glyph
};

template <typename T>
constexpr Color lighter_color(const T& color, float offset) {
	return T(color + offset);
}

template <typename T>
constexpr Color darker_color(const T& color, float offset) {
	return T(color - offset);
}

constexpr auto blink_rate = 500; // ms
constexpr auto blinker_height = 15.f;
constexpr auto scroll_sensitivity = 50.f;
constexpr auto chat_begin_height = 100.f;
constexpr auto erase_speedlimit = 100; // ms
constexpr auto font_size = 0.4;
constexpr auto user_divider_x(300);
constexpr auto text_position_x = user_divider_x + 50.f;
constexpr auto title_bar_height = 25.f;
constexpr auto type_box_height = 50.f;
constexpr auto chat_box_max_width = 200.f;
constexpr auto text_gap = 20.f;
constexpr auto chat_boxes_gap = 3.f;
constexpr auto user_box_size = vec2(user_divider_x, 80);
constexpr auto scroll_max_gap = 100.f;
constexpr vec2 gap_between_text_and_box(8, 15);
constexpr auto chat_box_height = 18.f + gap_between_text_and_box.y;

constexpr Color background_color((float)0x0e, 0x16, (float)0x21, 0xff, true);
constexpr Color exit_cross_color(0.8, 0.8, 0.8);
constexpr auto user_box_color = Color(0x17, 0x21, 0x2b, 0xff, true);
constexpr Color title_bar_color(darker_color(user_box_color, 0.05));
constexpr Color highlight_color = title_bar_color + Color(0.1);
constexpr Color select_color(0x2b, 0x52, 0x78, 0xff, true);
constexpr Color white_color(1);
constexpr Color red_color(1, 0, 0);
constexpr Color green_color(0, 1, 0);
constexpr Color blue_color(0, 0, 1);

constexpr auto my_chat_path = "my_chat";
constexpr auto their_chat_path = "their_chat";

class TextRenderer {
public:
	TextRenderer();

	vec2 get_length(std::string text, float scale, bool include_endl = false);

	void render(const std::string& text, vec2 position, float scale, const Color& color);

private:
	std::map<GLchar, Character> Characters;
	Shader shader;
	unsigned int VAO, VBO;
};

namespace fan_gui {
	enum class e_button {
		title_bar,
		exit,
		maximize,
		minimize
	};

	enum class text_box_side {
		LEFT,
		RIGHT
	};

	class text_box : public button_single {
	public:

		text_box();
		text_box(TextRenderer* renderer, std::string text, const vec2& position, const Color& color);

		std::string get_text() const;
		void set_text(std::string new_text, std::deque<std::string>& messages, uint64_t at);

		vec2 get_position() const;
		void set_position(const vec2& position, bool queue = false);

		void draw();

		static std::string get_finished_string(TextRenderer* renderer, std::string text);
		static vec2 get_size_all(TextRenderer* renderer, std::string text);

		static void refresh(
			std::vector<text_box>& chat_boxes,
			const std::deque<std::string>& messages,
			TextRenderer* tr,
			text_box_side side = text_box_side::RIGHT,
			int offset = -1
		);

		static void refresh(
			std::vector<text_box>& chat_boxes,
			const std::deque<std::string>& messages,
			std::vector<text_box>& second_boxes,
			const std::deque<std::string>& second_messages,
			TextRenderer* tr,
			int offset = -1
		);

	private:
		TextRenderer* renderer;
		std::string text;
		float first_line_size;
	};

	class Titlebar {
	public:
		Titlebar();

		void cursor_update();
		void resize_update();

		vec2 get_position(e_button button);

		void move_window();

		void callbacks();

		void move_window(bool state);
		bool allow_move() const;

		void draw();

	private:
		vec2 title_bar_button_size{
			title_bar_height,
			title_bar_height
		};

		button buttons{
			vec2(0, 0),
			vec2(window_size.x, title_bar_height),
			title_bar_color
		};

		LineVector exit_cross{
			mat2x2(
				buttons.get_position(eti(e_button::exit)) +
				title_bar_shapes_size,
				buttons.get_position(eti(e_button::exit)) +
				title_bar_button_size - title_bar_shapes_size
			),
			exit_cross_color
		};

		Box maximize_box{
			vec2(),
			exit_cross.get_length() / 2,
			exit_cross_color
		};

		LineVector minimize_line{
			mat2x2(),
			exit_cross_color
		};

		vec2 old_cursor_offset;
		float title_bar_shapes_size = 6;
		vec2 maximize_box_size{ title_bar_shapes_size * 2 };
		bool m_bMaximized = false;
		bool m_bAllowMoving = false;
	};

	typedef std::map<std::string, std::vector<fan_gui::text_box>> chat_box_t;
	typedef std::map<std::string, std::deque<std::string>> message_t;



	class Users {
	public:
		Users(const std::string& username, message_t chat);

		void add(const std::string& username, message_t chat);

		void draw();

		void color_callback();

		void resize_update();

		void render_text(TextRenderer& renderer);

		void select();

		void high_light(int i);

		int get_user_i() const;

		bool selected() const;

		std::string get_username(int i);

		void reset();

		std::string get_user() const;

		uint64_t size() const;
		std::vector<std::string> usernames;
	private:
		Line user_divider;
		button user_boxes;
		SquareVector background;

		std::string current_user;
		int current_user_i = 0;
	};
}

#endif

extern Camera camera3d;

class LineVector3D : public DefaultShapeVector {
public:
	LineVector3D();
	LineVector3D(const matrix<3, 2>& _M, const Color& color);

	matrix<2, 3> get_position(uint64_t _Index = 0) const;
	void set_position(uint64_t _Index, const matrix<3, 2>& _M, bool _Queue = false);

	void push_back(const matrix<3, 2>& _M, Color _Color = Color(-1, -1, -1, -1), bool _Queue = false);

	vec2 get_length(uint64_t _Index = 0) const;

	void draw() {
		if (_Vertices.empty()) {
			return;
		}
		_Shader.use();

		mat4 projection(1);
		mat4 view(1);

		view = _Camera->get_view_matrix();

		projection = Perspective(Radians(90.f), ((float)window_size.x / (float)window_size.y), 0.1f, 1000.0f);

		static int projLoc = glGetUniformLocation(_Shader.ID, "projection");
		static int viewLoc = glGetUniformLocation(_Shader.ID, "view");
		glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection[0][0]);
		glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view[0][0]);
		glBindVertexArray(_ShapeBuffer.VAO);
		glBindBuffer(GL_ARRAY_BUFFER, _VerticeBuffer.VBO);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, _ColorBuffer.VBO);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glDrawArrays(_Mode, 0, _Points);
		glBindVertexArray(0);
	}

};

constexpr auto texture_coordinate_size = 72;

constexpr float square_vertices[108] = {

	 0.5,  0.5, -0.5, // left
	 0.5, -0.5, -0.5,
	-0.5, -0.5, -0.5,

	-0.5, -0.5, -0.5,
	-0.5,  0.5, -0.5,
	 0.5,  0.5, -0.5,


	-0.5,  0.5,  0.5, // right
	-0.5, -0.5,  0.5,
	 0.5, -0.5,  0.5,

	 0.5, -0.5,  0.5,
	 0.5,  0.5,  0.5,
	-0.5,  0.5,  0.5,

	-0.5,  0.5, -0.5,
	-0.5, -0.5, -0.5,
	-0.5, -0.5,  0.5, // front

	-0.5, -0.5,  0.5,
	-0.5,  0.5,  0.5,
	-0.5,  0.5, -0.5,


	 0.5, 0.5, 0.5,
	 0.5, -0.5, 0.5,
	 0.5, -0.5, -0.5, // back

	 0.5, -0.5, -0.5,
	 0.5, 0.5, -0.5,
	 0.5, 0.5, 0.5,


	-0.5, -0.5, -0.5, // down
	 0.5, -0.5, -0.5,
	 0.5, -0.5,  0.5,

	 0.5, -0.5,  0.5,
	-0.5, -0.5,  0.5,
	-0.5, -0.5, -0.5,

	 0.5,  0.5, -0.5, // up
	-0.5,  0.5, -0.5,
	-0.5,  0.5,  0.5,

	-0.5,  0.5,  0.5,
	 0.5,  0.5,  0.5,
	 0.5,  0.5, -0.5,
};

struct mesh_vertex {
	vec3 position;
	vec3 normal;
	vec2 texture_coordinates;
};

struct mesh_texture {
	unsigned int id;
	std::string type;
	aiString path;
};

#include <cstddef>

class model_mesh {
public:
	std::vector<mesh_vertex> vertices;
	std::vector<unsigned int> indices;
	std::vector<mesh_texture> textures;
	unsigned int vao, vbo, ebo;

	model_mesh(
		const std::vector<mesh_vertex>& vertices,
		const std::vector<unsigned int>& indices,
		const std::vector<mesh_texture>& textures
	);

private:

	void initialize_mesh();
};

int load_texture(const std::string_view path, const std::string& directory);

class model_loader {
protected:
	model_loader(const std::string& path);

	std::vector<model_mesh> meshes;
	std::vector<mesh_texture> textures_loaded;
private:
	void load_model(const std::string& path);

	void process_node(aiNode* node, const aiScene* scene);

	model_mesh process_mesh(aiMesh* mesh, const aiScene* scene);

	std::vector<mesh_texture> load_material_textures(aiMaterial* mat, aiTextureType type, const std::string& type_name);

	std::string directory;
};

class basic_3d {
public:
	basic_3d() : _Camera(nullptr), _Shape_Matrix_VAO(0), _Shape_Matrix_VBO(0) {}

	void init(const std::string& vs, const std::string& fs);
	void init_matrices();

	void set_position(uint64_t index, const vec3& position, bool queue = false);

	vec3 get_size(uint64_t i) const;

	vec3 get_position(uint64_t i) const;

	void push_back(const vec3& position, const vec3& size, bool queue = false);
	void insert(const std::vector<mat4> positions, const vec3& size, bool queue = false);

	void free_queue();

	uint64_t size() const;

protected:

	void set_projection();

	Camera* _Camera;
	Shader _Shader;

	std::vector<mat4> object_matrix;

	unsigned int _Shape_Matrix_VAO;
	unsigned int _Shape_Matrix_VBO;
};

class SquareVector3D : public basic_3d {
public:
	SquareVector3D(std::string_view path);

	void init(std::string_view path);

	void free_queue(bool vertices = true, bool texture = true);

	void draw();

	void change_texture(uint64_t index, const vec2& texture_id);

	void insert(const std::vector<mat4> positions, const vec3& size, const vec2& texture_id, bool queue = false);

	void push_back(const vec3& position, const vec3& size, const vec2& texture_id, bool queue = false);

	void erase(uint64_t first, uint64_t last = -1, bool queue = false);

	static constexpr vec2 texture_size = vec2(32, 32);

private:
	using basic_3d::push_back;

	template <typename T>
	std::vector<T> get_texture_onsided(_vec2<uint32_t> size, _vec2<uint32_t> position);

	void generate_textures(std::string_view path);

	vec2 texturepack_size;

	unsigned int _Shape_VAO;
	unsigned int _Shape_Vertices_VBO;

	unsigned int _Texture_VBO;
	unsigned int _Texture_SSBO;
	unsigned int _Texture_Id_SSBO;

	std::vector<std::vector<vec2::type>> _Textures;
	std::vector<int> _Texture_Ids;
};

void add_chunk(SquareVector3D& square_vector, const vec3& position, const vec3& chunk_size, const vec2& texture_id, bool queue = false);
void remove_chunk(SquareVector3D& square_vector, uint64_t chunk);

class Model : public model_loader, public basic_3d {
public:
	Model(const std::string& path, 
		const std::string& vs = "GLSL/models.vs",
		const std::string& frag = "GLSL/models.frag"
	);

	void draw();
};

class skybox {
public:
	skybox(
		const std::string& left,
		const std::string& right,
		const std::string& front,
		const std::string back,
		const std::string bottom,
		const std::string& top
	);

	~skybox();

	void draw();

private:
	unsigned int texture_id;
	unsigned int skybox_vao, skybox_vbo;


	Shader shader;
	Camera* camera;
	static constexpr float skyboxVertices[108] = {
		-1.0f,  1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,
		1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,

		-1.0f, -1.0f,  1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f,  1.0f,
		-1.0f, -1.0f,  1.0f,

		1.0f, -1.0f, -1.0f,
		1.0f, -1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,

		-1.0f, -1.0f,  1.0f,
		-1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f, -1.0f,  1.0f,
		-1.0f, -1.0f,  1.0f,

		-1.0f,  1.0f, -1.0f,
		1.0f,  1.0f, -1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		-1.0f,  1.0f,  1.0f,
		-1.0f,  1.0f, -1.0f,

		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f,  1.0f,
		1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f,  1.0f,
		1.0f, -1.0f,  1.0f
	};
};

class model_skybox : public Model {
public:
	model_skybox(const std::string& path);

	void draw();
};

//struct Particle3D {
//	float life_time;
//	Timer time;
//	bool display;
//	vec3 particle_speed;
//};
//
//class Particles3D {
//public:
//	uint64_t particles_per_second = 1000;
//	//Color begin;
//	Particles3D(uint64_t particles_amount, const vec3& particle_size, const vec3& particle_speed, float life_time, const char* path) :
//		particles(particles_amount, -particle_size,
//			vec3(particle_size), path), particle(), particleIndex(particles_amount - 1), begin(begin), end(end), life_time(life_time) {
//		for (int i = 0; i < particles_amount; i++) {
//			particle.push_back({
//				life_time,
//				Timer(high_resolution_clock::now(), 0),
//				0,
//				vec3(random(1, 10) / 10) * vec3(random(1, 10) / 10, 0, 0)
//			});
//		}
//	}
//
//	void add(const vec3& position) {
//		static Timer click_timer = {
//			high_resolution_clock::now(),
//			particles_per_second ? uint64_t(1000 / particles_per_second) : uint64_t(1e+10)
//		};
//			if (particle[particleIndex].time.finished() && click_timer.finished()) {
//				//(position - particles.get_size(0) / 2).print();
//		particles.set_position(position - particles.get_size(0) / 2, particleIndex, true);
//		particle[particleIndex].time.start(life_time);
//		particle[particleIndex].display = true;
//		if (--particleIndex <= -1) {
//			particleIndex = particles.amount() - 1;
//		}
//		click_timer.restart();
//		}
//	}
//
//	void draw() {
//		for (int i = 0; i < particles.amount(); i++) {
//			/*if (!particle[i].display) {
//				continue;
//			}
//			if (particle[i].time.finished()) {
//				particles.set_position(vec3(-particles.get_size(0)), i,  true);
//				particle[i].display = false;
//				particle[i].time.start(life_time);
//				continue;
//			}*/
//			/*Color color = particles.get_color(i);
//			const float passed_time = particle[i].time.elapsed();
//			float life_time = particle[i].life_time;
//
//			color.r = ((end.r - begin.r) / life_time) * passed_time + begin.r;
//			color.g = ((end.g - begin.g) / life_time) * passed_time + begin.g;
//			color.b = ((end.b - begin.b) / life_time) * passed_time + begin.b;
//			color.a = (particle[i].life_time - passed_time / 1.f) / particle[i].life_time;
//			particles.set_color(i, color, true);*/
//			particles.set_position(particles.get_position(i) + particle[i].particle_speed * delta_time, i, true);
//		}
//		particles.break_queue();
//		particles.draw();
//	}
//
//	uint64_t size() {
//		return particle.size();
//	}
//
//	void break_queue() {
//		this->particles.break_queue();
//	}
//
//	void set_color(const Color& color, uint64_t i, bool queue = false) {
//		particles.set_color(i, color, queue);
//	}
//
//	void set_speed(const vec3& speed, uint64_t i) {
//		particle[i].particle_speed = speed;
//	}
//
//private:
//	int64_t particleIndex;
//	SquareVector3D particles;
//	std::vector<Particle3D> particle;
//	Color begin;
//	Color end;
//	float life_time;
//};	

void move_camera(bool noclip, float movement_speed);
void rotate_camera();

enum class e_cube {
	left,
	right,
	front,
	back,
	down,
	up
};

extern std::initializer_list<e_cube> e_cube_loop;

extern std::vector<float> g_distances;

template <typename T>
inline vec3 intersection_point3d(const T& plane_position, const T& plane_size, const T& position, e_cube side) {
	T p0;
	T a;
	T b;
	const T l0 = position;

	switch (side) {
	case e_cube::left: {
		p0 = plane_position - T(plane_size.x / 2, plane_size.y / 2, plane_size.z / 2);
		a = p0 + vec3(0, plane_size.y, 0);
		b = p0 + vec3(plane_size.x, 0, 0);
		break;
	}
	case e_cube::right: {
		p0 = plane_position - T(plane_size.x / 2, plane_size.y / 2, -plane_size.z / 2);
		a = p0 + vec3(0, plane_size.y, 0);
		b = p0 + vec3(plane_size.x, 0, 0);
		break;
	}
	case e_cube::front: {
		p0 = plane_position - T(plane_size.x / 2, plane_size.y / 2, plane_size.z / 2);
		a = p0 + vec3(0, plane_size.y, 0);
		b = p0 + vec3(0, 0, plane_size.z);
		break;
	}
	case e_cube::back: {
		p0 = plane_position - T(-plane_size.x / 2, plane_size.y / 2, plane_size.z / 2);
		a = p0 + vec3(0, plane_size.y, 0);
		b = p0 + vec3(0, 0, plane_size.z);
		break;
	}
	case e_cube::up: {
		p0 = plane_position - T(plane_size.x / 2, -plane_size.y / 2, plane_size.z / 2);
		a = p0 + vec3(plane_size.x, 0, 0);
		b = p0 + vec3(0, 0, plane_size.z);
		break;
	}
	case e_cube::down: {
		p0 = plane_position - T(plane_size.x / 2, plane_size.y / 2, plane_size.z / 2);
		a = p0 + vec3(plane_size.x, 0, 0);
		b = p0 + vec3(0, 0, plane_size.z);
		break;
	}
	}

	const T n = Normalize(Cross((a - p0), (b - p0)));

	const T l = DirectionVector(camera3d.yaw, camera3d.pitch);

	const float nl_dot(Dot(n, l));

	if (!nl_dot) {
		return T(-1);
	}

	const float d = Dot(p0 - l0, n) / nl_dot;
	if (d <= 0) {
		return T(-1);
	}

	g_distances[eti(side)] = d;

	const T intersect(l0 + l * d);
	switch (side) {
	case e_cube::right:
	case e_cube::left: {
		if (intersect.y > b.y && intersect.y < a.y &&
			intersect.x > a.x && intersect.x < b.x) {
			return intersect;
		}
		break;
	}
	case e_cube::back:
	case e_cube::front: {
		if (intersect.y > b.y && intersect.y < a.y &&
			intersect.z > a.z && intersect.z < b.z) {
			return intersect;
		}
		break;
	}
	case e_cube::up:
	case e_cube::down: {
		if (intersect.x > b.x && intersect.x < a.x &&
			intersect.z > a.z && intersect.z < b.z) {
			return intersect;
		}
		break;
	}
	}

	return T(-1);
}

double ValueNoise_2D(double x, double y);

struct hash_vector_operators {
	size_t operator()(const vec3& k) const {
		return std::hash<float>()(k.x) ^ std::hash<float>()(k.y) ^ std::hash<float>()(k.z);
	}

	bool operator()(const vec3& a, const vec3& b) const {
		return a.x == b.x && a.y == b.y && a.z == b.z;
	}
};

constexpr int world_size = 150;

#define pass_array(d_map, d_position) \
		d_map[d_position.x] \
			 [d_position.y] \
			 [d_position.z]

constexpr auto grid_direction(const vec3& src, const vec3& dst) {
	vec3 x(src.x - dst.x, src.y - dst.y, src.z - dst.z);
	return x / x.abs().max();
}

struct grid_raycast_s {
	vec3 direction, begin;
	vec3i grid;
};

constexpr bool grid_raycast_single(grid_raycast_s& caster, float grid_size) {
	vec3 position(caster.begin % grid_size);
	for (uint8_t i = 0; i < vec3::size(); i++) {
		position[i] = ((caster.direction[i] < 0) ? position[i] : grid_size - position[i]);
		position[i] = std::abs((!position[i] ? grid_size : position[i]) / caster.direction[i]);
	}
	caster.grid = (caster.begin += caster.direction * position.min()) / grid_size;
	for (uint8_t i = 0; i < vec3::size(); i++)
		caster.grid[i] -= ((caster.direction[i] < 0) & (position[i] == position.min()));
	return 1;
}

inline vec3i grid_raycast(const vec3& start, const vec3& end, const map_t& map, float block_size) {
	if (start == end) {
		return start;
	}
	grid_raycast_s raycast = { grid_direction(end, start), start, vec3() };
	vec3 distance = end - start;
	auto max = distance.abs().max();
	for (int i = 0; i < max; i++) {
		grid_raycast_single(raycast, block_size);
		if (raycast.grid[0] < 0 || raycast.grid[1] < 0 || raycast.grid[2] < 0 ||
			raycast.grid[0] >= world_size || raycast.grid[1] >= world_size || raycast.grid[2] >= world_size) {
			continue;
		}
		if (map[raycast.grid[0]][raycast.grid[1]][raycast.grid[2]]) {
			return raycast.grid;
		}
	}
	return vec3(RAY_DID_NOT_HIT);
}

#define d_grid_raycast(start, end, raycast, block_size) \
	grid_raycast_s raycast = { grid_direction(end, start), start, vec3() }; \
	if (!(start == end)) \
		while(grid_raycast_single(raycast, block_size))
//#endif