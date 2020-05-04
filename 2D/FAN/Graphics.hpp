#pragma once
//#ifndef __INTELLISENSE__ 

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <ft2build.h>
#include FT_FREETYPE_H  

#include "Alloc.hpp"
#include "Input.hpp"
#include "Math.hpp"
#include "Settings.hpp"
#include "Shader.h"
#include "Vectors.hpp"
#include "Time.hpp"
#include "Network.hpp"

#include <map>
#include <vector>
#include <deque>

#ifdef _MSC_VER
#pragma warning (disable : 26495)
#endif

#define BackgroundSize 1500
#define COLORSIZE 4
#define COORDSIZE 2

class Sprite;

class SquareVector;

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
uint64_t _2d_1d(vec2 position = cursor_position);

_vec2<int> window_position();


class Camera {
public:
	Camera(
		vec3 position = vec3(0.0f, 0.0f, 0.0f),
		vec3 up = vec3(0.0f, 1.0f, 0.0f),
		float yaw = -90.0f,
		float pitch = 0.0f
	);

	matrix<4, 4> GetViewMatrix(matrix<4, 4> m);

	vec3 get_position() const;
	void set_position(const vec3& position);

	GLfloat yaw;
	GLfloat pitch;
	void updateCameraVectors();

private:
	vec3 front;
	vec3 up;
	vec3 right;
	vec3 worldUp;
	vec3 position;
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

	~vertice_handler();

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
	Camera* camera;
};

static vertice_handler<shapes::line> line_handler;
static vertice_handler<shapes::square> square_handler;

static void draw_all() {
	line_handler.draw();
	square_handler.draw();
}

class Line;
class Square;

template <shapes shape>
class default_shape {
public:
	using shape_type = std::remove_const_t<decltype(shape)>;

	virtual void draw();

protected:
	int draw_id;
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


class DefaultShapeVector {
public:
	~DefaultShapeVector();

	Color get_color(uint64_t _Index = 0) const;
	auto& get_color_ptr() const;
	void set_color(uint64_t _Index, const Color& color, bool queue = false);

	void draw(uint64_t first = 0, uint64_t last = 0) const;

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

private:
	Alloc<vec2> _Length;
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
	Alloc<vec2> _Position;
	Alloc<vec2> _Length;
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
	Alloc<vec2> _Position;
	Alloc<vec2> _Length;
};

class CircleVector : public DefaultShapeVector {
public:
	CircleVector(uint64_t _Number_Of_Points, float _Radius);
	CircleVector(const vec2& _Position, float _Radius, uint64_t _Number_Of_Points, const Color& _Color);

	void set_position(uint64_t _Index, const vec2& _Position);

	void push_back(vec2 _Position, float _Radius, Color _Color, bool _Queue = false);

private:
	Alloc<vec2> _Position;
	Alloc<float> _Radius;
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

	void draw();
	void init_image();
	void load_image(const char* path, Texture& object);

	Texture& get_texture();
	vec2 get_size() const;

	vec2 get_position() const;
	void set_position(const vec2& position);

	float get_angle() const;
	void set_angle(float angle);
protected:
	Camera* camera;
	Shader shader;
	Texture texture;
	vec2 position;
	vec2 size;
	float angle;
	Alloc<float> _Vertices;
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
	Alloc<Particle> particle;
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

	void render(std::string text, vec2 position, float scale, const Color& color);

private:
	std::map<GLchar, Character> Characters;
	Shader shader;
	unsigned int VAO, VBO;
};
#endif

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

//#endif