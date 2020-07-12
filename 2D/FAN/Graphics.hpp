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
#if defined(_WIN64) || defined(_WIN32) && !defined(FAN_WINDOWS)
#define FAN_WINDOWS
#endif

#include <vector>
#include <array>
#include <map>

#include "Input.hpp"
#include "Math.hpp"
#include "Shader.h"
#include "Vectors.hpp"
#include "Time.hpp"
#include "Network.hpp"
#include <SOIL2/SOIL2.h>
#include <SOIL2/stb_image.h>

#ifdef _MSC_VER
#pragma warning (disable : 26495)
#endif

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#define COLORSIZE 4

class Camera;
extern Camera camera2d;

void GetFps(bool title = true, bool print = false);

extern bool window_init;
constexpr auto WINDOWSIZE = vec2i(800, 800);
extern float delta_time;
static constexpr int block_size = 50;
extern GLFWwindow* window;
constexpr auto grid_size = vec2i(WINDOWSIZE.x / block_size, WINDOWSIZE.y / block_size);

typedef std::vector<std::vector<std::vector<bool>>> map_t;

struct bmp {
	unsigned char* data;
	unsigned char* image;
};

struct Texture {
	Texture();

	unsigned int texture;
	int width, height;
	unsigned int VBO, VAO;
};

namespace BMP_Offsets {
	constexpr::ptrdiff_t PIXELDATA = 0xA;
	constexpr::ptrdiff_t WIDTH = 0x12;
	constexpr::ptrdiff_t HEIGHT = 0x16;
}

bmp LoadBMP(const char* path, Texture& texture);
uint64_t _2d_1d(vec2 position = cursor_position);

vec2i window_position();

class Camera {
public:
	Camera(
		vec3 position = vec3(0.0f, 0.0f, 0.0f),
		vec3 up = vec3(0.0f, 1.0f, 0.0f),
		float yaw = 0,
		float pitch = 0.0f
	);

	void move(bool noclip, float_t movement_speed);
	void rotate_camera();

	mat4 get_view_matrix();
	mat4 get_view_matrix(mat4 m);

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

int load_texture(const std::string_view path, const std::string& directory = std::string(), bool flip_image = false, bool alpha = false);

void write_vbo(unsigned int buffer, void* data, std::uint64_t size);

template <typename _Vector>
class basic_shape_vector {
public:

	basic_shape_vector(const Shader& shader);
	basic_shape_vector(const Shader& shader, const _Vector& position, const _Vector& size);
	~basic_shape_vector();

	_Vector get_size(std::uint64_t i);
	void set_size(std::uint64_t i, const _Vector& size, bool queue = false);

	_Vector get_position(std::uint64_t i);
	void set_position(std::uint64_t i, const _Vector& position, bool queue = false);

	void basic_push_back(const _Vector& position, const _Vector& size, bool queue = false);

	std::uint64_t size();

	void write_data(bool position, bool size);

protected:

	void initialize_buffers();

	void basic_draw(unsigned int mode, std::uint64_t count, std::uint64_t primcount);

	unsigned int vao;
	unsigned int position_vbo;
	unsigned int size_vbo;

	std::vector<_Vector> m_position;
	std::vector<_Vector> m_size;

	Shader m_shader;

};

class basic_shape_color_vector {
public:

	basic_shape_color_vector();
	basic_shape_color_vector(const Color& color);
	~basic_shape_color_vector();

	Color get_color(std::uint64_t i);
	void set_color(std::uint64_t i, const Color& color, bool queue = false);

	void basic_push_back(const Color& color, bool queue = false);

	void write_data();

protected:

	void initialize_buffers();

	unsigned int color_vbo;

	std::vector<Color> m_color;

};

enum class shape_types {
	LINE,
	SQUARE
};

namespace fan_2d {

	extern mat4 frame_projection;
	extern mat4 frame_view;

	namespace shader_paths {
		constexpr auto text_renderer_vs("GLSL/2D/text.vs");
		constexpr auto text_renderer_fs("GLSL/2D/text.fs");

		constexpr auto single_shapes_path_vs("GLSL/2D/shapes.vs");
		constexpr auto single_shapes_path_fs("GLSL/2D/shapes.fs");
		constexpr auto single_sprite_path_vs("GLSL/2D/sprite.vs");
		constexpr auto single_sprite_path_fs("GLSL/2D/sprite.fs");

		constexpr auto shape_vector_vs("GLSL/2D/shape_vector.vs");
		constexpr auto shape_vector_fs("GLSL/2D/shapes.fs");
		constexpr auto sprite_vector_vs("GLSL/2D/sprite_vector.vs");
		constexpr auto sprite_vector_fs("GLSL/2D/sprite_vector.fs");
	}

	class basic_single_shape {
	public:

		basic_single_shape();

		basic_single_shape(const Shader& shader, const vec2& position, const vec2& size);

		vec2 get_position();
		vec2 get_size();

		void set_size(const vec2& size);
		void set_position(const vec2& position);

		void basic_draw(GLenum mode, GLsizei count);

	protected:
		vec2 position;
		vec2 size;

		Shader shader;

		unsigned int vao;
	};

	struct basic_single_color {

		basic_single_color();
		basic_single_color(const Color& color);

		Color get_color();
		void set_color(const Color& color);

		Color color;

	};

	struct line : public basic_single_shape, basic_single_color {

		line();
		line(const vec2& begin, const vec2& end, const Color& color);

		void draw();

		void set_position(const vec2& begin, const vec2& end);

	private:
		using basic_single_shape::set_position;
		using basic_single_shape::set_size;
	};

	struct square : public basic_single_shape, basic_single_color {
		square();
		square(const vec2& position, const vec2& size, const Color& color);

		void draw();
	};

	struct image_info {
		vec2i image_size;
		unsigned int texture_id;
	};

	class sprite : public basic_single_shape {
	public:
		sprite();

		// scale with default is sprite size
		sprite(const std::string& path, const vec2& position, const vec2& size = 0);

		void draw();

		static image_info load_image(const std::string& path, bool flip_image = false);

	private:

		unsigned int texture;
	};

	class line_vector : public basic_shape_vector<vec2>, public basic_shape_color_vector {
	public:
		line_vector();
		line_vector(const mat2& begin_end, const Color& color);

		void push_back(const mat2& begin_end, const Color& color, bool queue = false);

		void draw();

		void set_position(std::uint64_t i, const mat2& begin_end, bool queue = false);

		void release_queue(bool position, bool color);

	private:
		using basic_shape_vector::set_position;
		using basic_shape_vector::set_size;
	};

	struct square_vector : public basic_shape_vector<vec2>, public basic_shape_color_vector {

		square_vector();
		square_vector(const vec2& position, const vec2& size, const Color& color);

		void release_queue(bool position, bool size, bool color);

		void push_back(const vec2& position, const vec2& size, const Color& color, bool queue = false);

		void draw();
	};

	class sprite_vector : public basic_shape_vector<vec2> {
	public:

		sprite_vector();
		sprite_vector(const std::string& path, const vec2& position, const vec2& size = 0);
		~sprite_vector();

		void push_back(const vec2& position, const vec2& size = 0, bool queue = false);

		void draw();

		void release_queue(bool position, bool size);

	private:

		unsigned int texture;
		vec2i original_image_size;

	};
}

namespace fan_3d {

	namespace shader_paths {
		constexpr auto shape_vector_vs("GLSL/3D/shape_vector.vs");
		constexpr auto shape_vector_fs("GLSL/3D/shape_vector.fs");

		constexpr auto skybox_vs("GLSL/3D/skybox.vs");
		constexpr auto skybox_fs("GLSL/3D/skybox.fs");
		constexpr auto skybox_model_vs("GLSL/3D/skybox_model.vs");
		constexpr auto skybox_model_fs("GLSL/3D/skybox_model.fs");

		constexpr auto instancing_vs("GLSL/3D/instancing.vs");
		constexpr auto instancing_fs("GLSL/3D/instancing.fs");
	}

	extern Camera camera;
	extern mat4 frame_projection;
	extern mat4 frame_view;

	class line_vector : public basic_shape_vector<vec3>, public basic_shape_color_vector {
	public:

		line_vector();
		line_vector(const mat2x3& begin_end, const Color& color);

		void push_back(const mat2x3& begin_end, const Color& color, bool queue = false);

		void draw();

		void set_position(std::uint64_t i, const mat2x3 begin_end, bool queue = false);
		
		void release_queue(bool position, bool color);

	private:

		using basic_shape_vector::set_position;
		using basic_shape_vector::set_size;

	};

	class square_vector : public basic_shape_vector<vec3> {
	public:

		square_vector(const std::string& path, std::uint64_t block_size);

		void push_back(const vec3& position, const vec3& size, const vec2& texture_id, bool queue = false);

		void draw();

		void set_texture(std::uint64_t i, const vec2& texture_id, bool queue = false);

		void generate_textures(const std::string& path, const vec2& block_size);

		void write_textures();

		void release_queue(bool position, bool size, bool textures);

	private:

		unsigned int m_texture;
		unsigned int m_texture_ssbo;
		unsigned int m_texture_id_ssbo;

		vec2i block_size;

		std::vector<int> m_textures;

	};

}

constexpr auto texture_coordinate_size = 72;

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

enum class e_cube {
	left,
	right,
	front,
	back,
	down,
	up
};

extern std::vector<float> g_distances;

vec3 intersection_point3d(const vec3& plane_position, const vec3& plane_size, const vec3& position, e_cube side);

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

template <typename T>
constexpr auto grid_direction(const T& src, const T& dst) {
	T vector(src - dst);
	return vector / vector.abs().max();
}

template <template <typename> typename T>
struct grid_raycast_s {
	T<float_t> direction, begin;
	T<int> grid;
};

template <template <typename> typename T>
constexpr bool grid_raycast_single(grid_raycast_s<T>& caster, float_t grid_size) {
	T position(caster.begin % grid_size);
	for (uint8_t i = 0; i < T<float_t>::size(); i++) {
		position[i] = ((caster.direction[i] < 0) ? position[i] : grid_size - position[i]);
		position[i] = std::abs((!position[i] ? grid_size : position[i]) / caster.direction[i]);
	}
	caster.grid = (caster.begin += caster.direction * position.min()) / grid_size;
	for (uint8_t i = 0; i < T<float_t>::size(); i++)
		caster.grid[i] -= ((caster.direction[i] < 0) & (position[i] == position.min()));
	return 1;
}

template <template <typename> typename T>
constexpr T<int> grid_raycast(const T<float_t>& start, const T<float_t>& end, const map_t& map, float_t block_size) {
	if (start == end) {
		return start;
	}
	grid_raycast_s raycast = { grid_direction(end, start), start, T<int>() };
	T distance = end - start;
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
	return T(RAY_DID_NOT_HIT);
}

#define d_grid_raycast(start, end, raycast, block_size) \
	grid_raycast_s raycast = { grid_direction(end, start), start, vec3() }; \
	if (!(start == end)) \
		while(grid_raycast_single(raycast, block_size))

#include <ft2build.h>
#include FT_FREETYPE_H

typedef struct {
	std::vector<uint8_t> data;
	uint_t datasize;

	uint_t fontsize;
	vec2i offset;
}suckless_font_t;

typedef struct {
	vec2i pos;
	uint_t width;
	uint_t height;
}letter_info_t;

typedef struct {
	vec2 pos;
	float_t width;
}letter_info_opengl_t;

static letter_info_opengl_t letter_to_opengl(const suckless_font_t& font, const letter_info_t& letter) 
{
	letter_info_opengl_t ret;
	ret.pos = (vec2)letter.pos / (vec2)font.datasize;
	ret.width = (float_t)letter.width / font.datasize;
	return ret;
}

inline void emplace_vertex_data(std::vector<vec2>& vector, const vec2& position, const vec2& size) 
{
	vector.emplace_back(vec2(position.x, position.y));
	vector.emplace_back(vec2(position.x, position.y + size.y));
	vector.emplace_back(vec2(position.x + size.x, position.y + size.y));
	vector.emplace_back(vec2(position.x, position.y));
	vector.emplace_back(vec2(position.x + size.x, position.y + size.y));
	vector.emplace_back(vec2(position.x + size.x, position.y));
}

inline void edit_vertex_data(std::uint64_t offset, std::vector<vec2>& vector, const vec2& position, const vec2& size) 
{
	vector[offset] =     vec2(position.x, position.y);
	vector[offset + 1] = vec2(position.x, position.y + size.y);
	vector[offset + 2] = vec2(position.x + size.x, position.y + size.y);
	vector[offset + 3] = vec2(position.x, position.y);
	vector[offset + 4] = vec2(position.x + size.x, position.y + size.y);
	vector[offset + 5] = vec2(position.x + size.x, position.y);
}

inline void erase_vertex_data(std::uint64_t offset, std::vector<vec2>& vector, std::uint64_t size) {
	vector.erase(vector.begin() + offset * (size * 6), vector.begin() + (offset * ((size * 6))) + size * 6);
}

template <typename T>
constexpr auto vector_2d_to_1d(const std::vector<std::vector<T>>& vector) {
	std::vector<T> new_vector(vector.size());
	for (auto i : vector) {
		new_vector.insert(new_vector.end(), i.begin(), i.end());
	}
	return new_vector;
}

constexpr uint_t max_ascii = 248;
constexpr uint_t max_font_size = 1024;

namespace fan_gui {

	constexpr Color default_text_color(1);
	constexpr float_t font_size(128);

	class text_renderer {
	public:
		text_renderer();

		~text_renderer();

		void alloc_storage(const std::vector<std::wstring>& vector);
		void realloc_storage(const std::vector<std::wstring>& vector);

		void store_to_renderer(std::wstring& text, vec2 position, const Color& color, float_t scale, float_t max_width = -1);
		void edit_storage(uint64_t i, const std::wstring& text, vec2 position, const Color& color, float_t scale);

		void upload_stored();
		void upload_stored(uint64_t i);

		void render_stored();
		void set_scale(uint64_t i, float_t font_size, vec2 position);

		void clear_storage();

		void render(const std::vector<std::wstring>& text, std::vector<vec2> position, const std::vector<Color>& color, const std::vector<float_t>& scale);
		void render(const std::wstring& text, vec2 position, const Color& color, float_t scale, bool use_old = true);

		vec2 get_length(const std::wstring& text, float_t scale);
		std::vector<vec2> get_length(const std::vector<std::wstring>& texts, const std::vector<float_t>& scales, bool half = false);

		std::array<float_t, 248> widths;
		std::vector<std::vector<int>> characters;
		std::vector<std::vector<Color>> colors;
		std::vector<std::vector<vec2>> draw_data;
		std::vector<float_t> scales;

		int storage_id;

		suckless_font_t font;

	private:

		Shader shader;
		unsigned int VAO, vertex_ssbo;
		unsigned int texture;
		unsigned int text_ssbo;
		unsigned int _Texture_Id_SSBO;
		unsigned int colors_ssbo;

	};

	namespace text_button {
		constexpr vec2 gap_scale(0.25, 0.25);
		constexpr float_t space_width = 10;
		constexpr float_t space_between_characters = 5;

		constexpr vec2 get_gap_scale(const vec2& size) {
			return size * gap_scale;
		}

		constexpr float_t get_gap_scale_x(float_t width) {
			return width * gap_scale.x;
		}

		constexpr float_t get_gap_scale_y(float_t height) {
			return height * gap_scale.y;
		}

		constexpr float_t get_character_x_offset(float_t width, float_t scale) {
			return width * scale + space_between_characters;
		}

		constexpr float_t get_space(float_t scale) {
			return scale / (font_size / 2) * space_width;
		}

		class basic_text_button_vector : public text_renderer {
		public:

			basic_text_button_vector();

			vec2 edit_size(uint64_t i, const std::wstring& text, float_t scale);

		protected:
			std::vector<std::wstring> texts;
		};

		class text_button_vector : public basic_text_button_vector, public fan_2d::square_vector {
		public:

			text_button_vector();

			text_button_vector(const std::wstring& text, const vec2& position, const Color& box_color, float_t font_scale, float_t left_offset, float_t max_width);

			text_button_vector(const std::wstring& text, const vec2& position, const Color& color, float_t scale);

			void add(const std::wstring& text, const vec2& position, const Color& color, float_t scale);

			void edit_string(uint64_t i, const std::wstring& text, float_t scale);

			vec2 get_string_length(const std::wstring& text, float_t scale);

			float_t get_scale(uint64_t i);

			void set_font_size(uint64_t i, float_t scale);
			void set_position(uint64_t i, const vec2& position);

			void draw();

		private:
			//using 
			using fan_2d::square_vector::draw;
			using fan_2d::square_vector::set_size;
		};
	}
}

//#endif


