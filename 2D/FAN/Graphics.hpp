#pragma once
//#ifndef __INTELLISENSE__ 

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#define REQUIRE_GRAPHICS
#include <FAN/Global_Vars.hpp>

//#define FAN_PERFORMANCE
#define RAM_SAVER
#if defined(_WIN64) || defined(_WIN32) && !defined(FAN_WINDOWS)
#define FAN_WINDOWS
#endif

#include <vector>
#include <array>
#include <map>

#include <FAN/Input.hpp>
#include <FAN/Math.hpp>
#include <FAN/Shader.h>
#include <FAN/Vectors.hpp>
#include <FAN/Time.hpp>
#include <FAN/Network.hpp>
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

static constexpr int block_size = 50;
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

	void move(bool noclip, f_t movement_speed);
	void rotate_camera(bool when);

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
	bool firstMouse = true;

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

	_Vector get_size(std::uint64_t i) const;
	void set_size(std::uint64_t i, const _Vector& size, bool queue = false);

	std::vector<_Vector> get_positions() const;
	_Vector get_position(std::uint64_t i) const;
	void set_position(std::uint64_t i, const _Vector& position, bool queue = false);

	void basic_push_back(const _Vector& position, const _Vector& size, bool queue = false);

	void erase(std::uint64_t i);

	std::uint64_t size() const;

	bool empty() const;

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

protected:

	void basic_push_back(const Color& color, bool queue = false);

	void write_data();

	void initialize_buffers();

	unsigned int color_vbo;

	std::vector<Color> m_color;

};

constexpr da_t<f_t, 4, 2> get_square_corners(const da_t<f_t, 2, 2>& squ) {
	return da_t<f_t, 4, 2>{
		da_t<f_t, 2>(squ[0]),
			da_t<f_t, 2>(squ[1][0], squ[0][1]),
			da_t<f_t, 2>(squ[0][0], squ[1][1]),
			da_t<f_t, 2>(squ[1])
	};
}

enum class shape_types {
	LINE,
	SQUARE,
	TRIANGLE
};

namespace fan_2d {

	extern mat4 frame_projection;
	extern mat4 frame_view;
	extern Camera camera;

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

		~basic_single_shape();

		vec2 get_position() const;
		vec2 get_size() const;
		vec2 get_velocity() const;

		void set_size(const vec2& size);
		void set_position(const vec2& position);
		void set_velocity(const vec2& velocity);

		void basic_draw(GLenum mode, GLsizei count);

		void move(f_t speed, f_t gravity, f_t jump_force = -800, f_t friction = 10);

	protected:
		vec2 position;
		vec2 size;

		vec2 velocity;

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
		line(const mat2x2& begin_end, const Color& color);

		void draw();

		void set_position(const mat2x2& begin_end);

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
		sprite(const std::vector<unsigned char>& pixels, const vec2& position, const vec2i& size = 0);

		void reload_image(const std::vector<unsigned char>& pixels, const vec2i& size);
		void reload_image(const std::string& path, const vec2i& size);

		void draw();

		f_t get_rotation();
		void set_rotation(f_t degrees);

		static image_info load_image(const std::string& path, bool flip_image = false);
		static image_info load_image(const std::vector<unsigned char>& pixels, const vec2i& size);

	private:

		f_t m_rotation;

		unsigned int texture;
	};

	class animation : public basic_single_shape {
	public:

		animation(const vec2& position, const vec2& size);

		void add(const std::string& path);

		void draw(std::uint64_t texture);

	private:
		std::vector<unsigned int> m_textures;
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

	struct triangle_vector : public basic_shape_vector<vec2>, public basic_shape_color_vector {

		triangle_vector();
		triangle_vector(const mat3x2& corners, const Color& color);
		
		void set_position(std::uint64_t i, const mat3x2& corners);
		void push_back(const mat3x2& corners, const Color& color);

		void draw();

	private:
		std::vector<vec2> m_lcorners;
		std::vector<vec2> m_mcorners;
		std::vector<vec2> m_rcorners;

		uint_t l_vbo, m_vbo, r_vbo;

	};

	class square_vector : public basic_shape_vector<vec2>, public basic_shape_color_vector {
	public:

		square_vector();
		square_vector(const vec2& position, const vec2& size, const Color& color);

		void release_queue(bool position, bool size, bool color);

		void push_back(const vec2& position, const vec2& size, const Color& color, bool queue = false);
		void erase(uint_t i);

		void draw();

		std::vector<mat2x2> get_icorners() const;

	private:

		std::vector<mat2x2> m_icorners;

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

	struct particle {
		vec2 m_velocity;
		Timer m_timer; // milli
	};

	class particles : public fan_2d::square_vector {
	public:

		void add(
			const vec2& position, 
			const vec2& size, 
			const vec2& velocity, 
			const Color& color, 
			std::uint64_t time
		);

		void update();

	private:

		std::vector<fan_2d::particle> m_particles;
	};
}

namespace fan_3d {

	namespace shader_paths {
		constexpr auto shape_vector_vs("GLSL/3D/shape_vector.vs");
		constexpr auto shape_vector_fs("GLSL/3D/shape_vector.fs");

		constexpr auto model_vs("GLSL/3D/models.vs");
		constexpr auto model_fs("GLSL/3D/models.fs");

		constexpr auto skybox_vs("GLSL/3D/skybox.vs");
		constexpr auto skybox_fs("GLSL/3D/skybox.fs");
		constexpr auto skybox_model_vs("GLSL/3D/skybox_model.vs");
		constexpr auto skybox_model_fs("GLSL/3D/skybox_model.fs");
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

	class model_loader {
	protected:
		model_loader(const std::string& path, const vec3& size);

		std::vector<model_mesh> meshes;
		std::vector<mesh_texture> textures_loaded;
	private:
		void load_model(const std::string& path, const vec3& size);

		void process_node(aiNode* node, const aiScene* scene, const vec3& size);

		model_mesh process_mesh(aiMesh* mesh, const aiScene* scene, const vec3& size);

		std::vector<mesh_texture> load_material_textures(aiMaterial* mat, aiTextureType type, const std::string& type_name);

		std::string directory;
	};

	class model : public model_loader {
	public:
		model(const std::string& path, const vec3& position, const vec3& size);

		void draw();

		vec3 get_position();
		void set_position(const vec3& position);

		vec3 get_size();
		void set_size(const vec3& size);

	private:
		Shader m_shader;

		vec3 m_position;
		vec3 m_size;

	};

}

void GetFps(bool title = true, bool print = false);

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
	T<f_t> direction, begin;
	T<int> grid;
};

template <template <typename> typename T>
constexpr bool grid_raycast_single(grid_raycast_s<T>& caster, f_t grid_size) {
	T position(caster.begin % grid_size);
	for (uint8_t i = 0; i < T<f_t>::size(); i++) {
		position[i] = ((caster.direction[i] < 0) ? position[i] : grid_size - position[i]);
		position[i] = std::abs((!position[i] ? grid_size : position[i]) / caster.direction[i]);
	}
	caster.grid = (caster.begin += caster.direction * position.min()) / grid_size;
	for (uint8_t i = 0; i < T<f_t>::size(); i++)
		caster.grid[i] -= ((caster.direction[i] < 0) & (position[i] == position.min()));
	return 1;
}

template <template <typename> typename T>
constexpr T<int> grid_raycast(const T<f_t>& start, const T<f_t>& end, const map_t& map, f_t block_size) {
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
	f_t width;
}letter_info_opengl_t;

static letter_info_opengl_t letter_to_opengl(const suckless_font_t& font, const letter_info_t& letter) 
{
	letter_info_opengl_t ret;
	ret.pos = (vec2)letter.pos / (vec2)font.datasize;
	ret.width = (f_t)letter.width / font.datasize;
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
	constexpr f_t font_size(128);

	class text_renderer {
	public:
		text_renderer();

		~text_renderer();

		void render(const std::wstring& text, vec2 position, const Color& color, f_t scale, bool use_old = false);

	protected:

		void alloc_storage(const std::vector<std::wstring>& vector);
		void realloc_storage(const std::vector<std::wstring>& vector);

		void store_to_renderer(std::wstring& text, vec2 position, const Color& color, f_t scale, f_t max_width = -1);
		void edit_storage(uint64_t i, const std::wstring& text, vec2 position, const Color& color, f_t scale);

		void upload_vertices();
		void upload_colors();
		void upload_characters();

		void upload_stored();
		void upload_stored(uint64_t i);

		void render_stored();
		void set_scale(uint64_t i, f_t font_size, vec2 position);

		vec2 get_length(const std::wstring& text, f_t scale);
		std::vector<vec2> get_length(const std::vector<std::wstring>& texts, const std::vector<f_t>& scales, bool half = false);

		void clear_storage();

		std::vector<std::vector<int>> m_characters;
		std::vector<std::vector<Color>> m_colors;
		std::vector<std::vector<vec2>> m_vertices;
		std::vector<f_t> m_scales;

		static std::array<f_t, 248> widths;

		static suckless_font_t font;

		Shader m_shader;
		unsigned int m_vao, m_vertex_ssbo;
		unsigned int m_texture;
		unsigned int m_text_ssbo;
		unsigned int m_texture_id_ssbo;
		unsigned int m_colors_ssbo;

	};

	namespace text_button {
		constexpr vec2 gap_scale(0.25, 0.25);
		constexpr f_t space_width = 30;
		constexpr f_t space_between_characters = 5;

		constexpr vec2 get_gap_scale(const vec2& size) {
			return size * gap_scale;
		}

		constexpr f_t get_gap_scale_x(f_t width) {
			return width * gap_scale.x;
		}

		constexpr f_t get_gap_scale_y(f_t height) {
			return height * gap_scale.y;
		}

		constexpr f_t get_character_x_offset(f_t width, f_t scale) {
			return width * scale + space_between_characters;
		}

		constexpr f_t get_space(f_t scale) {
			return scale / (font_size / 2) * space_width;
		}

		class basic_text_button_vector : public text_renderer {
		public:

			basic_text_button_vector();

		protected:
			vec2 edit_size(uint64_t i, const std::wstring& text, f_t scale);

			std::vector<std::wstring> m_texts;
		};

		class text_button_vector : public basic_text_button_vector, public fan_2d::square_vector {
		public:

			text_button_vector();

			text_button_vector(const std::wstring& text, const vec2& position, const Color& box_color, f_t font_scale, f_t left_offset, f_t max_width);

			text_button_vector(const std::wstring& text, const vec2& position, const Color& color, f_t scale);
			text_button_vector(const std::wstring& text, const vec2& position, const Color& color, f_t scale, const vec2& box_size);

			void add(const std::wstring& text, const vec2& position, const Color& color, f_t scale);
			void add(const std::wstring& text, const vec2& position, const Color& color, f_t scale, const vec2& box_size);

			void edit_string(uint64_t i, const std::wstring& text, f_t scale);

			vec2 get_string_length(const std::wstring& text, f_t scale);

			f_t get_scale(uint64_t i);

			void set_font_size(uint64_t i, f_t scale);
			void set_position(uint64_t i, const vec2& position);

			void set_press_callback(int key, const std::function<void()>& function);

			void draw();

			bool inside(std::uint64_t i);

		private:

			using fan_2d::square_vector::set_position;
			using fan_2d::square_vector::draw;
			using fan_2d::square_vector::set_size;
		};
	}
}

#define fan_window_loop() \
	while(!glfwWindowShouldClose(window))

void begin_render(const Color& background_color);
void end_render();

inline da_t<f_t, 2> lines_intersection(da_t<f_t, 2, 2> src, da_t<f_t, 2, 2> dst, const da_t<f_t, 2>& normal) {
	f_t s1_x, s1_y, s2_x, s2_y;
	s1_x = src[1][0] - src[0][0]; s1_y = src[1][1] - src[0][1];
	s2_x = dst[1][0] - dst[0][0]; s2_y = dst[1][1] - dst[0][1];

	const f_t s = (-s1_y * (src[0][0] - dst[0][0]) + s1_x * (src[0][1] - dst[0][1])) / (-s2_x * s1_y + s1_x * s2_y);
	const f_t t = (s2_x * (src[0][1] - dst[0][1]) - s2_y * (src[0][0] - dst[0][0])) / (-s2_x * s1_y + s1_x * s2_y);

	if (s < 0 || s > 1 || t < 0 || t > 1)
		return FLT_MAX;

	int signy = sign(normal.gfne());
	if (dcom_fr(signy > 0, src[1][!!normal[1]], dst[0][!!normal[1]]))
		return FLT_MAX;

	da_t<f_t, 2> min = dst.min();
	da_t<f_t, 2> max = dst.max();
	for (uint_t i = 0; i < 2; i++) {
		if (!normal[i])
			continue;
		if (src[0][i ^ 1] == min[i ^ 1])
			return FLT_MAX;
		if (src[0][i ^ 1] == max[i ^ 1])
			return FLT_MAX;
	}

	return { src[0][0] + (t * s1_x), src[0][1] + (t * s1_y) };
}

constexpr da_t<uint_t, 3> GetPointsTowardsVelocity3(da_t<f_t, 2> vel) {
	if (vel[0] >= 0)
		if (vel[1] >= 0)
			return { 2, 1, 3 };
		else
			return { 0, 3, 1 };
	else
		if (vel[1] >= 0)
			return { 0, 3, 2 };
		else
			return { 2, 1, 0 };
}

template <typename T, typename T2>
constexpr auto get_cross(const T& a, const T2& b) {
	return cross(T2{ a[0], a[1], 0 }, b);
}

template <
	template <typename, std::size_t, std::size_t> typename inner_da_t,
	template <typename, std::size_t> typename outer_da_t, std::size_t n
>
constexpr da_t<da_t<f_t, 2>, n> get_normals(const outer_da_t<inner_da_t<f_t, 2, 2>, n>& lines) {
	da_t<da_t<f_t, 2>, n> normals;
	for (int i = 0; i < n; i++) {
		normals[i] = get_cross(lines[i][1] - lines[i][0], da_t<f_t, 3>(0, 0, 1));
	}
	return normals;
}

inline void calculate_velocity(const da_t<f_t, 2>& spos, const da_t<f_t, 2>& svel, const da_t<f_t, 2>& dpos, const da_t<f_t, 2>& dvel, const da_t<f_t, 2>& normal, f_t sign, da_t<f_t, 2>& lvel, da_t<f_t, 2>& nvel) {
	da_t<f_t, 2, 2> sline = { spos, spos + svel };
	da_t<f_t, 2, 2> dline = { dpos, dpos + dvel };
	da_t<f_t, 2> inter = lines_intersection(sline, dline, normal);
	if (inter == FLT_MAX)
		return;
	da_t<f_t, 2> tvel = (inter - spos) * sign;
	if (tvel.abs() >= lvel.abs())
		return;
	nvel = svel * sign - tvel;
	lvel = tvel;
	nvel[0] = normal[1] ? nvel[0] : 0;
	nvel[1] = normal[0] ? nvel[1] : 0;
}

struct collision_info {
	da_t<f_t, 2> position;
	da_t<f_t, 2> velocity;
};

inline void process_rectangle_collision_2d(da_t<f_t, 2, 2>& pos, da_t<f_t, 2>& vel, const std::vector<da_t<f_t, 2, 2>>& walls) {
	while (1) {

		da_t<f_t, 2> pvel = vel;

		if (!pvel[0] && !pvel[1])
			return;

		da_t<f_t, 4, 2> ocorn = get_square_corners(pos);
		da_t<f_t, 4, 2> ncorn = ocorn + pvel;

		da_t<uint_t, 3> ptv3 = GetPointsTowardsVelocity3(pvel);
		da_t<uint_t, 3> ntv3 = GetPointsTowardsVelocity3(-pvel);

		da_t<uint_t, 4, 2> li = { da_t<uint_t, 2>{0, 1}, da_t<uint_t, 2>{1, 3}, da_t<uint_t, 2>{3, 2}, da_t<uint_t, 2>{2, 0} };

		const static da_t<da_t<f_t, 2>, 4> normals = get_normals(
			da_t<da_t<f_t, 2, 2>, 4>{
				da_t<f_t, 2, 2>{da_t<f_t, 2>{ 0, 0 }, da_t<f_t, 2>{ 1, 0 }},
				da_t<f_t, 2, 2>{da_t<f_t, 2>{ 1, 0 }, da_t<f_t, 2>{ 1, 1 }},
				da_t<f_t, 2, 2>{da_t<f_t, 2>{ 1, 1 }, da_t<f_t, 2>{ 0, 1 }},
				da_t<f_t, 2, 2>{da_t<f_t, 2>{ 0, 1 }, da_t<f_t, 2>{ 0, 0 }},
		});

		da_t<f_t, 2> lvel = pvel;
		da_t<f_t, 2> nvel = 0;
		for (uint_t iwall = 0; iwall < walls.size(); iwall++) {
			da_t<f_t, 4, 2> bcorn = get_square_corners(walls[iwall]);

			/* step -1 */
			for (uint_t i = 0; i < 4; i++) {
				for (uint_t iline = 0; iline < 4; iline++) {
					calculate_velocity(da_t<f_t, 2, 2>(ocorn[li[i][0]], ocorn[li[i][1]]).avg(), pvel, bcorn[li[iline][0]], bcorn[li[iline][1]] - bcorn[li[iline][0]], normals[iline], 1, lvel, nvel);
				}
			}

			/* step 0 and step 1*/
			for (uint_t i = 0; i < 3; i++) {
				for (uint_t iline = 0; iline < 4; iline++) {
					calculate_velocity(ocorn[ptv3[i]], ncorn[ptv3[i]] - ocorn[ptv3[i]], bcorn[li[iline][0]], bcorn[li[iline][1]] - bcorn[li[iline][0]], normals[iline], 1, lvel, nvel);
					calculate_velocity(bcorn[ntv3[i]], -pvel, ocorn[li[iline][0]], ocorn[li[iline][1]] - ocorn[li[iline][0]], normals[iline], -1, lvel, nvel);
				}
			}
		}
		pos += lvel;
		vel = nvel;
	}
}

static void rectangle_collision_2d(fan_2d::square& player, const vec2& old_position, const fan_2d::square_vector& walls) {
	mat2x2 pl(
		old_position,
		old_position + player.get_size()
	);

	da_t<f_t, 2> vel = player.get_velocity() * delta_time;

	process_rectangle_collision_2d(
		pl,
		vel,
		walls.get_icorners()
	);
	player.set_position(pl[0]);
	//player.set_velocity(0);
}

constexpr bool rectangles_collide(const vec2& a, const  vec2& a_size, const vec2& b, const vec2& b_size) {
	bool x = a[0] + a_size[0] > b[0] &&
		a[0] < b[0] + b_size[0];
	bool y = a[1] + a_size[1] > b[1] &&
		a[1] < b[1] + b_size[1];
	return x && y;
}

//#endif