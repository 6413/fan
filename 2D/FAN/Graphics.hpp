#pragma once
//#ifndef __INTELLISENSE__ 

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

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

constexpr auto WINDOWSIZE = vec2i(800, 800);
extern float delta_time;
static constexpr int block_size = 50;
extern GLFWwindow* window;
constexpr auto grid_size = vec2i(WINDOWSIZE.x / block_size, WINDOWSIZE.y / block_size);

typedef std::vector<std::vector<std::vector<bool>>> map_t;

extern bool is_colliding;

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

	_Vector get_position(std::uint64_t i) const;
	void set_position(std::uint64_t i, const _Vector& position, bool queue = false);

	void basic_push_back(const _Vector& position, const _Vector& size, bool queue = false);

	void erase(std::uint64_t i);

	std::uint64_t size() const;

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

		vec2 get_position();
		vec2 get_size();
		vec2 get_velocity();

		void set_size(const vec2& size);
		void set_position(const vec2& position);
		void set_velocity(const vec2& velocity);

		void basic_draw(GLenum mode, GLsizei count);

		void move(f_t speed, f_t gravity, f_t friction = 10);

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

		void draw();

		f_t get_rotation();
		void set_rotation(f_t degrees);

		static image_info load_image(const std::string& path, bool flip_image = false);

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

void begin_render(const Color& background_color);
void end_render();

static bool rectangles_collide(const vec2& a, const vec2& a_size, const vec2& b, const vec2& b_size) {
	bool x = a.x + a_size.x / 2 > b.x - b_size.x / 2 &&
		a.x - a_size.x / 2 < b.x + b_size.x / 2;
	bool y = a.y + a_size.y / 2 > b.y - b_size.y / 2 &&
		a.y - a_size.y / 2 < b.y + b_size.y / 2;
	return x && y;
}

static bool rectangles_collide(
	f_t la_x, f_t la_y,
	f_t ra_x, f_t ra_y,
	f_t lb_x, f_t lb_y,
	f_t rb_x, f_t rb_y
)
{
	bool x = ra_x > lb_x && la_x < rb_x;
	bool y = ra_y > lb_y && la_y < rb_y;
	return x && y;
}

inline std::array<bool, 4> get_sides(f_t angle, const vec2& position, const vec2& size) {
	const vec2 top_left = position - size / 2;
	const vec2 top_right = position + vec2(size.x / 2, -size.y / 2);
	const vec2 bottom_left = position - vec2(size.x / 2, -size.y / 2);
	const vec2 bottom_right = position + size;

	const f_t atop_left = Degrees(AimAngle(top_left, position));
	const f_t atop_right = Degrees(AimAngle(top_right, position));
	const f_t abottom_left = Degrees(AimAngle(bottom_left, position));
	const f_t abottom_right = Degrees(AimAngle(bottom_right, position));

	return {
		angle <= atop_left &&
		angle >= abottom_left,
		angle >= atop_right ||
		angle <= abottom_right,
		angle > atop_left &&
		angle < atop_right,
		angle <  abottom_left&&
		angle >  abottom_right
	};
}

inline bool point_inside_square(const vec2& point, const vec2& square, const vec2& size) {
	return
		point.x > square.x - size.x / 2 &&
		point.x < square.x + size.x / 2 &&
		point.y > square.y - size.y / 2 &&
		point.y < square.y + size.y / 2;
}

inline mat4x2 square_corners(const vec2& position, const vec2& size) {
	return mat4x2(
		position - size / 2,
		position + vec2(size.x / 2, -size.y / 2),
		position + vec2(-size.x / 2, size.y / 2),
		position + vec2(size.x / 2, size.y / 2)
	);
}

struct collision_info {
	vec2 position;
	vec2 velocity;
};

constexpr auto NO_COLLISION(-1);

inline bool colliding(const vec2& result) {
	return result != NO_COLLISION;
}

inline bool colliding(const collision_info& result) {
	return result.position != NO_COLLISION;
}

inline collision_info rectangle_collision_2d(const vec2& old_position, const vec2& new_position, const vec2& player_size, const vec2& player_velocity, const fan_2d::square_vector& walls) {

	std::vector<collision_info> possible_collisions;

	for (int iwall = 0; iwall < walls.size(); iwall++) {
		vec2 wall_position = walls.get_position(iwall);
		vec2 wall_size = walls.get_size(iwall);

		f_t angle = Degrees(AimAngle(old_position, walls.get_position(iwall)));

		std::array<bool, 4> sides = get_sides(angle, wall_position, wall_size);

		auto corners = square_corners(wall_position, wall_size);

		constexpr std::pair<int, int> corner_order[] = { {0, 1}, {1, 3}, {3, 2} , {2, 0} };

		for (int icorner = 0; icorner < 4; icorner++) {
			vec2 point = IntersectionPoint(old_position, new_position, vec2(corners[corner_order[icorner].first]), vec2(corners[corner_order[icorner].second]), false);
			if (ray_hit(point)) {

				f_t intersection_angle = Degrees(AimAngle(point, walls.get_position(iwall)));

				std::array<bool, 4> intersection_sides = get_sides(intersection_angle, wall_position, wall_size);

				for (int iside = 0; iside < 4; iside++) {
					if (intersection_sides[iside] != sides[iside]) {
						goto g_skip_side;
					}
				}

				possible_collisions.push_back({ vec2(
					sides[0] ? point.x - player_size.x / 2 : sides[1] ?
					point.x + player_size.x / 2 :
					old_position.x, sides[2] ? point.y - player_size.y / 2 : sides[3] ?
					point.y + player_size.y / 2 : old_position.y
				), vec2((sides[0] || sides[1] ? 0 : player_velocity.x), (sides[2] || sides[3] ? 0 : player_velocity.y)) });
			}
		g_skip_side:;
		}

		if (rectangles_collide(old_position, player_size, wall_position, wall_size)) {
			possible_collisions.push_back({ vec2(
				sides[0] ? wall_position.x - wall_size.x / 2 - player_size.x / 2 : sides[1] ?
				wall_position.x + wall_size.x / 2 + player_size.x / 2 :
				old_position.x, sides[2] ? wall_position.y - wall_size.y / 2 - player_size.y / 2 : sides[3] ?
				wall_position.y + wall_size.y / 2 + player_size.y / 2 : old_position.y
			), vec2((sides[0] || sides[1] ? 0 : player_velocity.x), (sides[2] || sides[3] ? 0 : player_velocity.y)) });
		}
	}

	auto closest = std::min_element(possible_collisions.begin(), possible_collisions.end(),
		[&](const collision_info& a, const collision_info& b) {
			return Distance(old_position, a.position) < Distance(old_position, b.position);
		});

	if (!possible_collisions.empty() && closest != possible_collisions.end()) {
		auto c_info = possible_collisions[std::distance(possible_collisions.begin(), closest)];
		return *closest;
	}

	return { NO_COLLISION };
}

//#endif