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
	unsigned int VBO, VAO;
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

extern Shader shape_shader2d;
extern Camera camera3d;

class default_2d_base_vector {
public:

	default_2d_base_vector();

	template <typename _Type, uint64_t N>
	default_2d_base_vector(const std::array<_Type, N>& init_vertices);

	~default_2d_base_vector();

	void free_queue();

	vec2 get_position(uint64_t index) const;
	void set_position(uint64_t index, const vec2& position, bool queue = false);

	vec2 get_size(uint64_t index) const;
	void set_size(uint64_t index, const vec2& size, bool queue = false);

	void realloc_copy_data(unsigned int& buffer, uint64_t buffer_type, int size, GLenum usage, unsigned int& allocator) const;
	void copy_data(unsigned int& buffer, uint64_t buffer_type, int size, GLenum usage, unsigned int& allocator) const;
	void realloc_buffer(unsigned int& buffer, uint64_t buffer_type, int location, int size, GLenum usage, unsigned int& allocator) const;

	int size() const;

	virtual void erase(uint64_t first, uint64_t last = -1);

protected:

	void basic_shape_draw(unsigned int mode, uint64_t points, const Shader& shader = shape_shader2d) const;

	unsigned int shape_vao;
	unsigned int matrix_vbo;
	unsigned int vertex_vbo;

	unsigned int matrix_allocator_vbo;
	unsigned int matrix_allocator_size;

	bool matrix_allocated;

	uint64_t shapes_size;

	static constexpr auto copy_buffer = 5000000; // * sizeof(type)
};

class basic_2dshape_vector : public default_2d_base_vector {
public:

	basic_2dshape_vector();

	template <typename _Type, uint64_t N>
	basic_2dshape_vector(const std::array<_Type, N>& init_vertices);

	~basic_2dshape_vector();

	Color get_color(uint64_t index) const;
	void set_color(uint64_t index, const Color& color, bool queue);

	void free_queue(bool colors = true, bool matrices = true);

	void erase(uint64_t first, uint64_t last = -1) override;

	void push_back(const vec2& position, const vec2& size, const Color& color, bool queue = false);

protected:

	unsigned int color_vbo;
	unsigned int color_allocator_vbo;
	unsigned int color_allocator_size;

	bool color_allocated;

private:
	using default_2d_base_vector::erase;
};

class line_vector2d : public basic_2dshape_vector {
public:

	line_vector2d();
	line_vector2d(const mat2& position, const Color& color);

	void push_back(const mat2& position, const Color& color, bool queue = false);

	mat2 get_position(uint64_t index) const;
	void set_position(uint64_t index, const mat2& position, bool queue = false);

	void draw() const;

private:

	using basic_2dshape_vector::get_size;
	using basic_2dshape_vector::set_size;
	using basic_2dshape_vector::get_position;
	using basic_2dshape_vector::set_position;

};

struct square_vector2d : basic_2dshape_vector {

	square_vector2d();
	square_vector2d(const vec2& position, const vec2& size, const Color& color);

	void draw();

};

int load_texture(const std::string_view path, const std::string& directory = std::string(), bool flip_image = false, bool alpha = false);

class sprite_vector2d : public default_2d_base_vector {
public:

	sprite_vector2d();
	sprite_vector2d(const char* path, const vec2& position, const vec2& size);

	~sprite_vector2d();

	void free_queue(bool textures, bool matrices);
	void draw() const;

	void push_back(const vec2& position, const vec2& size, bool queue = false);

	void erase(uint64_t first, uint64_t last = -1) override;

private:

	using default_2d_base_vector::free_queue;
	using default_2d_base_vector::erase;

	unsigned int texture_ssbo;
	unsigned int texture_allocator;
	uint64_t texture_allocator_size;
	bool texture_allocated;

	unsigned int texture_id;
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

struct Line;
class Square;

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
	Camera* camera;
protected:
	Shader shader;
	Texture texture;
	vec2 position;
	vec2 size;
	float angle;
	std::vector<float> _Vertices;
};

//class LineVector3D : public basic_3d {
//public:
//	LineVector3D();
//	LineVector3D(const matrix<3, 2>& _M, const Color& color);
//
//	matrix<2, 3> get_position(uint64_t _Index = 0) const;
//	void set_position(uint64_t _Index, const matrix<3, 2>& _M, bool _Queue = false);
//
//	void push_back(const matrix<3, 2>& _M, Color _Color = Color(-1, -1, -1, -1), bool _Queue = false);
//
//	vec2 get_length(uint64_t _Index = 0) const;
//	
//	void draw() {
//
//		if (_Vertices.empty()) {
//			return;
//		}
//		_Shader.use();
//
//		mat4 projection(1);
//		mat4 view(1);
//
//		view = _Camera->get_view_matrix();
//
//		projection = Perspective(Radians(90.f), ((float)window_size.x / (float)window_size.y), 0.1f, 1000.0f);
//
//		static int projLoc = glGetUniformLocation(_Shader.ID, "projection");
//		static int viewLoc = glGetUniformLocation(_Shader.ID, "view");
//		glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection[0][0]);
//		glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view[0][0]);
//		glBindVertexArray(_ShapeBuffer.VAO);
//		glBindBuffer(GL_ARRAY_BUFFER, _VerticeBuffer.VBO);
//		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
//		glBindBuffer(GL_ARRAY_BUFFER, _ColorBuffer.VBO);
//		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
//		glDrawArrays(_Mode, 0, _Points);
//		glBindVertexArray(0);
//	}
//
//};

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
	basic_3d() : _Shape_Matrix_VAO(0), _Shape_Matrix_VBO(0) {}

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

#include <map>

struct Character {
	GLuint TextureID;
	vec2i Size;
	vec2i Bearing;
	GLuint Advance;
};

class TextRenderer {
public:
	TextRenderer();

	vec2 get_length(std::string text, float scale, bool include_endl = false);

	void render(const std::string& text, vec2 position, float_t scale, const Color& color);

private:
	std::vector<Character> Characters;
	//std::map<GLchar, Character> Characters;
	Shader shader;
	unsigned int VAO, VBO;

	static constexpr float_t text_gap = 10;
};

//#endif