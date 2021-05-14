#pragma once
//#ifndef __INTELLISENSE__ 

#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>

#include <fan/graphics/base.hpp>

#include <fan/types/da.hpp>

constexpr f64_t meters_in_pixels = 100;

namespace fan {

	inline fan::vec2 supported_gl_version;

	void depth_test(bool value);

	void print_opengl_version();

	constexpr auto draw_count_table = [](uint_t mode) {
		switch (mode) {
			case GL_LINES:
			{
				return 2;
			}
			case GL_TRIANGLES:
			{
				return 3;
			}
			default:
			{
				return -1;
			}
		}
	};

	static void draw_2d(const std::function<void()>& function_) {
		glDisable(GL_DEPTH_TEST);
		function_();
		glEnable(GL_DEPTH_TEST);
	}

	// editing this requires change in glsl file
	enum class e_shapes {
		VERTICE,
		LINE,
		SQUARE,
		TRIANGLE
	};

}

namespace fan_2d {

	namespace graphics {

		namespace shader_paths {
			constexpr auto text_renderer_vs("glsl/2D/text.vs");
			constexpr auto text_renderer_fs("glsl/2D/text.fs");

			constexpr auto single_shapes_vs("glsl/2D/shapes.vs");
			constexpr auto single_shapes_fs("glsl/2D/shapes.fs");

			constexpr auto single_shapes_bloom_vs("glsl/2D/bloom.vs");
			constexpr auto single_shapes_bloom_fs("glsl/2D/bloom.fs");
			constexpr auto single_shapes_blur_vs("glsl/2D/blur.vs");
			constexpr auto single_shapes_blur_fs("glsl/2D/blur.fs");

			constexpr auto single_shapes_bloom_final_vs("glsl/2D/bloom_final.vs");
			constexpr auto single_shapes_bloom_final_fs("glsl/2D/bloom_final.fs");

			constexpr auto post_processing_vs("glsl/2D/post_processing.vs");
			constexpr auto post_processing_fs("glsl/2D/post_processing.fs");

			constexpr auto shape_vector_vs("glsl/2D/shape_vector.vs");
			constexpr auto shape_vector_fs("glsl/2D/shapes.fs");
		}

		fan::mat4 get_projection(const fan::vec2i& window_size);
		fan::mat4 get_view_translation(const fan::vec2i& window_size, const fan::mat4& view);

		// returns how much object moved
		fan::vec2 move_object(fan::window* window, fan::vec2& position, fan::vec2& velocity, f32_t speed, f32_t gravity, f32_t jump_force = -800, f32_t friction = 10);

		struct rectangle_corners_t {

			fan::vec2 top_left;
			fan::vec2 top_right;
			fan::vec2 bottom_left;
			fan::vec2 bottom_right;

			const fan::vec2 operator[](const uint_t i) const {
				return !i ? top_left : i == 1 ? top_right : i == 2 ? bottom_left : bottom_right;
			}

			fan::vec2& operator[](const uint_t i) {
				return !i ? top_left : i == 1 ? top_right : i == 2 ? bottom_left : bottom_right;
			}

		};

		// 0 left right, 1 top right, 2 bottom left, 3 bottom right

		constexpr rectangle_corners_t get_rectangle_corners(const fan::vec2& position, const fan::vec2& size) {
			return { position, position + fan::vec2(size.x, 0), position + fan::vec2(0, size.y), position + size };
		}

		struct image_info {
			fan::vec2i size;
			uint32_t texture_id;
		};

		namespace image_load_properties {
			inline uint_t internal_format = GL_RGBA;
			inline uint_t format = GL_RGBA;
			inline uint_t type = GL_UNSIGNED_BYTE;
			inline uint_t filter = GL_LINEAR;
		}

		image_info load_image(const std::string& path, bool flip_image = false);
		image_info load_image(uint32_t texture_id, const std::string& path, bool flip_image = false);
		image_info load_image(unsigned char* pixels, const fan::vec2i& size);

		class vertice_vector : public fan::basic_vertice_vector<fan::vec2>, public fan::ebo_handler<> {
		public:

			static constexpr auto position_location_name = "position";
			static constexpr auto color_location_name = "in_color";

			vertice_vector(fan::camera* camera, uint_t index_restart = UINT32_MAX);
			vertice_vector(fan::camera* camera, const fan::vec2& position, const fan::color& color, uint_t index_restart);
			vertice_vector(const vertice_vector& vector);
			vertice_vector(vertice_vector&& vector) noexcept;

			vertice_vector& operator=(const vertice_vector& vector);
			vertice_vector& operator=(vertice_vector&& vector) noexcept;

			void release_queue(bool position, bool color, bool indices);

			virtual void push_back(const fan::vec2& position, const fan::color& color, bool queue = false);

			void reserve(uint_t size);
			void resize(uint_t size, const fan::color& color);

			virtual void draw(uint32_t mode, uint32_t single_draw_amount, uint_t begin = fan::uninitialized, uint_t end = fan::uninitialized) const;

			void erase(uint_t i, bool queue = false);
			void erase(uint_t begin, uint_t end, bool queue = false);

			void initialize_buffers();

		protected:

			void write_data();

			uint_t m_index_restart;

			std::vector<uint32_t> m_indices;

			uint_t m_offset;

		};

		struct line : protected fan_2d::graphics::vertice_vector {

			line(fan::camera* camera);

			line(const line& line_);
			line(line&& line_) noexcept;

			line& operator=(const line& line_);
			line& operator=(line&& line_) noexcept;

			fan::mat2 get_line(uint_t i) const;
			void set_line(uint_t i, const fan::vec2& start, const fan::vec2& end, bool queue = false);

			void push_back(const fan::vec2& start, const fan::vec2& end, const fan::color& color, bool queue = false);

			void reserve(uint_t size);
			void resize(uint_t size, const fan::color& color);

			void draw(uint_t i = fan::uninitialized) const;

			void erase(uint_t i, bool queue = false);
			void erase(uint_t begin, uint_t end, bool queue = false);

			const fan::color get_color(uint_t i) const;
			void set_color(uint_t i, const fan::color& color);

			void release_queue(bool line, bool color);

			uint_t size() const;

		};

		fan_2d::graphics::line create_grid(fan::camera* camera, const fan::vec2i& block_size, const fan::vec2i& grid_size, const fan::color& color);

		struct rectangle : protected fan_2d::graphics::vertice_vector{

			rectangle(fan::camera* camera);

			rectangle(const rectangle& rectangle_);
			rectangle(rectangle&& rectangle_) noexcept;

			rectangle& operator=(const rectangle& rectangle_);
			rectangle& operator=(rectangle&& rectangle_) noexcept;

			void push_back(const fan::vec2& position, const fan::vec2& size, const fan::color& color, bool queue = false);

			void reserve(uint_t size);
			void resize(uint_t size, const fan::color& color);

			void draw(uint_t begin = fan::uninitialized, uint_t end = fan::uninitialized);

			void erase(uint_t i, bool queue = false);
			void erase(uint_t begin, uint_t end, bool queue = false);

			rectangle_corners_t get_corners(uint_t i = 0) const;

			fan::vec2 get_center(uint_t i = 0) const;

			f_t get_rotation(uint_t i = 0) const;
			void set_rotation(uint_t i, f_t angle, bool queue = false);

			const fan::color get_color(uint_t i = 0) const;
			void set_color(uint_t i, const fan::color& color, bool queue = false);

			fan::vec2 get_position(uint_t i = 0) const;
			void set_position(uint_t i, const fan::vec2& position, bool queue = false);

			fan::vec2 get_size(uint_t i = 0) const;
			void set_size(uint_t i, const fan::vec2& size, bool queue = false);

			fan::vec2 move(uint_t i, f32_t speed, f32_t gravity = 0, f32_t jump_force = -800, f32_t friction = 10);

			uint_t size() const;

			void release_queue(bool rectangle, bool color);

			bool inside(uint_t i) const;

			fan::vec2 get_vertices(uint32_t i) const;

			using fan_2d::graphics::vertice_vector::get_velocity;
			using fan_2d::graphics::vertice_vector::set_velocity;
			using fan_2d::graphics::vertice_vector::m_camera;

		protected:

			std::vector<fan::vec2> m_position;
			std::vector<fan::vec2> m_size;

			std::vector<f_t> m_rotation;

		};

		class rounded_rectangle : public fan_2d::graphics::vertice_vector {
		public:

			static constexpr int m_segments = 10;

			rounded_rectangle(fan::camera* camera);
			rounded_rectangle(fan::camera* camera, const fan::vec2& position, const fan::vec2& size, f_t radius, const fan::color& color);

			void push_back(const fan::vec2& position, const fan::vec2& size, f_t radius, const fan::color& color, bool queue = false);

			fan::vec2 get_position(uint_t i) const;
			void set_position(uint_t i, const fan::vec2& position, bool queue = false);

			fan::vec2 get_size(uint_t i) const;
			void set_size(uint_t i, const fan::vec2& size, bool queue = false);

			f_t get_radius(uint_t i) const;
			void set_radius(uint_t i, f_t radius, bool queue = false);

			void draw() const;

			bool inside(uint_t i) const;

			fan::color get_color(uint_t i) const;
			void set_color(uint_t i, const fan::color& color, bool queue = false);

			uint_t size() const;

		private:

			using fan_2d::graphics::vertice_vector::push_back;

			void edit_rectangle(uint_t i, bool queue = false);

			std::vector<fan::vec2> m_position;
			std::vector<fan::vec2> m_size;
			std::vector<f_t> m_radius;

			std::vector<uint_t> m_data_offset;

		};

		class circle : public fan_2d::graphics::vertice_vector{
		public:

			circle(fan::camera* camera);

			void push_back(const fan::vec2& position, f32_t radius, const fan::color& color, bool queue = false);

			fan::vec2 get_position(uint_t i) const;
			void set_position(uint_t i, const fan::vec2& position, bool queue = false);

			f32_t get_radius(uint_t i) const;
			void set_radius(uint_t i, f32_t radius, bool queue = false);

			void draw() const;

			bool inside(uint_t i) const;

			fan::color get_color(uint_t i) const;
			void set_color(uint_t i, const fan::color& color, bool queue = false);

			uint_t size() const;

			void erase(uint_t i, bool queue = false);
			void erase(uint_t begin, uint_t end, bool queue = false);

		protected:

			static constexpr int m_segments = 50;

			std::vector<fan::vec2> m_position;
			std::vector<f32_t> m_radius;

		};

		class sprite :
			protected fan_2d::graphics::rectangle,
			public fan::texture_handler<1>, // screen texture
			public fan::render_buffer_handler<>,
			public fan::frame_buffer_handler<> {

		public:

			sprite(fan::camera* camera);

			//size with default is the size of the image 
			sprite(fan::camera* camera, const std::string& path, const fan::vec2& position, const fan::vec2& size = 0);
			sprite(fan::camera* camera, unsigned char* pixels, const fan::vec2& position, const fan::vec2i& size);
			sprite(fan::camera* camera, uint32_t texture_id, const fan::vec2& position, const fan::vec2& size);

			sprite(const fan_2d::graphics::sprite& sprite);
			sprite(fan_2d::graphics::sprite&& sprite) noexcept;



			fan_2d::graphics::sprite& operator=(const fan_2d::graphics::sprite& sprite);
			fan_2d::graphics::sprite& operator=(fan_2d::graphics::sprite&& sprite);

			~sprite();

			void reload_sprite(uint32_t i, const std::string& path, const fan::vec2& size = 0);
			void reload_sprite(uint32_t i, unsigned char* pixels, const fan::vec2i& size);

			void push_back(const fan::vec2& position, const fan::vec2& size);
			void push_back(uint32_t texture_id, const fan::vec2& position, const fan::vec2& size);

			void draw(uint_t begin = fan::uninitialized, uint_t end = fan::uninitialized);

			using fan_2d::graphics::rectangle::get_corners;
			using fan_2d::graphics::rectangle::get_size;
			using fan_2d::graphics::rectangle::set_size;
			using fan_2d::graphics::rectangle::get_position;
			using fan_2d::graphics::rectangle::get_positions;
			using fan_2d::graphics::rectangle::set_position;
			using fan_2d::graphics::rectangle::get_velocity;
			using fan_2d::graphics::rectangle::set_velocity;
			using fan_2d::graphics::rectangle::release_queue;
			using fan_2d::graphics::rectangle::get_rotation;
			using fan_2d::graphics::rectangle::set_rotation;
			using fan_2d::graphics::rectangle::get_center;


		private:

			void initialize_buffers(const fan::vec2& size);

			fan::shader m_screen_shader;

			std::vector<f_t> m_transparency;

			std::vector<uint32_t> m_textures;

			std::vector<uint32_t> m_texture_offsets;

			uint32_t m_amount_of_textures;

		};


		struct particle {
			fan::vec2 m_velocity;
			fan::timer<> m_timer; // milli
		};

		class particles : public fan_2d::graphics::rectangle {
		public:

			particles(fan::camera* camera);

			void add(
				const fan::vec2& position,
				const fan::vec2& size,
				const fan::vec2& velocity,
				const fan::color& color,
				uint_t time
			);

			void update();

		private:

			std::vector<fan_2d::graphics::particle> m_particles;
		};

	}
}

namespace fan_3d {

	namespace graphics {

		namespace shader_paths {
			constexpr auto triangle_vector_vs("glsl/3D/terrain_generator.vs");
			constexpr auto triangle_vector_fs("glsl/3D/terrain_generator.fs");

			constexpr auto shape_vector_vs("glsl/3D/shape_vector.vs");
			constexpr auto shape_vector_fs("glsl/3D/shape_vector.fs");

			constexpr auto model_vs("glsl/3D/models.vs");
			constexpr auto model_fs("glsl/3D/models.fs");

			constexpr auto animation_vs("glsl/3D/animation.vs");
			constexpr auto animation_fs("glsl/3D/animation.fs");

			constexpr auto skybox_vs("glsl/3D/skybox.vs");
			constexpr auto skybox_fs("glsl/3D/skybox.fs");
			constexpr auto skybox_model_vs("glsl/3D/skybox_model.vs");
			constexpr auto skybox_model_fs("glsl/3D/skybox_model.fs");
		}

		void add_camera_rotation_callback(fan::camera* camera);

		using triangle_vertices_t = fan::vec3;

		class terrain_generator : public fan::basic_shape_color_vector<true> {
		public:

			terrain_generator(fan::camera* camera, const std::string& path, const f32_t texture_scale, const fan::vec3& position, const fan::vec2ui& map_size, f_t triangle_size, const fan::vec2& mesh_size);
			~terrain_generator();

			void insert(const std::vector<triangle_vertices_t>& vertices, const std::vector<fan::color>& color, bool queue = false);
			void push_back(const triangle_vertices_t& vertices, const fan::color& color, bool queue = false);

			template <uint_t i = uint_t(-1)>
			std::conditional_t<i == -1, std::vector<triangle_vertices_t>, triangle_vertices_t> get_vertices();

			void edit_data(uint_t i, const triangle_vertices_t& vertices, const fan::color& color);

			void release_queue();

			void draw();

			void erase_all();

			uint_t size();

			fan::camera* m_camera;

		private:

			fan::shader m_shader;

			uint32_t m_texture;
			uint32_t m_texture_vbo;
			uint32_t m_vao;
			uint32_t m_vertices_vbo;
			uint32_t m_ebo;
			uint32_t m_triangle_size;
			uint32_t m_normals_vbo;

			//	fan_3d::line_vector lv;

			std::vector<triangle_vertices_t> m_triangle_vertices;
			std::vector<unsigned int> m_indices;
			static constexpr auto m_vertice_size = sizeof(triangle_vertices_t);

		};

		struct plane_corners {
			fan::da_t<f32_t, 3> top_left;
			fan::da_t<f32_t, 3> top_right;
			fan::da_t<f32_t, 3> bottom_left;
			fan::da_t<f32_t, 3> bottom_right;
		};

		struct square_corners {

			plane_corners left;
			plane_corners right;
			plane_corners front;
			plane_corners back;
			plane_corners top;
			plane_corners bottom;

		};

		class rectangle_vector : public fan::basic_shape<true, fan::vec3>, public fan::texture_handler<> {
		public:

			rectangle_vector(fan::camera* camera, const std::string& path, uint_t block_size);
			//rectangle_vector(fan::camera* camera, const fan::color& color, uint_t block_size);
			~rectangle_vector();

			void push_back(const fan::vec3& src, const fan::vec3& dst, const fan::vec2& texture_id, bool queue = false);

			fan::vec3 get_src(uint_t i) const;
			fan::vec3 get_dst(uint_t i) const;
			fan::vec3 get_size(uint_t i) const;

			void set_position(uint_t i, const fan::vec3& src, const fan::vec3& dst, bool queue = false);
			void set_size(uint_t i, const fan::vec3& size, bool queue = false);

			void draw();

			void set_texture(uint_t i, const fan::vec2& texture_id, bool queue = false);

			void generate_textures(const std::string& path, const fan::vec2& block_size);

			void write_textures();

			void release_queue(bool position, bool size, bool textures);

			square_corners get_corners(uint_t i) const;

			uint_t size() const;

		private:

			unsigned int m_texture_ssbo;
			unsigned int m_texture_id_ssbo;

			fan::vec2i block_size;
			fan::vec2i m_amount_of_textures;

			std::vector<int> m_textures;

		};

		class skybox {
		public:
			skybox(
				fan::camera* camera,
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
			unsigned int m_texture_id;
			unsigned int m_skybox_vao, m_skybox_vbo;

			fan::shader m_shader;
			fan::camera* m_camera;

		};

		struct mesh_vertex {
			fan::vec3 position;
			fan::vec3 normal;
			fan::vec2 texture_coordinates;
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

		/*class model_loader {
		protected:
		model_loader(const std::string& path, const fan::vec3& size);

		std::vector<model_mesh> meshes;
		std::vector<mesh_texture> textures_loaded;
		private:
		void load_model(const std::string& path, const fan::vec3& size);

		void process_node(aiNode* node, const aiScene* scene, const fan::vec3& size);

		model_mesh process_mesh(aiMesh* mesh, const aiScene* scene, const fan::vec3& size);

		std::vector<mesh_texture> load_material_textures(aiMaterial* mat, aiTextureType type, const std::string& type_name);

		std::string directory;
		};*/

		/*class model : public model_loader {
		public:
		model(fan::camera* camera);
		model(fan::camera* camera, const std::string& path, const fan::vec3& position, const fan::vec3& size);

		void draw();

		fan::vec3 get_position();
		void set_position(const fan::vec3& position);

		fan::vec3 get_size();
		void set_size(const fan::vec3& size);

		fan::camera* m_camera;

		private:
		fan::shader m_shader;

		fan::vec3 m_position;
		fan::vec3 m_size;

		};*/

		fan::vec3 line_triangle_intersection(const fan::da_t<f32_t, 2, 3>& line, const fan::da_t<f32_t, 3, 3>& triangle);
		fan::vec3 line_plane_intersection(const fan::da_t<f32_t, 2, 3>& line, const fan::da_t<f32_t, 4, 3>& square);

		template<uint_t i>
		inline std::conditional_t<i == -1, std::vector<triangle_vertices_t>, triangle_vertices_t> terrain_generator::get_vertices()
		{
			if constexpr(i == (uint_t)-1) {
				return fan_3d::graphics::terrain_generator::m_triangle_vertices;
			}
			else {
				return fan_3d::graphics::terrain_generator::m_triangle_vertices[i];
			}
		}

	}

}