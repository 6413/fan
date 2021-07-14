#pragma once

#include <fan/graphics/renderer.hpp>

#if fan_renderer == fan_renderer_opengl

#include <fan/graphics/opengl/gl_core.hpp>

#include <fan/types/da.hpp>

#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>

#include <fan/graphics/shared_graphics.hpp>

#ifdef fan_platform_windows
	#pragma comment(lib, "lib/assimp/assimp.lib")
#endif

constexpr f32_t meter_scale = 100;

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
}

namespace fan_2d {

	namespace graphics {

		namespace global_vars {
			inline uint8_t line_thickness = 1;
		}

		static void draw(const std::function<void()>& function_) {
			fan::depth_test(false);
			function_();
			fan::depth_test(true);
		}

		namespace shader_paths {
			constexpr auto text_renderer_vs("glsl/2D/opengl/text.vs");
			constexpr auto text_renderer_fs("glsl/2D/opengl/text.fs");

			constexpr auto single_shapes_vs("glsl/2D/opengl/shapes.vs");
			constexpr auto single_shapes_fs("glsl/2D/opengl/shapes.fs");

			constexpr auto single_shapes_bloom_vs("glsl/2D/opengl/bloom.vs");
			constexpr auto single_shapes_bloom_fs("glsl/2D/opengl/bloom.fs");
			constexpr auto single_shapes_blur_vs("glsl/2D/opengl/blur.vs");
			constexpr auto single_shapes_blur_fs("glsl/2D/opengl/blur.fs");

			constexpr auto single_shapes_bloom_final_vs("glsl/2D/opengl/bloom_final.vs");
			constexpr auto single_shapes_bloom_final_fs("glsl/2D/opengl/bloom_final.fs");

			constexpr auto post_processing_vs("glsl/2D/opengl/post_processing.vs");
			constexpr auto post_processing_fs("glsl/2D/opengl/post_processing.fs");

			constexpr auto shape_vector_vs("glsl/2D/opengl/shape_vector.vs");
			constexpr auto shape_vector_fs("glsl/2D/opengl/shapes.fs");

			constexpr auto particles_vs("glsl/2D/opengl/particles.vs");

			constexpr auto rectangle_vs("glsl/2D/opengl/rectangle.vs");
			constexpr auto rectangle_fs("glsl/2D/opengl/rectangle.fs");

			constexpr auto sprite_vs("glsl/2D/opengl/sprite.vs");
			constexpr auto sprite_fs("glsl/2D/opengl/sprite.fs");
		}

		fan::mat4 get_projection(const fan::vec2i& window_size);
		fan::mat4 get_view_translation(const fan::vec2i& window_size, const fan::mat4& view);

		// returns how much object moved
		fan::vec2 move_object(fan::window* window, fan::vec2& position, fan::vec2& velocity, f32_t speed, f32_t gravity, f32_t jump_force = -800, f32_t friction = 10);


		// 0 left right, 1 top right, 2 bottom left, 3 bottom right

		namespace image_load_properties {
			inline uint32_t visual_output = GL_REPEAT;
			inline uint_t internal_format = GL_RGBA;
			inline uint_t format = GL_RGBA;
			inline uint_t type = GL_UNSIGNED_BYTE;
			inline uint_t filter = GL_LINEAR;
		}

		// fan::get_device(window)
		image_info load_image(fan::window* window, const std::string& path);

		class lighting_properties {
		public:

			lighting_properties(fan::shader* shader);

			bool get_lighting() const;
			void set_lighting(bool value);

			f32_t get_world_light_strength() const;
			void set_world_light_strength(f32_t value);

		protected:

			bool m_lighting_on;

			f32_t m_world_light_strength;

			fan::shader* m_shader;

		};

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

			virtual void push_back(const fan::vec2& position, const fan::color& color);

			void reserve(uint_t size);
			void resize(uint_t size, const fan::color& color);

			virtual void draw(uint32_t mode, uint32_t single_draw_amount, uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized, bool texture = false) const;

			void erase(uint_t i);
			void erase(uint_t begin, uint_t end);

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
			void set_line(uint_t i, const fan::vec2& start, const fan::vec2& end);

			void push_back(const fan::vec2& start, const fan::vec2& end, const fan::color& color);

			void reserve(uint_t size);
			void resize(uint_t size, const fan::color& color);

			void draw(uint_t i = fan::uninitialized) const;

			void erase(uint_t i);
			void erase(uint_t begin, uint_t end);

			const fan::color get_color(uint_t i) const;
			void set_color(uint_t i, const fan::color& color);

			void release_queue(bool line, bool color);

			uint_t size() const;

			static void set_thickness(f32_t thickness);
		};

		fan_2d::graphics::line create_grid(fan::camera* camera, const fan::vec2i& block_size, const fan::vec2i& grid_size, const fan::color& color);

		enum class draw_mode {
			no_draw,
			draw
		};

		struct rectangle :
			protected fan::shader,
			protected fan::buffer_object<fan::color, 0>,
			protected fan::buffer_object<fan::vec2, 1>,
			protected fan::buffer_object<fan::vec2, 2>,
			protected fan::buffer_object<f32_t, 3>,
			public fan::buffer_object<uint32_t, 0, true, fan::opengl_buffer_type::buffer_object, false, GL_ELEMENT_ARRAY_BUFFER>,
			public fan::vao_handler<>,
			public lighting_properties
			//protected fan::buffer_object<uint32_t, 0, true, fan::opengl_buffer_type::buffer_object, false, GL_ELEMENT_ARRAY_BUFFER> 
		{

			using color_t = fan::buffer_object<fan::color, 0>;
			using position_t = fan::buffer_object<fan::vec2, 1>;
			using size_t = fan::buffer_object<fan::vec2, 2>;
			using angle_t = fan::buffer_object<f32_t, 3>;

			using ebo_t = fan::buffer_object<uint32_t, 0, true, fan::opengl_buffer_type::buffer_object, false, GL_ELEMENT_ARRAY_BUFFER>;

			rectangle();

			rectangle(fan::camera* camera);

			void push_back(const fan::vec2& position, const fan::vec2& size, const fan::color& color, f32_t angle = 0);

			void insert(uint32_t i, const fan::vec2& position, const fan::vec2& size, const fan::color& color, f32_t angle = 0);

			void reserve(uint32_t size);
			void resize(uint32_t size, const fan::color& color);

			void draw(uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized) const;

			void erase(uint32_t i);
			void erase(uint32_t begin, uint32_t end);

			// erases everything
			void clear();

			rectangle_corners_t get_corners(uint32_t i = 0) const;

			f32_t get_angle(uint32_t i = 0) const;
			void set_angle(uint32_t i, f32_t angle);

			const fan::color get_color(uint32_t i = 0) const;
			void set_color(uint32_t i, const fan::color& color);

			fan::vec2 get_position(uint32_t i = 0) const;
			void set_position(uint32_t i, const fan::vec2& position);

			fan::vec2 get_size(uint32_t i = 0) const;
			void set_size(uint32_t i, const fan::vec2& size);

			uint_t size() const;

			// used when editing data, not push_back
			void release_queue(uint32_t avoid_flags = 0);

			// used after push_back
			void write_data();

			bool inside(uint_t i, const fan::vec2& position = fan::math::inf) const;

			uint32_t* get_vao();

			fan::camera* m_camera = nullptr;

			fan::shader* get_shader();

		protected:

			rectangle(const fan::shader& shader);
			rectangle(fan::camera* camera, const fan::shader& shader);

			static constexpr auto location_color = "layout_color";
			static constexpr auto location_position = "layout_position";
			static constexpr auto location_size = "layout_size";
			static constexpr auto location_angle = "layout_angle";

			static constexpr uint32_t primitive_restart = (uint32_t)-1;

			uint32_t queue_flag = 0;

		};

		class rounded_rectangle : public fan_2d::graphics::vertice_vector {
		public:

			static constexpr int m_segments = 10;

			rounded_rectangle(fan::camera* camera);
			rounded_rectangle(fan::camera* camera, const fan::vec2& position, const fan::vec2& size, f_t radius, const fan::color& color);

			void push_back(const fan::vec2& position, const fan::vec2& size, f_t radius, const fan::color& color);

			fan::vec2 get_position(uint_t i) const;
			void set_position(uint_t i, const fan::vec2& position);

			fan::vec2 get_size(uint_t i) const;
			void set_size(uint_t i, const fan::vec2& size);

			f_t get_radius(uint_t i) const;
			void set_radius(uint_t i, f_t radius);

			void draw() const;

			bool inside(uint_t i) const;

			fan::color get_color(uint_t i) const;
			void set_color(uint_t i, const fan::color& color);

			uint_t size() const;

		private:

			using fan_2d::graphics::vertice_vector::push_back;

			void edit_rectangle(uint_t i);

			std::vector<fan::vec2> m_position;
			std::vector<fan::vec2> m_size;
			std::vector<f_t> m_radius;

			std::vector<uint_t> m_data_offset;

		};

		class circle : public fan_2d::graphics::vertice_vector{
		public:

			circle(fan::camera* camera);

			void push_back(const fan::vec2& position, f32_t radius, const fan::color& color);

			fan::vec2 get_position(uint_t i) const;
			void set_position(uint_t i, const fan::vec2& position);

			f32_t get_radius(uint_t i) const;
			void set_radius(uint_t i, f32_t radius);

			void draw() const;

			bool inside(uint_t i) const;

			fan::color get_color(uint_t i) const;
			void set_color(uint_t i, const fan::color& color);

			uint_t size() const;

			void erase(uint_t i);
			void erase(uint_t begin, uint_t end);

		protected:

			static constexpr int m_segments = 50;

			std::vector<fan::vec2> m_position;
			std::vector<f32_t> m_radius;

		};

		class sprite :
			protected fan_2d::graphics::rectangle,
			protected fan::buffer_object<fan::vec2, 2, true, fan::opengl_buffer_type::shader_storage_buffer_object, false>,
			public fan::texture_handler<1>, // screen texture
			public fan::render_buffer_handler<>,
			public fan::frame_buffer_handler<> {

		public:

			using texture_coordinates_t = fan::buffer_object<fan::vec2, 2, true, fan::opengl_buffer_type::shader_storage_buffer_object, false>;

			sprite(fan::camera* camera);

		protected:

			// requires manual initialization of m_camera
			sprite(const fan::shader& shader);
			sprite(fan::camera* camera, const fan::shader& shader);

		public:

			sprite(const fan_2d::graphics::sprite& sprite);
			sprite(fan_2d::graphics::sprite&& sprite) noexcept;

			fan_2d::graphics::sprite& operator=(const fan_2d::graphics::sprite& sprite);
			fan_2d::graphics::sprite& operator=(fan_2d::graphics::sprite&& sprite);

			~sprite();

			void push_back(std::unique_ptr<fan_2d::graphics::texture_id_handler>& handler, const fan::vec2& position, const fan::vec2& size, const sprite_properties& properties = sprite_properties());

			void insert(uint32_t i, uint32_t texture_coordinates_i, uint32_t texture_id, const fan::vec2& position, const fan::vec2& size, const sprite_properties& properties = sprite_properties());

			void draw(uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized) const;

			void release_queue(uint32_t avoid_flags = 0);

			void erase(uint32_t i);
			void erase(uint32_t begin, uint32_t end);

			// removes everything
			void clear();

			using fan_2d::graphics::rectangle::size;
			using fan_2d::graphics::rectangle::get_size;
			using fan_2d::graphics::rectangle::set_size;
			using fan_2d::graphics::rectangle::get_position;

			using fan_2d::graphics::rectangle::set_position;

			using fan_2d::graphics::rectangle::get_angle;
			using fan_2d::graphics::rectangle::set_angle;
			using fan_2d::graphics::rectangle::inside;

			using fan_2d::graphics::rectangle::m_camera;

		protected:

			void regenerate_texture_switch();

			static constexpr auto location_texture_coordinate = "layout_texture_coordinate";

			std::vector<f32_t> m_transparency;

			std::vector<uint32_t> m_textures;

			std::vector<uint32_t> m_switch_texture;
		};

		class sprite_sheet : protected fan_2d::graphics::sprite {
		public:

			sprite_sheet(fan::camera* camera, uint32_t time);

			using fan_2d::graphics::sprite::push_back;

			void draw();

		private:

			fan::timer<> sheet_timer;

			int32_t current_sheet;

		};

		//class rope : protected fan_2d::graphics::line {

		//public:

		//	rope(fan::camera* camera);

		//	void push_back(const std::vector<std::pair<fan::vec2, fan::vec2>>& joints, const fan::color& color);

		//	using fan_2d::graphics::line::draw;
		//	using fan_2d::graphics::line::size;

		//protected:

		//};

		struct particle {
			f32_t m_angle_velocity;
			fan::vec2 m_velocity;
			fan::timer<> m_timer; // milli
		};

		class particles : protected fan_2d::graphics::rectangle {
		public:

			particles(fan::camera* camera);

			void push_back(
				const fan::vec2& position,
				const fan::vec2& size,
				f32_t angle,
				f32_t angle_velocity,
				const fan::vec2& velocity,
				const fan::color& color,
				uint_t time
			);

			void update();

			using fan_2d::graphics::rectangle::draw;
			using fan_2d::graphics::rectangle::size;

			using fan_2d::graphics::rectangle::release_queue;

		private:

			std::vector<fan_2d::graphics::particle> m_particles;

		};

		class base_lighting :
			protected fan::buffer_object<fan::vec2, 5>,
			protected fan::buffer_object<fan::color, 6>,
			protected fan::buffer_object<f32_t, 7>,
			protected fan::buffer_object<f32_t, 8>
		{
		public:

			// 99 - to avoid collisions

			using light_position_t = fan::buffer_object<fan::vec2, 5>;
			using light_color_t = fan::buffer_object<fan::color, 6>;
			using light_brightness_t = fan::buffer_object<f32_t, 7>;
			using light_angle_t = fan::buffer_object<f32_t, 8>;

			base_lighting(fan::shader* shader, uint32_t* vao);

			void push_back(const fan::vec2& position, f32_t strength, const fan::color& color, f32_t angle);

			fan::vec2 get_position(uint32_t i) const;
			void set_position(uint32_t i, const fan::vec2& position);

			fan::color get_color(uint32_t i) const;
			void set_color(uint32_t i, const fan::color& color);

			f32_t get_brightness(uint32_t i) const;
			void set_brightness(uint32_t i, f32_t brightness);

			f32_t get_angle(uint32_t i) const;
			void set_angle(uint32_t i, f32_t angle);


		protected:

			static constexpr auto location_light_position = "layout_light_position";
			static constexpr auto location_light_color = "layout_light_color";
			static constexpr auto location_light_brightness = "layout_light_brightness";
			static constexpr auto location_light_angle = "layout_light_angle";

			fan::shader* m_shader;

			uint32_t* m_vao;

		};

		struct light : 
			public rectangle,
			public base_lighting
		{

			// gets shader of object that needs lighting
			light(fan::camera* camera, fan::shader* shader, uint32_t* vao);

			void push_back(const fan::vec2& position, const fan::vec2& size, f32_t strength, const fan::color& color, f32_t angle = 0);

			using rectangle::get_position;
			void set_position(uint32_t i, const fan::vec2& position);

			using rectangle::get_color;
			void set_color(uint32_t i, const fan::color& color);

			using rectangle::draw;

		};

	}
}

namespace fan_3d {

	namespace graphics {

		namespace shader_paths {

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

			void push_back(const fan::vec3& src, const fan::vec3& dst, const fan::vec2& texture_id);

			fan::vec3 get_src(uint_t i) const;
			fan::vec3 get_dst(uint_t i) const;
			fan::vec3 get_size(uint_t i) const;

			void set_position(uint_t i, const fan::vec3& src, const fan::vec3& dst);
			void set_size(uint_t i, const fan::vec3& size);

			void draw();

			void set_texture(uint_t i, const fan::vec2& texture_id);

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

	}

}

#endif