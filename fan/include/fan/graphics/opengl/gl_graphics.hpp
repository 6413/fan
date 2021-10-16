#pragma once

#include <fan/graphics/renderer.hpp>

#if fan_renderer == fan_renderer_opengl

#include <fan/graphics/opengl/gl_core.hpp>

#include <fan/graphics/shared_core.hpp>

#include <fan/types/da.hpp>

#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>

#include <fan/graphics/shared_graphics.hpp>

#ifdef fan_platform_windows
	#pragma comment(lib, "lib/assimp/assimp.lib")
#endif

namespace fan {

	inline fan::vec2 supported_gl_version;

	void depth_test(bool value);

	void print_opengl_version();

	constexpr auto draw_count_table = [](uintptr_t mode) {
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

		static void set_viewport(const fan::vec2& position, const fan::vec2& size) {
			glViewport(position.x, position.y, size.x, size.y);
		}

		static void draw(const std::function<void()>& function_) {
			fan::depth_test(false);
			function_();
			fan::depth_test(true);
		}

		namespace shader_paths {
			constexpr auto text_renderer_vs("glsl/2D/opengl/text.vs");
			constexpr auto text_renderer_fs("glsl/2D/opengl/text.fs");

			constexpr auto single_shapes_bloom_vs("glsl/2D/opengl/bloom.vs");
			constexpr auto single_shapes_bloom_fs("glsl/2D/opengl/bloom.fs");
			constexpr auto single_shapes_blur_vs("glsl/2D/opengl/blur.vs");
			constexpr auto single_shapes_blur_fs("glsl/2D/opengl/blur.fs");

			constexpr auto single_shapes_bloom_final_vs("glsl/2D/opengl/bloom_final.vs");
			constexpr auto single_shapes_bloom_final_fs("glsl/2D/opengl/bloom_final.fs");

			constexpr auto post_processing_vs("glsl/2D/opengl/post_processing.vs");
			constexpr auto post_processing_fs("glsl/2D/opengl/post_processing.fs");

			constexpr auto vertice_vector_vs("glsl/2D/opengl/vertice_vector.vs");
			constexpr auto vertice_vector_fs("glsl/2D/opengl/vertice_vector.fs");

			constexpr auto convex_vs("glsl/2D/opengl/convex.vs");
			constexpr auto convex_fs("glsl/2D/opengl/vertice_vector.fs");

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
			inline uint32_t visual_output = GL_CLAMP_TO_EDGE;
			inline uintptr_t internal_format = GL_RGBA;
			inline uintptr_t format = GL_RGBA;
			inline uintptr_t type = GL_UNSIGNED_BYTE;
			inline uintptr_t filter = GL_LINEAR;
		}

		// fan::get_device(window)
		image_t load_image(fan::window* window, const std::string& path);
		image_t load_image(fan::window* window, const pixel_data_t& pixel_data);

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

			struct properties_t {
				fan::vec2 position;
				f32_t angle;
				fan::vec2 rotation_point;
				fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
				fan::color color;
			};

			static constexpr auto position_location_name = "layout_position";
			static constexpr auto color_location_name = "layout_color";
			static constexpr auto angle_location_name = "layout_angle";
			static constexpr auto rotation_point_location_name = "layout_rotation_point";
			static constexpr auto rotation_vector_location_name = "layout_rotation_vector";

			vertice_vector(fan::camera* camera);
			
			vertice_vector(const vertice_vector& vector);
			vertice_vector(vertice_vector&& vector) noexcept;

			vertice_vector& operator=(const vertice_vector& vector);
			vertice_vector& operator=(vertice_vector&& vector) noexcept;

			virtual void push_back(const properties_t& properties);

			void reserve(uintptr_t size);
			void resize(uintptr_t size, const fan::color& color);

			virtual void draw(fan_2d::graphics::shape shape, uint32_t single_draw_amount, uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized, bool texture = false) const;

			void erase(uintptr_t i);
			void erase(uintptr_t begin, uintptr_t end);

			fan::vec2 get_position(uint32_t i) const;
			void set_position(uint32_t i, const fan::vec2& position);

			fan::color get_color(uint32_t i) const;
			void set_color(uint32_t i, const fan::color& color);

			void set_angle(uint32_t i, f32_t angle);

			void write_data();

			void edit_data(uint32_t i);

			void edit_data(uint32_t begin, uint32_t end);

		protected:

			vertice_vector(fan::camera* camera, const fan::shader& shader);

			void initialize_buffers();

			std::vector<uint32_t> m_indices;

			uintptr_t m_offset;

		};


		struct convex : private vertice_vector {

			convex(fan::camera* camera);

			struct properties_t : public vertice_vector::properties_t {
				std::vector<fan::vec2> points;
			};

			std::size_t size() const;

			void set_angle(uint32_t i, f32_t angle);

			void set_position(uint32_t i, const fan::vec2& position);

			void push_back(convex::properties_t property);

			void draw(fan_2d::graphics::shape shape, uint32_t single_draw_amount, uint32_t begin = -1, uint32_t end = -1);

			using vertice_vector::write_data;

			std::vector<uint8_t> convex_amount;
		};

		/*fan_2d::graphics::line create_grid(fan::camera* camera, const fan::vec2i& block_size, const fan::vec2i& grid_size, const fan::color& color);*/

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
			protected fan::buffer_object<fan::vec2, 4>,
			protected fan::buffer_object<fan::vec3, 5>,
			public fan::buffer_object<uint32_t, 0, true, fan::opengl_buffer_type::buffer_object, false, GL_ELEMENT_ARRAY_BUFFER>,
			public fan::vao_handler<>,
			public lighting_properties
			//protected fan::buffer_object<uint32_t, 0, true, fan::opengl_buffer_type::buffer_object, false, GL_ELEMENT_ARRAY_BUFFER> 
		{
			struct properties_t {
				fan::vec2 position;
				fan::vec2 size;
				f32_t angle = 0;
				// offset from top left
				fan::vec2 rotation_point;
				fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
				fan::color color; 
			};

			using color_t = fan::buffer_object<fan::color, 0>;
			using position_t = fan::buffer_object<fan::vec2, 1>;
			using size_t = fan::buffer_object<fan::vec2, 2>;
			using angle_t = fan::buffer_object<f32_t, 3>;
			using rotation_point_t = fan::buffer_object<fan::vec2, 4>;
			using rotation_vector_t = fan::buffer_object<fan::vec3, 5>;

			using ebo_t = fan::buffer_object<uint32_t, 0, true, fan::opengl_buffer_type::buffer_object, false, GL_ELEMENT_ARRAY_BUFFER>;

			rectangle();

			rectangle(fan::camera* camera);

			void push_back(const rectangle::properties_t& properties);

			void insert(uint32_t i, const rectangle::properties_t& properties);

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

			fan::vec2 get_rotation_point(uint32_t i = 0) const;
			void set_rotation_point(uint32_t i, const fan::vec2& rotation_point);

			fan::vec2 get_rotation_vector(uint32_t i = 0) const;
			void set_rotation_vector(uint32_t i, const fan::vec2& rotation_vector);

			uintptr_t size() const;

			bool inside(uintptr_t i, const fan::vec2& position = fan::math::inf) const;

			uint32_t* get_vao();

			fan::camera* m_camera = nullptr;

			fan::shader* get_shader();

			void write_data();

			void edit_data(uint32_t i);

			void edit_data(uint32_t begin, uint32_t end);

		protected:

			rectangle(const fan::shader& shader);
			rectangle(fan::camera* camera, const fan::shader& shader);

			static constexpr auto location_color = "layout_color";
			static constexpr auto location_position = "layout_position";
			static constexpr auto location_size = "layout_size";
			static constexpr auto location_angle = "layout_angle";
			static constexpr auto location_rotation_point = "layout_rotation_point";
			static constexpr auto location_rotation_vector = "layout_rotation_vector";

		};

		struct rectangle_dynamic : public fan_2d::graphics::rectangle {


			using fan_2d::graphics::rectangle::rectangle;

			// if value is inserted without allocation, returns index of insertion, otherwise fan::uninitialized
			uint32_t push_back(const rectangle::properties_t& properties);

			void erase(uint32_t i);
			void erase(uint32_t begin, uint32_t end);

			// free slot after erase to save allocation time
			std::vector<uint32_t> m_free_slots;

		};

		// makes line from src (line start top left) to dst (line end top left)
		struct line : protected fan_2d::graphics::rectangle {

		public:

			using fan_2d::graphics::rectangle::rectangle;

			struct src_dst_t {
				fan::vec2 src;
				fan::vec2 dst;
			};

			static std::array<src_dst_t, 4> create_box(const fan::vec2& position, const fan::vec2& size) {

				std::array<src_dst_t, 4> box;

				box[0].src = position;
				box[0].dst = position + fan::vec2(size.x, 0);

				box[1].src = position + fan::vec2(size.x, 0);
				box[1].dst = position + fan::vec2(size.x, size.y);

				box[2].src = position + fan::vec2(size.x, size.y);
				box[2].dst = position + fan::vec2(0, size.y);

				box[3].src = position + fan::vec2(0, size.y);
				box[3].dst = position;

				return box;
			}

			void push_back(const fan::vec2& src, const fan::vec2& dst, const fan::color& color, f32_t thickness = 1) {

				line_instance.emplace_back(line_instance_t{
					src,
					dst,
					thickness
					});

				// - fan::vec2(0, 0.5 * thickness)

				rectangle::properties_t property;
				property.position = src;
				property.size = fan::vec2((dst - src).length(), thickness);
				property.angle = -fan::math::aim_angle(src, dst);
				property.color = color;

				rectangle::push_back(property);

			}

			fan::vec2 get_src(uint32_t i) const {
				return line_instance[i].src;
			}
			fan::vec2 get_dst(uint32_t i) const {
				return line_instance[i].dst;
			}

			void set_line(uint32_t i, const fan::vec2& src, const fan::vec2& dst) {

				const auto thickness = this->get_thickness(i);

				// - fan::vec2(0, 0.5 * thickness)
				position_t::set_value(i, src);
				size_t::set_value(i, fan::vec2((dst - src).length(), thickness));
				angle_t::set_value(i, -fan::math::aim_angle(src, dst));
			}

			f32_t get_thickness(uint32_t i) const {
				return line_instance[i].thickness;
			}
			void set_thickness(uint32_t i, const f32_t thickness) {

				const auto src = line_instance[i].src;
				const auto dst = line_instance[i].dst;

				const auto new_src = src;
				const auto new_dst = fan::vec2((dst - src).length(), thickness);

				line_instance[i].thickness = thickness;

				rectangle::set_position(i, new_src);
				rectangle::set_size(i, new_dst);
			}

			using fan_2d::graphics::rectangle::draw;
			using fan_2d::graphics::rectangle::edit_data;
			using fan_2d::graphics::rectangle::write_data;
			using fan_2d::graphics::rectangle::get_color;
			using fan_2d::graphics::rectangle::set_color;
			using fan_2d::graphics::rectangle::get_rotation_point;
			using fan_2d::graphics::rectangle::set_rotation_point;
			using fan_2d::graphics::rectangle::size;

		protected:

			struct line_instance_t {
				fan::vec2 src;
				fan::vec2 dst;
				f32_t thickness;
			};

			std::vector<line_instance_t> line_instance;

		};

		#include <fan/graphics/shared_inline_graphics.hpp>

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

			struct properties_t {

				fan_2d::graphics::image_t image; 
				fan::vec2 position;
				fan::vec2 size;

				f32_t angle = 0;
				fan::vec2 rotation_point;
				fan::vec3 rotation_vector = fan::vec3(0, 0, 1);

				std::array<fan::vec2, 6> texture_coordinates = {
					fan::vec2(0, 1),
					fan::vec2(1, 1),
					fan::vec2(1, 0),

					fan::vec2(0, 1),
					fan::vec2(0, 0),
					fan::vec2(1, 0)
				};

				f32_t transparency = 1;

			};

			sprite(const fan_2d::graphics::sprite& sprite);
			sprite(fan_2d::graphics::sprite&& sprite) noexcept;

			fan_2d::graphics::sprite& operator=(const fan_2d::graphics::sprite& sprite);
			fan_2d::graphics::sprite& operator=(fan_2d::graphics::sprite&& sprite);

			~sprite();

			// fan_2d::graphics::load_image::texture
			void push_back(const sprite::properties_t& properties);

			void insert(uint32_t i, uint32_t texture_coordinates_i, const sprite::properties_t& properties);

			void reload_sprite(uint32_t i, fan_2d::graphics::image_t image);

			void draw(uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized) const;

			void erase(uint32_t i);
			void erase(uint32_t begin, uint32_t end);

			// removes everything
			void clear();

			void write_data();

			void edit_data(uint32_t i);

			void edit_data(uint32_t begin, uint32_t end);

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

			std::vector<uint32_t> m_textures;

			std::vector<uint32_t> m_switch_texture;
			//std::vector<fan_2d::graphics::image_t> m_images;

		};

		struct yuv420p_renderer : 
			public fan_2d::graphics::sprite {

			struct properties_t : public fan_2d::graphics::sprite::properties_t {
				fan_2d::graphics::pixel_data_t pixel_data;
			};

			yuv420p_renderer(fan::camera* camera);

			void push_back(const yuv420p_renderer::properties_t& properties);

			void reload_pixels(uint32_t i, const fan_2d::graphics::pixel_data_t& pixel_data);

			void write_data();

			void draw();

			fan::vec2ui get_image_size(uint32_t i) const;
			
		protected:

			std::vector<fan::vec2ui> image_size;

			static constexpr auto layout_y = "layout_y";
			static constexpr auto layout_u = "layout_u";
			static constexpr auto layout_v = "layout_v";

		};

		class sprite_sheet : protected fan_2d::graphics::sprite {
		public:

			sprite_sheet(fan::camera* camera, uint32_t time);

			using fan_2d::graphics::sprite::push_back;

			void draw();

		private:

			fan::time::clock sheet_timer;

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
			fan::time::clock m_timer; // milli
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
				uintptr_t time
			);

			void update();

			using fan_2d::graphics::rectangle::draw;
			using fan_2d::graphics::rectangle::size;

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

			rectangle_vector(fan::camera* camera, const std::string& path, uintptr_t block_size);
			//rectangle_vector(fan::camera* camera, const fan::color& color, uintptr_t block_size);
			~rectangle_vector();

			void push_back(const fan::vec3& src, const fan::vec3& dst, const fan::vec2& texture_id);

			fan::vec3 get_src(uintptr_t i) const;
			fan::vec3 get_dst(uintptr_t i) const;
			fan::vec3 get_size(uintptr_t i) const;

			void set_position(uintptr_t i, const fan::vec3& src, const fan::vec3& dst);
			void set_size(uintptr_t i, const fan::vec3& size);

			void draw();

			void set_texture(uintptr_t i, const fan::vec2& texture_id);

			void generate_textures(const std::string& path, const fan::vec2& block_size);

			void write_textures();

			void release_queue(bool position, bool size, bool textures);

			square_corners get_corners(uintptr_t i) const;

			uintptr_t size() const;

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