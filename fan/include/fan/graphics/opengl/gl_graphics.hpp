#pragma once

#include <fan/graphics/renderer.hpp>

#if fan_renderer == fan_renderer_opengl

#include <fan/graphics/opengl/gl_core.hpp>

#include <fan/graphics/shared_core.hpp>

#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>

#include <fan/graphics/shared_graphics.hpp>

#include <fan/graphics/webp.h>

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

		inline fan_2d::graphics::face_e polygon_face = face_e::front_and_back;
		inline fan_2d::graphics::fill_mode_e polygon_fill_mode = fill_mode_e::fill;

		static void set_viewport(const fan::vec2& position, const fan::vec2& size) {
			glViewport(position.x, position.y, size.x, size.y);
		}

		static void draw(const std::function<void()>& function_) {
			fan::depth_test(false);
			function_();
			fan::depth_test(true);
		}

		fan::mat4 get_projection(const fan::vec2i& window_size);
		fan::mat4 get_view_translation(const fan::vec2i& window_size, const fan::mat4& view);

		// returns how much object moved
		fan::vec2 move_object(fan::window* window, fan::vec2& position, fan::vec2& velocity, f32_t speed, f32_t gravity, f32_t jump_force = -800, f32_t friction = 10);


		// 0 left right, 1 top right, 2 bottom left, 3 bottom right

		namespace image_load_properties {
			inline uint32_t visual_output = GL_CLAMP_TO_BORDER;
			inline uintptr_t internal_format = GL_RGBA;
			inline uintptr_t format = GL_RGBA;
			inline uintptr_t type = GL_UNSIGNED_BYTE;
			inline uintptr_t filter = GL_LINEAR_MIPMAP_LINEAR;
		}

		// fan::get_device(window)
		image_t load_image(fan::window* window, const std::string& path);
		//image_t load_image(fan::window* window, const pixel_data_t& pixel_data);
		fan_2d::graphics::image_t load_image(fan::window* window, const fan::webp::image_info_t& image_info);

		class lighting_properties {
		public:

			lighting_properties(fan::shader_t* shader);

			bool get_lighting() const;
			void set_lighting(bool value);

			f32_t get_world_light_strength() const;
			void set_world_light_strength(f32_t value);

		protected:

			bool m_lighting_on;

			f32_t m_world_light_strength;

			fan::shader_t* m_lighting_shader;

		};

		class vertice_vector : 
			public fan::basic_vertice_vector<fan::vec2>
		{
		public:

			struct properties_t {
				fan::vec2 position;
				f32_t angle = 0;
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

			void erase(uintptr_t i);
			void erase(uintptr_t begin, uintptr_t end);
			void clear();

			fan::vec2 get_position(uint32_t i) const;
			void set_position(uint32_t i, const fan::vec2& position);

			fan::color get_color(uint32_t i) const;
			void set_color(uint32_t i, const fan::color& color);

			void set_angle(uint32_t i, f32_t angle);

			void enable_draw(fan_2d::graphics::shape shape, uint32_t single_draw_amount);
			void disable_draw();

			fan_2d::graphics::fill_mode_e get_draw_mode() const;
			void set_draw_mode(fan_2d::graphics::fill_mode_e fill_mode);

		protected:

			void write_data();

			void edit_data(uint32_t i);

			void edit_data(uint32_t begin, uint32_t end);

			virtual void draw(fan_2d::graphics::shape shape, uint32_t single_draw_amount, uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized, bool texture = false) const;

			vertice_vector(fan::camera* camera, bool init);

			void initialize_buffers();

			uintptr_t m_offset;

			uint32_t m_draw_index = -1;

			queue_helper_t m_queue_helper;

			fan_2d::graphics::fill_mode_e m_fill_mode;

		};

		struct vertices_sprite : 
			public vertice_vector,
			public fan::buffer_object<fan::vec2, 99, true>
		{

			struct properties_t {

				fan_2d::graphics::image_t image; 
				fan::vec2 position;

				f32_t angle = 0;
				fan::vec2 rotation_point;
				fan::vec3 rotation_vector = fan::vec3(0, 0, 1);

				std::vector<fan::vec2> texture_coordinates;

				fan::color color = fan::color(1, 1, 1, 1);

			};

			vertices_sprite(fan::camera* camera);

			void push_back(const vertices_sprite::properties_t& properties);

			void enable_draw(fan_2d::graphics::shape shape, const std::vector<uint32_t>& single_draw_amount);
			void disable_draw();

		protected:

			void write_data();

			void draw(fan_2d::graphics::shape shape, const std::vector<uint32_t>& single_draw_amount);

			void initialize();

			using texture_coordinates_t = fan::buffer_object<fan::vec2, 99, true>;

			static constexpr auto location_texture_coordinate = "layout_texture_coordinates";

			std::vector<uint32_t> m_textures;

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
			protected fan::buffer_object<fan::color, 0>,
			protected fan::buffer_object<fan::vec2, 1>,
			protected fan::buffer_object<fan::vec2, 2>,
			protected fan::buffer_object<f32_t, 3>,
			protected fan::buffer_object<fan::vec2, 4>,
			protected fan::buffer_object<fan::vec3, 5>,
			public fan::vao_handler<>
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

			rectangle(fan::camera* camera);
			~rectangle();

			void push_back(const rectangle::properties_t& properties);

			void insert(uint32_t i, const rectangle::properties_t& properties);

			// requires manual initialization of rotation point
			void reserve(uint32_t size);
			// requires manual initialization of rotation point
			void resize(uint32_t size, const fan::color& color);

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

			constexpr uint64_t element_size() const;

			bool inside(uintptr_t i, const fan::vec2& position = fan::math::inf) const;

			uint32_t* get_vao();

			fan::camera* m_camera = nullptr;

			fan::shader_t get_shader();

			void enable_draw();
			void disable_draw();

			fan_2d::graphics::fill_mode_e get_draw_mode() const;
			void set_draw_mode(fan_2d::graphics::fill_mode_e fill_mode);

			struct read_t {

				uint8_t stage = 0;

			};

			struct write_t {

				uint8_t stage = 0;

			};

			bool read(read_t* read, void* ptr, uintptr_t* size);
			bool write(write_t* write, void* ptr, uintptr_t* size);

			// sets shape's draw order in window
			//void set_draw_order(uint32_t i);

			uint32_t m_draw_index = -1;

			fan::shader_t m_shader;

			lighting_properties m_lighting_properties;

		protected:

			rectangle(fan::camera* camera, bool init);

			void write_data();

			void edit_data(uint32_t i);

			void edit_data(uint32_t begin, uint32_t end);

			// pushed to window draw queue
			void draw(uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized) const;

			void initialize();

			queue_helper_t m_queue_helper;

			fan_2d::graphics::fill_mode_e m_fill_mode = (fan_2d::graphics::fill_mode_e)fan::uninitialized;

			static constexpr auto location_color = "layout_color";
			static constexpr auto location_position = "layout_position";
			static constexpr auto location_size = "layout_size";
			static constexpr auto location_angle = "layout_angle";
			static constexpr auto location_rotation_point = "layout_rotation_point";
			static constexpr auto location_rotation_vector = "layout_rotation_vector";

		};

		struct rectangle0 : public rectangle {

			rectangle0(fan::camera* camera, void* user_ptr, std::function<void(void*, uint64_t, uint32_t)> erase_cb) 
				: rectangle(camera),
					m_erase_cb(erase_cb),
					m_user_ptr(user_ptr)
			{

			}

			struct properties_t : public rectangle::properties_t {
				uint64_t id = -1;
			};
			
			void push_back(const properties_t& properties) {
				m_push_back_ids.emplace_back(properties.id);
				fan_2d::graphics::rectangle::push_back(properties);
			}

			void erase(uint32_t i) {

				if (i != this->size() - 1) {

					std::memcpy(color_t::m_buffer_object.data() + i * 6, color_t::m_buffer_object.data() + color_t::m_buffer_object.size() - 6, sizeof(fan::color) * 6);

					std::memcpy(position_t::m_buffer_object.data() + i * 6, position_t::m_buffer_object.data() + position_t::m_buffer_object.size() - 6, sizeof(fan::vec2) * 6);

					std::memcpy(size_t::m_buffer_object.data() + i * 6, size_t::m_buffer_object.data() + size_t::m_buffer_object.size() - 6, sizeof(fan::vec2) * 6);

					std::memcpy(angle_t::m_buffer_object.data() + i * 6, angle_t::m_buffer_object.data() + angle_t::m_buffer_object.size() - 6, sizeof(f32_t) * 6);

					std::memcpy(rotation_point_t::m_buffer_object.data() + i * 6, rotation_point_t::m_buffer_object.data() + rotation_point_t::m_buffer_object.size() - 6, sizeof(fan::vec2) * 6);

					std::memcpy(rotation_vector_t::m_buffer_object.data() + i * 6, rotation_vector_t::m_buffer_object.data() + rotation_vector_t::m_buffer_object.size() - 6, sizeof(fan::vec3) * 6);

					uint32_t begin = (this->size() - 1) * 6;

					color_t::erase(begin, begin + 6);
					position_t::erase(begin, begin + 6);
					size_t::erase(begin, begin + 6);
					angle_t::erase(begin, begin + 6);
					rotation_point_t::erase(begin, begin + 6);
					rotation_vector_t::erase(begin, begin + 6);

					m_erase_cb(m_user_ptr, *(m_push_back_ids.end() - 1), i);

					m_push_back_ids[i] = *(m_push_back_ids.end() - 1);
					m_push_back_ids.pop_back();

					m_queue_helper.write([&] {
						this->write_data();
					});
				}
				else {
					rectangle::erase(i);
					m_push_back_ids.pop_back();
				}

			}

			void erase(uint32_t, uint32_t) = delete;

		protected:

			void* m_user_ptr = nullptr;

			std::vector<uint64_t> m_push_back_ids;

			std::function<void(void*, uint64_t, uint32_t)> m_erase_cb;

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

		// makes line from src (line start) to dst (line end)
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
					(src + ((dst - src) / 2)),
					dst,
					thickness
				});

				// - fan::vec2(0, 0.5 * thickness)

				rectangle::properties_t property;
				property.position = src + ((dst - src) / 2);
				property.size = fan::vec2((dst - src).length(), thickness) / 2;
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
				position_t::set_value(i, src + ((dst - src) / 2));
				size_t::set_value(i, fan::vec2((dst - src).length(), thickness) / 2);
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

				rectangle::set_position(i, new_src + ((new_dst - new_src) / 2));
				rectangle::set_size(i, new_dst / 2);
			}

			using fan_2d::graphics::rectangle::draw;
			using fan_2d::graphics::rectangle::get_color;
			using fan_2d::graphics::rectangle::set_color;
			using fan_2d::graphics::rectangle::get_rotation_point;
			using fan_2d::graphics::rectangle::set_rotation_point;
			using fan_2d::graphics::rectangle::size;
			using fan_2d::graphics::rectangle::enable_draw;
			using fan_2d::graphics::rectangle::disable_draw;

		protected:

			struct line_instance_t {
				fan::vec2 src;
				fan::vec2 dst;
				f32_t thickness;
			};

			std::vector<line_instance_t> line_instance;

		};

		class sprite :
			protected fan_2d::graphics::rectangle,
			public fan::buffer_object<fan::vec2, 99, true>,
			public fan::texture_handler<1>, // screen texture
			public fan::render_buffer_handler<>,
			public fan::frame_buffer_handler<>,
			public fan::buffer_object<uint32_t, 30213>
		{

		public:

			using texture_coordinates_t = fan::buffer_object<fan::vec2, 99, true>;
			using RenderOPCode_t = fan::buffer_object<uint32_t, 30213>;

			sprite(fan::camera* camera);

		protected:

			// requires manual initialization of m_camera
			sprite(fan::camera* camera, bool init);

		public:

			struct properties_t {

				fan_2d::graphics::image_t image; 
				fan::vec2 position;
				fan::vec2 size;

				f32_t angle = 0;
				fan::vec2 rotation_point;
				fan::vec3 rotation_vector = fan::vec3(0, 0, 1);

				std::array<fan::vec2, 4> texture_coordinates = {
					fan::vec2(0, 1),
					fan::vec2(1, 1),
					fan::vec2(1, 0),
					fan::vec2(0, 0)
				};

				fan::color color = fan::color(1, 1, 1, 1);

				uint32_t RenderOPCode = 0;

			};

			sprite(const fan_2d::graphics::sprite& sprite);
			sprite(fan_2d::graphics::sprite&& sprite) noexcept;

			fan_2d::graphics::sprite& operator=(const fan_2d::graphics::sprite& sprite);
			fan_2d::graphics::sprite& operator=(fan_2d::graphics::sprite&& sprite);

			// fan_2d::graphics::load_image::texture
			void push_back(const sprite::properties_t& properties);

			void insert(uint32_t i, uint32_t texture_coordinates_i, const sprite::properties_t& properties);

			void reload_sprite(uint32_t i, fan_2d::graphics::image_t image);

			f32_t get_transparency(uint32_t i) const;
			void set_transparency(uint32_t i, f32_t transparency);

			uint32_t get_RenderOPCode(uint32_t i) const;
			void set_RenderOPCode(uint32_t i, uint32_t OPCode);

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
			using fan_2d::graphics::rectangle::get_color;
			using fan_2d::graphics::rectangle::set_color;
			using fan_2d::graphics::rectangle::inside;

			using fan_2d::graphics::rectangle::m_camera;

			void enable_draw();
			void disable_draw();

		protected:

			void initialize();

			void draw(uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized) const;

			void write_data();

			void edit_data(uint32_t i);

			void edit_data(uint32_t begin, uint32_t end);

			void regenerate_texture_switch();

			static constexpr auto location_texture_coordinate = "layout_texture_coordinates";
			static constexpr auto location_RenderOPCode = "layout_RenderOPCode";

			std::vector<uint32_t> m_textures;

			std::vector<uint32_t> m_switch_texture;
			//std::vector<fan_2d::graphics::image_t> m_images;

		};

		// moves last to erased spot
		struct sprite0 : public sprite {

			sprite0(fan::camera* camera, void* user_ptr, std::function<void(void*, uint64_t, uint32_t)> erase_cb) 
				: sprite(camera),
					m_erase_cb(erase_cb),
					m_user_ptr(user_ptr)
			{

			}

		protected:

			sprite0(fan::camera* camera, bool init, void* user_ptr, std::function<void(void*, uint64_t, uint32_t)> erase_cb) 
				: sprite(camera, init),
					m_erase_cb(erase_cb),
					m_user_ptr(user_ptr)
			{

			}

		public:

			struct properties_t : public sprite::properties_t {
				uint64_t id = -1;
			};
			
			void push_back(const properties_t& properties) {
				m_push_back_ids.emplace_back(properties.id);
				fan_2d::graphics::sprite::push_back(properties);
			}

			void erase(uint32_t i) {

				if (i != this->size() - 1) {

					std::memcpy(color_t::m_buffer_object.data() + i * 6, color_t::m_buffer_object.data() + color_t::m_buffer_object.size() - 6, sizeof(fan::color) * 6);

					std::memcpy(position_t::m_buffer_object.data() + i * 6, position_t::m_buffer_object.data() + position_t::m_buffer_object.size() - 6, sizeof(fan::vec2) * 6);

					std::memcpy(size_t::m_buffer_object.data() + i * 6, size_t::m_buffer_object.data() + size_t::m_buffer_object.size() - 6, sizeof(fan::vec2) * 6);

					std::memcpy(angle_t::m_buffer_object.data() + i * 6, angle_t::m_buffer_object.data() + angle_t::m_buffer_object.size() - 6, sizeof(f32_t) * 6);

					std::memcpy(rotation_point_t::m_buffer_object.data() + i * 6, rotation_point_t::m_buffer_object.data() + rotation_point_t::m_buffer_object.size() - 6, sizeof(fan::vec2) * 6);

					std::memcpy(rotation_vector_t::m_buffer_object.data() + i * 6, rotation_vector_t::m_buffer_object.data() + rotation_vector_t::m_buffer_object.size() - 6, sizeof(fan::vec3) * 6);

					std::memcpy(texture_coordinates_t::m_buffer_object.data() + i * 6, texture_coordinates_t::m_buffer_object.data() + texture_coordinates_t::m_buffer_object.size() - 6, sizeof(fan::vec2) * 6);

					std::memcpy(RenderOPCode_t::m_buffer_object.data() + i * 6, RenderOPCode_t::m_buffer_object.data() + RenderOPCode_t::m_buffer_object.size() - 6, sizeof(uint32_t) * 6);

					uint32_t begin = (this->size() - 1) * 6;

					color_t::erase(begin, begin + 6);
					position_t::erase(begin, begin + 6);
					size_t::erase(begin, begin + 6);
					angle_t::erase(begin, begin + 6);
					rotation_point_t::erase(begin, begin + 6);
					rotation_vector_t::erase(begin, begin + 6);
					texture_coordinates_t::erase(begin, begin + 6);
					RenderOPCode_t::erase(begin, begin + 6);

					m_erase_cb(m_user_ptr, *(m_push_back_ids.end() - 1), i);

					m_push_back_ids[i] = *(m_push_back_ids.end() - 1);
					m_push_back_ids.pop_back();

					m_textures[i] = *(m_textures.end() - 1);

					m_textures.pop_back();
	
					regenerate_texture_switch();

					m_queue_helper.write([&] {
						this->write_data();
					});
				}
				else {
					sprite::erase(i);
					m_push_back_ids.pop_back();
				}

			}

			void erase(uint32_t, uint32_t) = delete;

		protected:

			void* m_user_ptr = nullptr;

			std::vector<uint64_t> m_push_back_ids;

			std::function<void(void*, uint64_t, uint32_t)> m_erase_cb;

		private:

			using sprite::sprite;

		};

		/*
			in vec2 texture_coordinate;
			in float transparency;
			out vec4 color;
			uniform sampler2D texture_sampler;
		*/

		struct shader_sprite : public sprite {

			shader_sprite(fan::camera* camera, const std::string& shader_main) 
				: sprite(camera, true)
			{
				m_shader->set_vertex(
					#include <fan/graphics/glsl/opengl/2D/sprite.vs>
				);

				std::string fs = 
					#include <fan/graphics/glsl/opengl/2D/sprite.fs>
				;

				auto found = fs.find("void main(");

				fs = fs.substr(0, found);
				fs.append(shader_main);

				m_shader->set_fragment(
					fs
				);

				m_shader->compile();

				sprite::initialize();

			}

		};

		/*
			in vec2 texture_coordinate;
			in float transparency;
			out vec4 color;
			uniform sampler2D texture_sampler;
		*/
		struct shader_sprite0 : public sprite0 {

			shader_sprite0(fan::camera* camera, const std::string& shader_main, void* user_ptr, std::function<void(void*, uint64_t, uint32_t)> erase_cb) 
				: sprite0(camera, true, user_ptr, erase_cb)
			{

				m_shader->set_vertex(
					#include <fan/graphics/glsl/opengl/2D/rectangle.vs>
				);

				std::string fs = 
					#include <fan/graphics/glsl/opengl/2D/rectangle.fs>
				;

				auto found = fs.find("void main(");

				fs = fs.substr(found);
				fs.append(shader_main);

				m_shader->set_fragment(
					fs
				);

				m_shader->compile();

				sprite0::initialize();

			}

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

			fan::vec2ui get_image_size(uint32_t i) const;

			void enable_draw();
			void disable_draw();
			
		protected:

			void draw();

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

			base_lighting(fan::shader_t* shader, uint32_t* vao);

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

			fan::shader_t* m_base_lighting_shader;

			uint32_t* m_vao;

		};

		struct light : 
			public rectangle,
			public base_lighting
		{

			// gets shader of object that needs lighting
			light(fan::camera* camera, fan::shader_t* shader, uint32_t* vao);

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

	}

}

#include <fan/graphics/shared_inline_graphics.hpp>

#endif