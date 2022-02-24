#pragma once

#include <fan/types/types.hpp>

#include <fan/graphics/renderer.hpp>

#if fan_renderer == fan_renderer_opengl

#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>

#include <fan/graphics/opengl/gl_core.hpp>

#include <fan/graphics/shared_core.hpp>

#include <fan/graphics/shared_graphics.hpp>

#include <fan/graphics/webp.h>

#ifdef fan_platform_windows
	#pragma comment(lib, "lib/assimp/assimp.lib")
#endif

namespace fan {

	inline fan::vec2 supported_gl_version;

	void depth_test(bool value);
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

		// 0 left right, 1 top right, 2 bottom left, 3 bottom right

		namespace image_load_properties {
			inline uint32_t visual_output = GL_CLAMP_TO_BORDER;
			inline uintptr_t internal_format = GL_RGBA;
			inline uintptr_t format = GL_RGBA;
			inline uintptr_t type = GL_UNSIGNED_BYTE;
			inline uintptr_t filter = GL_LINEAR;
		}

		// fan::get_device(window)
		image_t load_image(const std::string& path);
		//image_t load_image(fan::window* window, const pixel_data_t& pixel_data);
		fan_2d::graphics::image_t load_image(const fan::webp::image_info_t& image_info);
		fan_2d::graphics::image_t load_image(const fan_2d::graphics::image_info_t& image_info);

		struct rectangle {

			rectangle() = default;

			struct properties_t {
				fan::color color;
				fan::vec2 position;
				fan::vec2 size;
				f32_t angle = 0;
				fan::vec2 rotation_point;
				fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
			};

			static constexpr uint32_t vertex_count = 6;

			static constexpr uint32_t offset_color = offsetof(properties_t, color);
			static constexpr uint32_t offset_position = offsetof(properties_t, position);
			static constexpr uint32_t offset_size = offsetof(properties_t, size);
			static constexpr uint32_t offset_angle = offsetof(properties_t, angle);
			static constexpr uint32_t offset_rotation_point = offsetof(properties_t, rotation_point);
			static constexpr uint32_t offset_rotation_vector = offsetof(properties_t, rotation_vector);

			void open(fan::opengl::context_t* context);
			void close(fan::opengl::context_t* context);

			void push_back(fan::opengl::context_t* context, rectangle::properties_t properties);

			void insert(fan::opengl::context_t* context, uint32_t i, rectangle::properties_t properties);

			void erase(fan::opengl::context_t* context, uint32_t i);
			void erase(fan::opengl::context_t* context, uint32_t begin, uint32_t end);

			// erases everything
			void clear(fan::opengl::context_t* context);

			rectangle_corners_t get_corners(fan::opengl::context_t* context, uint32_t i) const;

			const fan::color get_color(fan::opengl::context_t* context, uint32_t i) const;
			void set_color(fan::opengl::context_t* context, uint32_t i, const fan::color& color);

			fan::vec2 get_position(fan::opengl::context_t* context, uint32_t i) const;
			void set_position(fan::opengl::context_t* context, uint32_t i, const fan::vec2& position);

			fan::vec2 get_size(fan::opengl::context_t* context, uint32_t i) const;
			void set_size(fan::opengl::context_t* context, uint32_t i, const fan::vec2& size);

			f32_t get_angle(fan::opengl::context_t* context, uint32_t i) const;
			void set_angle(fan::opengl::context_t* context, uint32_t i, f32_t angle);

			fan::vec2 get_rotation_point(fan::opengl::context_t* context, uint32_t i) const;
			void set_rotation_point(fan::opengl::context_t* context, uint32_t i, const fan::vec2& rotation_point);

			fan::vec3 get_rotation_vector(fan::opengl::context_t* context, uint32_t i) const;
			void set_rotation_vector(fan::opengl::context_t* context, uint32_t i, const fan::vec3& rotation_vector);

			uint32_t size(fan::opengl::context_t* context) const;

			bool inside(fan::opengl::context_t* context, uint32_t i, const fan::vec2& position) const;

			void enable_draw(fan::opengl::context_t* context);
			void disable_draw(fan::opengl::context_t* context);

		protected:

			// pushed to window draw queue
			void draw(fan::opengl::context_t* context, uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized);

			fan::shader_t m_shader;
			fan::graphics::core::glsl_buffer_t m_glsl_buffer;
			fan::graphics::core::queue_helper_t m_queue_helper;
			uint32_t m_draw_node_reference;
		};
		/*struct rectangle0 : public rectangle {

			void open(fan::opengl::context_t* context, void* user_ptr, std::function<void(void*, uint64_t, uint32_t)> erase_cb);

			struct properties_t : public rectangle::properties_t {
				uint64_t id = fan::uninitialized;
			};
			
			void push_back(fan::opengl::context_t* context, const properties_t& properties);

			void erase(fan::opengl::context_t* context, uint32_t i);

			void erase(fan::opengl::context_t* context, uint32_t, uint32_t) = delete;

		protected:

			void* m_user_ptr = nullptr;

			fan::hector_t<uint64_t> m_push_back_ids;

			std::function<void(void*, uint64_t, uint32_t)> m_erase_cb;

		};*/

		// makes line from src (line start) to dst (line end)
		struct line : protected fan_2d::graphics::rectangle {
		public:

			struct src_dst_t {
				fan::vec2 src;
				fan::vec2 dst;
			};

			static std::array<src_dst_t, 4> create_box(fan::opengl::context_t* context, const fan::vec2& position, const fan::vec2& size);

			void push_back(fan::opengl::context_t* context, const fan::vec2& src, const fan::vec2& dst, const fan::color& color, f32_t thickness = 1);

			fan::vec2 get_src(fan::opengl::context_t* context, uint32_t i) const;
			fan::vec2 get_dst(fan::opengl::context_t* context, uint32_t i) const;

			void set_line(fan::opengl::context_t* context, uint32_t i, const fan::vec2& src, const fan::vec2& dst);

			f32_t get_thickness(fan::opengl::context_t* context, uint32_t i) const;
			void set_thickness(fan::opengl::context_t* context, uint32_t i, const f32_t thickness);

			using fan_2d::graphics::rectangle::open;
			using fan_2d::graphics::rectangle::close;
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

		struct sprite : 
			protected fan_2d::graphics::rectangle {

			sprite() = default;

			void open(fan::opengl::context_t* context);
			void close(fan::opengl::context_t* context);

			struct properties_t : rectangle::properties_t {

				properties_t() : rectangle::properties_t() {
					color = fan::color(1, 1, 1, 1);
				}

				std::array<fan::vec2, 4> texture_coordinates = {
					fan::vec2(0, 1),
					fan::vec2(1, 1),
					fan::vec2(1, 0),
					fan::vec2(0, 0)
				};
				fan_2d::graphics::image_t image;
			};

			static constexpr uint32_t offset_texture_coordinates = offsetof(properties_t, texture_coordinates);

			// fan_2d::graphics::load_image::texture
			void push_back(fan::opengl::context_t* context, const sprite::properties_t& properties);

			void insert(fan::opengl::context_t* context, uint32_t i, const sprite::properties_t& properties);

			void reload_sprite(fan::opengl::context_t* context, uint32_t i, fan_2d::graphics::image_t image);

			f32_t get_transparency(fan::opengl::context_t* context, uint32_t i) const;
			void set_transparency(fan::opengl::context_t* context, uint32_t i, f32_t transparency);

			std::array<fan::vec2, 4> get_texture_coordinates(fan::opengl::context_t* context, uint32_t i);
			// set texture coordinates before position or size
			void set_texture_coordinates(fan::opengl::context_t* context, uint32_t i, const std::array<fan::vec2, 4>& texture_coordinates);

			void erase(fan::opengl::context_t* context, uint32_t i);
			void erase(fan::opengl::context_t* context, uint32_t begin, uint32_t end);

			// removes everything
			void clear(fan::opengl::context_t* context);

			using fan_2d::graphics::rectangle::close;
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

			void enable_draw(fan::opengl::context_t* context);
			void disable_draw(fan::opengl::context_t* context);

		protected:

			void draw(fan::opengl::context_t* context, uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized);

			void regenerate_texture_switch();

			struct store_sprite_t {
				uint32_t m_texture;
				uint32_t m_switch_texture;
			};

			fan::hector_t<store_sprite_t> m_store_sprite;
		};

		// moves last to erased spot
		struct sprite0 : public sprite {

			typedef void(*erase_cb_t)(void*, uint64_t, uint32_t);

			sprite0() = default;

			void open(fan::opengl::context_t* context, void* user_ptr, erase_cb_t erase_cb) {
				sprite::open(context);

				m_erase_cb = erase_cb;
				m_user_ptr = user_ptr;
				m_push_back_ids.open();
			}

			void close(fan::opengl::context_t* context) {
				m_push_back_ids.close();
			}

			struct properties_t : public sprite::properties_t {
				uint64_t id = fan::uninitialized;
			};
			
			void push_back(fan::opengl::context_t* context, const properties_t& properties) {
				m_push_back_ids.push_back(properties.id);
				fan_2d::graphics::sprite::push_back(context, properties);
			}

			void erase(fan::opengl::context_t* context, uint32_t i) {

				if (i != this->size(context) - 1) {

					std::memmove(
						&m_glsl_buffer.m_buffer[i * sprite::vertex_count * m_glsl_buffer.m_element_size],
						&m_glsl_buffer.m_buffer[(this->size(context) - 1) * sprite::vertex_count * m_glsl_buffer.m_element_size],
						m_glsl_buffer.m_element_size * sprite::vertex_count
					);

					m_glsl_buffer.erase_instance((this->size(context) - 1) * sprite::vertex_count, 1, sprite::vertex_count);

					m_erase_cb(m_user_ptr, *(m_push_back_ids.end() - 1), i);

					m_push_back_ids[i] = *(m_push_back_ids.end() - 1);
					m_push_back_ids.pop_back();

					m_store_sprite[i] = *(m_store_sprite.end() - 1);

					m_store_sprite.pop_back();
	
					regenerate_texture_switch();

					m_queue_helper.edit(
						context, 
						i * sprite::vertex_count *  m_glsl_buffer.m_element_size,
						(this->size(context)) * sprite::vertex_count * m_glsl_buffer.m_element_size,
						&m_glsl_buffer
					);
				}
				else {
					sprite::erase(context, i);
					m_push_back_ids.pop_back();
				}
			}

			void erase(fan::opengl::context_t* context, uint32_t, uint32_t) = delete;

		protected:

			void* m_user_ptr;

			fan::hector_t<uint64_t> m_push_back_ids;

			erase_cb_t m_erase_cb;

		};
//
//		struct sprite1 : public sprite,
//			protected fan::buffer_object<fan::color, 30215>
//		{
//
//			using light_data_t = fan::buffer_object<fan::color, 30215>;
//
//			sprite1(fan::camera* camera, void* user_ptr, std::function<void(void*, uint64_t, uint32_t)> erase_cb) 
//				: sprite(camera, true),
//					m_erase_cb(erase_cb),
//					m_user_ptr(user_ptr)
//			{
//				m_shader.set_vertex(
//					#include <fan/graphics/glsl/opengl/2D/sprite_light_frame_map.vs>
//				);
//
//				m_shader.set_fragment(
//					#include <fan/graphics/glsl/opengl/2D/sprite_light_frame_map.fs>
//				);
//
//				m_shader.compile();
//
//				rectangle::initialize();
//
//				fan::bind_vao(*rectangle::get_vao(), [&] {
//					texture_coordinates_t::initialize_buffers(m_shader.id, location_texture_coordinate, false, 2);
//					RenderOPCode0_t::initialize_buffers(m_shader.id, location_RenderOPCode0, false, 1);
//					RenderOPCode1_t::initialize_buffers(m_shader.id, location_RenderOPCode1, false, 1);
//					light_data_t::initialize_buffers(m_shader.id, layout_light_data, false, light_data_t::value_type::size());
//				});
//			}
//
//		protected:
//
//			sprite1(fan::camera* camera, bool init, void* user_ptr, std::function<void(void*, uint64_t, uint32_t)> erase_cb) 
//				: sprite(camera, init),
//					m_erase_cb(erase_cb),
//					m_user_ptr(user_ptr)
//			{
//
//			}
//
//		public:
//
//			struct properties_t : public sprite::properties_t {
//				uint64_t id = -1;
//				f32_t light_ambient;
//				f32_t light_diffuse;
//				f32_t light_decrease;
//			};
//			
//			uint32_t size() const {
//				return sprite::size();
//			}
//
//			void push_back(const properties_t& properties) {
//				m_push_back_ids.emplace_back(properties.id);
//				
//				for (int i = 0; i < 6; i++) {
//					light_data_t::push_back(fan::color(properties.light_ambient, properties.light_diffuse, properties.light_decrease));
//				}
//
//				bool write_ = m_queue_helper.m_write;
//				fan_2d::graphics::sprite::push_back(properties);
//				if (!write_) {
//					m_camera->m_window->edit_write_call(m_queue_helper.m_write_index, this, [&] {
//						this->write_data();
//					});
//				}
//			}
//
//			void erase(uint32_t i) {
//
//				if (i != sprite::size() - 1) {
//
//					std::memcpy(color_t::m_buffer_object.data() + i * 6, color_t::m_buffer_object.data() + color_t::m_buffer_object.size() - 6, sizeof(fan::color) * 6);
//
//					std::memcpy(position_t::m_buffer_object.data() + i * 6, position_t::m_buffer_object.data() + position_t::m_buffer_object.size() - 6, sizeof(fan::vec2) * 6);
//
//					std::memcpy(size_t::m_buffer_object.data() + i * 6, size_t::m_buffer_object.data() + size_t::m_buffer_object.size() - 6, sizeof(fan::vec2) * 6);
//
//					std::memcpy(angle_t::m_buffer_object.data() + i * 6, angle_t::m_buffer_object.data() + angle_t::m_buffer_object.size() - 6, sizeof(f32_t) * 6);
//
//					std::memcpy(rotation_point_t::m_buffer_object.data() + i * 6, rotation_point_t::m_buffer_object.data() + rotation_point_t::m_buffer_object.size() - 6, sizeof(fan::vec2) * 6);
//
//					std::memcpy(rotation_vector_t::m_buffer_object.data() + i * 6, rotation_vector_t::m_buffer_object.data() + rotation_vector_t::m_buffer_object.size() - 6, sizeof(fan::vec3) * 6);
//
//					std::memcpy(texture_coordinates_t::m_buffer_object.data() + i * 6, texture_coordinates_t::m_buffer_object.data() + texture_coordinates_t::m_buffer_object.size() - 6, sizeof(fan::vec2) * 6);
//
//					std::memcpy(RenderOPCode0_t::m_buffer_object.data() + i * 6, RenderOPCode0_t::m_buffer_object.data() + RenderOPCode0_t::m_buffer_object.size() - 6, sizeof(uint32_t) * 6);
//					std::memcpy(RenderOPCode1_t::m_buffer_object.data() + i * 6, RenderOPCode1_t::m_buffer_object.data() + RenderOPCode1_t::m_buffer_object.size() - 6, sizeof(uint32_t) * 6);
//					std::memcpy(light_data_t::m_buffer_object.data() + i * 6, light_data_t::m_buffer_object.data() + light_data_t::m_buffer_object.size() - 6, sizeof(fan::color) * 6);
//
//					uint32_t begin = (sprite::size() - 1) * 6;
//
//					color_t::erase(begin, begin + 6);
//					position_t::erase(begin, begin + 6);
//					size_t::erase(begin, begin + 6);
//					angle_t::erase(begin, begin + 6);
//					rotation_point_t::erase(begin, begin + 6);
//					rotation_vector_t::erase(begin, begin + 6);
//					texture_coordinates_t::erase(begin, begin + 6);
//					RenderOPCode0_t::erase(begin, begin + 6);
//					RenderOPCode1_t::erase(begin, begin + 6);
//					light_data_t::erase(begin, begin + 6);
//
//					m_erase_cb(m_user_ptr, *(m_push_back_ids.end() - 1), i);
//
//					m_push_back_ids[i] = *(m_push_back_ids.end() - 1);
//					m_push_back_ids.pop_back();
//
//					m_textures[i] = *(m_textures.end() - 1);
//
//					m_textures.pop_back();
//	
//					regenerate_texture_switch();
//
//					m_queue_helper.write([&] {
//						this->write_data();
//					});
//				}
//				else {
//					bool write_ = m_queue_helper.m_write;
//					rectangle::erase(i);
//	
//					texture_coordinates_t::erase(i * 6, i * 6 + 6);
//
//					m_textures.erase(m_textures.begin() + i);
//	
//					regenerate_texture_switch();
//
//					RenderOPCode0_t::erase(i * 6, i * 6 + 6);
//					RenderOPCode1_t::erase(i * 6, i * 6 + 6);
//					light_data_t::erase(i * 6, i * 6 + 6);
//
//					if (!write_) {
//						m_camera->m_window->edit_write_call(m_queue_helper.m_write_index, this, [&] {
//							this->write_data();
//						});
//					}
//					m_push_back_ids.pop_back();
//				}
//
//			}
//
//			void erase(uint32_t, uint32_t) = delete;
//
//		protected:
//
//			void write_data() {
//				sprite::write_data();
//				light_data_t::write_data();
//			}
//
//			static constexpr auto layout_light_data = "layout_light_data";
//
//			void* m_user_ptr = nullptr;
//
//			std::vector<uint64_t> m_push_back_ids;
//
//			std::function<void(void*, uint64_t, uint32_t)> m_erase_cb;
//
//		private:
//
//			using sprite::sprite;
//
//		};
//
//		struct shader_sprite : public sprite {
//
//			shader_sprite(fan::camera* camera, const std::string& custom_fragment_shader) 
//				: sprite(camera, true)
//			{
//				m_shader.set_vertex(
//					#include <fan/graphics/glsl/opengl/2D/sprite.vs>
//				);
//
//				m_shader.set_fragment(
//					custom_fragment_shader
//				);
//
//				m_shader.compile();
//
//				sprite::initialize();
//			}
//
//			void push_back_texture(fan_2d::graphics::image_t image) {
//				m_textures.push_back(image->texture);
//				if (m_textures.size() && m_textures[m_textures.size() - 1] != image->texture) {
//					m_switch_texture.emplace_back(this->size() - 1);
//				}
//			}
//
//		};
//
//		struct shader_sprite0 : public sprite0 {
//
//			std::vector<std::vector<image_t>> m_multi_textures;
//
//			struct properties_t : public sprite0::properties_t {
//			  private:
//				using sprite0::properties_t::image;
//			  public:
//				std::vector<fan_2d::graphics::image_t> images;
//			};
//
//			shader_sprite0(fan::camera* camera, void* user_ptr, std::function<void(void*, uint64_t, uint32_t)> erase_cb, const std::string& custom_fragment_shader) 
//				: sprite0(camera, true, user_ptr, erase_cb)
//			{
//
//				m_shader.set_vertex(
//					#include <fan/graphics/glsl/opengl/2D/sprite.vs>
//				);
//
//				m_shader.set_fragment(
//					custom_fragment_shader
//				);
//
//				m_shader.compile();
//
//				sprite0::initialize();
//			}
//			void push_back(const shader_sprite0::properties_t& properties) {
//				sprite::rectangle::properties_t property;
//				property.position = properties.position;
//				property.size = properties.size;
//				property.angle = properties.angle;
//				property.rotation_point = properties.rotation_point;
//				property.rotation_vector = properties.rotation_vector;
//				property.color = properties.color;
//
//				bool write_ = m_queue_helper.m_write;
//
//				rectangle::push_back(property);
//
//				std::array<fan::vec2, 6> texture_coordinates = {
//					properties.texture_coordinates[0],
//					properties.texture_coordinates[1],
//					properties.texture_coordinates[2],
//
//					properties.texture_coordinates[2],
//					properties.texture_coordinates[3],
//					properties.texture_coordinates[0]
//				};
//
//				texture_coordinates_t::insert(texture_coordinates_t::m_buffer_object.end(), texture_coordinates.begin(), texture_coordinates.end());
//
//				RenderOPCode0_t::m_buffer_object.insert(RenderOPCode0_t::m_buffer_object.end(), 6, properties.RenderOPCode0);
//				RenderOPCode1_t::m_buffer_object.insert(RenderOPCode1_t::m_buffer_object.end(), 6, properties.RenderOPCode1);
//
//				m_multi_textures.emplace_back(properties.images);
//
//				m_push_back_ids.emplace_back(properties.id);
//
//				if (!write_) {
//					m_camera->m_window->edit_write_call(m_queue_helper.m_write_index, this, [&] {
//						this->write_data();
//					});
//				}
//			}
//
//		protected:
//
//			void draw() {
//				m_shader.use();
//
//					for (int i = 0; i < m_multi_textures.size(); i++) {
//						
//						for (int j = 0; j < m_multi_textures[i].size(); j++) {
//							m_shader.set_int((std::string("texture_sampler") + std::to_string(j)).c_str(), j);
//							glActiveTexture(GL_TEXTURE0 + j);
//							glBindTexture(GL_TEXTURE_2D, m_multi_textures[i][j]->texture);
//						}
//
//						fan_2d::graphics::rectangle::draw(i, i + 1);
//					}
//			}
//
//			void temp_erase(uint32_t i)
//			{
//				bool write_ = m_queue_helper.m_write;
//				rectangle::erase(i);
//				
//				texture_coordinates_t::erase(i * 6, i * 6 + 6);
//
//				m_multi_textures.erase(m_multi_textures.begin() + i);
//				
//				regenerate_texture_switch();
//
//				RenderOPCode0_t::erase(i * 6, i * 6 + 6);
//				RenderOPCode1_t::erase(i * 6, i * 6 + 6);
//
//				if (!write_) {
//					m_camera->m_window->edit_write_call(m_queue_helper.m_write_index, this, [&] {
//						this->write_data();
//					});
//				}
//			}
//
//		public:
//
//			void enable_draw()
//			{
//				
//				if (m_draw_index == -1 || m_camera->m_window->m_draw_queue[m_draw_index].first != this) {
//					m_draw_index = m_camera->m_window->push_draw_call(this, [&] {
//						draw();
//					});
//				}
//				else {
//					m_camera->m_window->edit_draw_call(m_draw_index, this, [&] {
//						draw();
//					});
//				}
//			}
//			void erase(uint32_t i) {
//
//				if (i != this->size() - 1) {
//
//					std::memcpy(color_t::m_buffer_object.data() + i * 6, color_t::m_buffer_object.data() + color_t::m_buffer_object.size() - 6, sizeof(fan::color) * 6);
//
//					std::memcpy(position_t::m_buffer_object.data() + i * 6, position_t::m_buffer_object.data() + position_t::m_buffer_object.size() - 6, sizeof(fan::vec2) * 6);
//
//					std::memcpy(size_t::m_buffer_object.data() + i * 6, size_t::m_buffer_object.data() + size_t::m_buffer_object.size() - 6, sizeof(fan::vec2) * 6);
//
//					std::memcpy(angle_t::m_buffer_object.data() + i * 6, angle_t::m_buffer_object.data() + angle_t::m_buffer_object.size() - 6, sizeof(f32_t) * 6);
//
//					std::memcpy(rotation_point_t::m_buffer_object.data() + i * 6, rotation_point_t::m_buffer_object.data() + rotation_point_t::m_buffer_object.size() - 6, sizeof(fan::vec2) * 6);
//
//					std::memcpy(rotation_vector_t::m_buffer_object.data() + i * 6, rotation_vector_t::m_buffer_object.data() + rotation_vector_t::m_buffer_object.size() - 6, sizeof(fan::vec3) * 6);
//
//					std::memcpy(texture_coordinates_t::m_buffer_object.data() + i * 6, texture_coordinates_t::m_buffer_object.data() + texture_coordinates_t::m_buffer_object.size() - 6, sizeof(fan::vec2) * 6);
//
//					std::memcpy(RenderOPCode0_t::m_buffer_object.data() + i * 6, RenderOPCode0_t::m_buffer_object.data() + RenderOPCode0_t::m_buffer_object.size() - 6, sizeof(uint32_t) * 6);
//					std::memcpy(RenderOPCode1_t::m_buffer_object.data() + i * 6, RenderOPCode1_t::m_buffer_object.data() + RenderOPCode1_t::m_buffer_object.size() - 6, sizeof(uint32_t) * 6);
//
//					uint32_t begin = (this->size() - 1) * 6;
//
//					color_t::erase(begin, begin + 6);
//					position_t::erase(begin, begin + 6);
//					size_t::erase(begin, begin + 6);
//					angle_t::erase(begin, begin + 6);
//					rotation_point_t::erase(begin, begin + 6);
//					rotation_vector_t::erase(begin, begin + 6);
//					texture_coordinates_t::erase(begin, begin + 6);
//					RenderOPCode0_t::erase(begin, begin + 6);
//					RenderOPCode1_t::erase(begin, begin + 6);
//
//					m_erase_cb(m_user_ptr, *(m_push_back_ids.end() - 1), i);
//
//					m_push_back_ids[i] = *(m_push_back_ids.end() - 1);
//					m_push_back_ids.pop_back();
//
//					m_multi_textures[i] = *(m_multi_textures.end() - 1);
//					m_multi_textures.pop_back();
//	
//					regenerate_texture_switch();
//
//					m_queue_helper.write([&] {
//						this->write_data();
//					});
//				}
//				else {
//					temp_erase(i);
//					m_push_back_ids.pop_back();
//				}
//
//			}
//
//
//		};

		struct yuv420p_renderer : 
			public fan_2d::graphics::sprite {

			struct properties_t : public fan_2d::graphics::sprite::properties_t {
				fan_2d::graphics::pixel_data_t pixel_data;
			};

			void open(fan::opengl::context_t* context);

			void push_back(fan::opengl::context_t* context, const yuv420p_renderer::properties_t& properties);

			void reload_pixels(fan::opengl::context_t* context, uint32_t i, const fan_2d::graphics::pixel_data_t& pixel_data);

			fan::vec2ui get_image_size(fan::opengl::context_t* context, uint32_t i) const;

			void enable_draw(fan::opengl::context_t* context);

		protected:

			void draw(fan::opengl::context_t* context);

			std::vector<fan::vec2ui> image_size;

			static constexpr auto layout_y = "layout_y";
			static constexpr auto layout_u = "layout_u";
			static constexpr auto layout_v = "layout_v";

		};
	}
}
//
//namespace fan_3d {
//
//	namespace graphics {
//
//		void add_camera_rotation_callback(fan::camera* camera);
//
//		/*struct model_t : 
//			fan::vao_handler<1423>,
//			fan::buffer_object<fan::vec3, 124312>,
//			fan::buffer_object<fan::vec3, 124313>
//		{
//
//			using vao_t = fan::vao_handler<1423>;
//			using vertices_t = fan::buffer_object<fan::vec3, 124312>;
//			using normals_t = fan::buffer_object<fan::vec3, 124313>;
//
//			struct properties_t {
//				std::string path;
//				fan::vec3 position;
//				fan::vec3 size = 1;
//				f32_t angle = 0;
//				fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
//			};
//
//			model_t(fan::camera* camera);
//			~model_t();
//
//			void push_back(const properties_t& properties);
//
//			fan::vec3 get_position(uint32_t i) const;
//			void set_position(uint32_t i, const fan::vec3& position);
//
//			fan::vec3 get_size(uint32_t i) const;
//			void set_size(uint32_t i, const fan::vec3& size);
//
//			f32_t get_angle(uint32_t i) const;
//			void set_angle(uint32_t i, f32_t angle);
//
//			fan::vec3 get_rotation_vector(uint32_t i) const;
//			void set_rotation_vector(uint32_t i, const fan::vec3& rotation_vector);
//
//
//			void enable_draw();
//			void disable_draw();
//
//		protected:
//
//			static constexpr auto vertex_layout_location = "layout_vertex";
//			static constexpr auto normal_layout_location = "layout_normal";
//			
//			void parse_model(const properties_t& properties, uint32_t& current_index, uint32_t& max_index, aiNode *node, const aiScene *scene);
//
//			void draw();
//
//			uint32_t m_ebo;
//			std::vector<uint32_t> m_indices;
//			std::vector<fan::mat4> m_model;
//
//			uint32_t m_draw_index = -1;
//			fan::camera* m_camera;
//			fan::shader_t m_shader;
//
//			fan_2d::graphics::queue_helper_t m_queue_helper;
//
//		};*/
//
//		namespace animation {
//
//			struct vertex_t {
//				fan::vec3 position;
//				fan::vec3 normal;
//				fan::vec2 uv;
//				fan::vec4 bone_ids;
//				fan::vec4 bone_weights;
//			};
//
//			// structure to hold bone tree (skeleton)
//			struct joint_t {
//				int id; // position of the bone in final upload array
//				std::string name;
//				fan::mat4 offset;
//				std::vector<joint_t> children;
//			};
//
//			// sturction representing an animation track
//			struct bone_transform_track_t {
//				std::vector<f32_t> position_timestamps;
//				std::vector<f32_t> rotation_timestamps;
//				std::vector<f32_t> scale_timestamps;
//
//				std::vector<fan::vec3> positions;
//				std::vector<fan::quat> rotations;
//				std::vector<fan::vec3> scales;
//			};
//
//			// structure containing animation information
//			struct loaded_animation_t {
//				f_t duration;
//				f_t ticks_per_second;
//				std::unordered_map<std::string, bone_transform_track_t> bone_transforms;
//			};
//
//				// a recursive function to read all bones and form skeleton
//			static bool read_skeleton(
//				animation::joint_t& joint, 
//				aiNode* node,
//				std::unordered_map<std::string, 
//				std::pair<int, fan::mat4>>& boneInfoTable
//			);
//
//			static void load_model(
//				const aiScene* scene, 
//				std::vector<animation::vertex_t>& verticesOutput, 
//				std::vector<uint32_t>& indicesOutput, animation::joint_t& skeletonOutput, 
//				uint32_t &nBoneCount
//			);
//
//			static void load_animation(const aiScene* scene, fan_3d::graphics::animation::loaded_animation_t& animation);
//
//			static std::pair<uint32_t, f32_t> get_time_fraction(std::vector<f32_t>& times, f32_t& dt);
//
//			void get_pose(
//				animation::loaded_animation_t* animation, 
//				animation::joint_t* skeleton, 
//				f32_t dt, 
//				std::vector<fan::mat4>* output, 
//				fan::mat4 parentTransform, 
//				fan::mat4 transform,
//				uint32_t bone_count
//			);
//
//			struct animated_model_t_ {
//
//				animated_model_t_(std::string model_path, fan_2d::graphics::image_t image_diffuse_);
//
//				const aiScene* scene;
//				Assimp::Importer importer;
//				fan_2d::graphics::image_t image_diffuse;
//
//				std::vector<fan_3d::graphics::animation::vertex_t> vertices;
//				std::vector<uint32_t> indices;
//				fan_3d::graphics::animation::loaded_animation_t animation;
//
//				fan_3d::graphics::animation::joint_t skeleton;
//				uint32_t bone_count;
//
//			};
//
//			using animated_model_t = animated_model_t_*;
//
//			struct loaded_model_t {
//				std::vector<animation::vertex_t> vertices;
//				std::vector<uint32_t> indices;
//				animation::joint_t skeleton;
//				uint32_t bone_count;
//			};
//
//			struct simple_animation_t {
//				
//				struct properties_t {
//					animated_model_t model;
//					fan::vec3 position;
//					fan::vec3 size = 1;
//					fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
//					f32_t angle = 0;
//				};
//
//				simple_animation_t() {}
//
//				simple_animation_t(fan::camera* camera, const properties_t& properties);
//
//				fan::vec3 get_position() const;
//				void set_position(const fan::vec3& position);
//				
//				fan::vec3 get_size() const;
//				void set_size(const fan::vec3& size);
//
//				fan::vec3 get_rotation_vector() const;
//				void set_rotation_vector(const fan::vec3& vector);
//
//				f32_t get_angle() const;
//				void set_angle(f32_t angle);
//
//				f32_t get_keyframe() const;
//				void set_keyframe(uint32_t keyframe);
//
//				void enable_draw();
//				void disable_draw();
//
//			protected:
//
//				void draw();
//
//				fan::vec3 m_rotation_vector;
//				f32_t m_angle;
//
//				uint32_t m_keyframe;
//
//				fan::mat4 m_model;
//
//				uint32_t m_vao, m_vbo, m_ebo;
//
//				uint32_t m_draw_index = -1;
//
//				fan_2d::graphics::image_t image_diffuse;
//
//				fan::camera* m_camera;
//
//				std::vector<fan_3d::graphics::animation::vertex_t> m_vertices;
//				std::vector<uint32_t> m_indices;
//
//				fan::shader_t m_shader;
//
//				fan_3d::graphics::animation::loaded_animation_t m_animation;
//
//			};
//
//			class animator_t {
//			public:
//
//				animator_t() {}
//
//				struct properties_t {
//					animated_model_t model;
//					fan::vec3 position;
//					fan::vec3 size = 1;
//					fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
//					f32_t angle = 0;
//				};
//
//				animator_t(fan::camera* camera, const properties_t& properties);
//
//				fan::vec3 get_position() const;
//				void set_position(const fan::vec3& position);
//				
//				fan::vec3 get_size() const;
//				void set_size(const fan::vec3& size);
//
//				fan::vec3 get_rotation_vector() const;
//				void set_rotation_vector(const fan::vec3& vector);
//
//				f32_t get_angle() const;
//				void set_angle(f32_t angle);
//
//				f32_t get_timestamp() const;
//				void set_timestamp(f32_t timestamp);
//
//				void enable_draw();
//				void disable_draw();
//
//			protected:
//
//				void draw();
//
//				fan::vec3 m_rotation_vector;
//				f32_t m_angle;
//
//				f32_t m_timestamp;
//
//				fan::mat4 m_model;
//
//				uint32_t m_vao, m_vbo, m_ebo;
//
//				uint32_t m_draw_index = -1;
//
//				fan_2d::graphics::image_t image_diffuse;
//
//				fan::camera* m_camera;
//
//				std::vector<fan_3d::graphics::animation::vertex_t> m_vertices;
//				std::vector<uint32_t> m_indices;
//
//				std::vector<fan::mat4> m_current_pose;
//
//				fan_3d::graphics::animation::loaded_animation_t m_animation;
//
//				fan_3d::graphics::animation::joint_t m_skeleton;
//
//				fan::mat4 m_identity;
//				fan::mat4 m_transform;
//
//				fan::shader_t m_shader;
//
//				uint32_t m_bone_count;
//
//			};
//		}
//
//		using animation_t = animation::animator_t;
//	}
//}

#include <fan/graphics/shared_inline_graphics.hpp>

#endif