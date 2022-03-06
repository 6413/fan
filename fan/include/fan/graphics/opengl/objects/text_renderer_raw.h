#pragma once

#include <fan/graphics/opengl/gl_core.hpp>
#include <fan/graphics/shared_graphics.hpp>
#include <fan/physics/collision/rectangle.hpp>
#include <fan/font.hpp>

namespace fan_2d {
	namespace graphics {
		namespace gui {

			struct text_renderer_raw {

				text_renderer_raw() = default;

				struct properties_t {
					uint16_t font_id;
					f32_t font_size;
					fan::vec2 position = 0;
					fan::color text_color;
					fan::color outline_color = fan::color(0, 0, 0, 0);
					f32_t outline_size = 0.3;
					fan::vec2 rotation_point = 0;
					f32_t angle = 0;
				};

			private:

				struct instance_t {

					fan::color color;
					fan::vec2 position;
					fan::vec2 size;
					f32_t angle;
					fan::vec2 rotation_point;
					fan::vec3 rotation_vector;
					fan::vec2 texture_coordinates;

					f32_t font_size;
					fan::color outline_color;
					f32_t outline_size;

				};

			public:

				static constexpr uint32_t offset_color = offsetof(instance_t, color);
				static constexpr uint32_t offset_position = offsetof(instance_t, position);
				static constexpr uint32_t offset_size = offsetof(instance_t, size);
				static constexpr uint32_t offset_angle = offsetof(instance_t, angle);
				static constexpr uint32_t offset_rotation_point = offsetof(instance_t, rotation_point);
				static constexpr uint32_t offset_rotation_vector = offsetof(instance_t, rotation_vector);
				static constexpr uint32_t offset_texture_coordinates = offsetof(instance_t, texture_coordinates);
				static constexpr uint32_t offset_font_size = offset_texture_coordinates + sizeof(fan::vec2);
				static constexpr uint32_t offset_outline_color = offset_font_size + sizeof(f32_t);
				static constexpr uint32_t offset_outline_size = offset_outline_color + sizeof(fan::color);
				static constexpr uint32_t element_byte_size = offset_outline_size + sizeof(f32_t);

				static constexpr uint32_t vertex_count = 6;

				struct open_t {
					std::string font_name = "bitter";
				};

				void open(fan::opengl::context_t* context, const open_t& open_properties = open_t()) {

					m_shader.open();

					m_shader.set_vertex(
					#include <fan/graphics/glsl/opengl/2D/text.vs>
					);

					m_shader.set_fragment(
					#include <fan/graphics/glsl/opengl/2D/text.fs>
					);

					m_shader.compile();

					m_store.open();
					m_text_renderer_store.open();
					m_glsl_buffer.open();
					m_glsl_buffer.init(m_shader.id, element_byte_size);
					m_queue_helper.open();

					m_draw_node_reference = fan::uninitialized;

					font_image = fan_2d::graphics::load_image(std::string("fonts/") + open_properties.font_name + ".webp");
					
					font = fan::font::parse_font(std::string("fonts/") + open_properties.font_name + "_metrics.txt");
				}
				void close(fan::opengl::context_t* context) {

					m_glsl_buffer.close();
					m_queue_helper.close(context);
					m_shader.close();

					if (m_draw_node_reference == fan::uninitialized) {
						return;
					}

					context->disable_draw(m_draw_node_reference);
					m_draw_node_reference = fan::uninitialized;

					m_store.close();
					m_text_renderer_store.close();
				}


				void push_back(fan::opengl::context_t* context, properties_t properties) {

					auto letter_info = get_letter_info(context, properties.font_id, properties.font_size);

					instance_t letter_properties;
					letter_properties.position = fan::vec2(properties.position.x - letter_info.metrics.offset.x, properties.position.y);
					letter_properties.font_size = properties.font_size;
					letter_properties.color = properties.text_color;
					letter_properties.outline_color = properties.outline_color;
					letter_properties.outline_size = properties.outline_size;
					letter_properties.angle = properties.angle;
					letter_properties.rotation_point = fan::vec3(0, 0, 1);

					push_letter(context, &letter_info, letter_properties);
				}

				fan::vec2 get_position(fan::opengl::context_t* context, uint32_t i) const {
					return *(fan::vec2*)m_glsl_buffer.get_instance(i * vertex_count, element_byte_size, offset_position);
				}
				void set_position(fan::opengl::context_t* context, uint32_t i, const fan::vec2& position) {
					for (uint32_t j = 0; j < vertex_count; j++) {
						m_glsl_buffer.edit_ram_instance(
							i * vertex_count + j,
							&position,
							element_byte_size,
							offset_rotation_point,
							sizeof(fan::vec2)
						);
					}
					m_queue_helper.edit(
						context,
						i * vertex_count * element_byte_size + offset_position,
						(i + 1) * (vertex_count)*element_byte_size - offset_position,
						&m_glsl_buffer
					);
				}

				uint32_t size(fan::opengl::context_t* context) const {
					return m_text.size();
				}

				fan::font::single_info_t get_letter_info(fan::opengl::context_t* context, uint16_t font_index, f32_t font_size) const {
					return font.get_letter_info(font_index, font_size);
				}

				fan::font::single_info_t get_letter_info(fan::opengl::context_t* context, wchar_t c, f32_t font_size) const {
					return font.get_letter_info(c, font_size);
				}

				fan::font::single_info_t get_letter_info(fan::opengl::context_t* context, uint8_t* c, f32_t font_size) const {
					return font.get_letter_info(fan::utf8_to_utf16(c)[0], font_size);
				}

				/*fan::vec2 get_character_position(fan::opengl::context_t* context, uint32_t i, uint32_t j, f32_t font_size) const {
					fan::vec2 position = text_renderer::get_position(context, i);

					auto converted_size = convert_font_size(context, font_size);

					for (int k = 0; k < j; k++) {
						position.x += font.font[m_store[i].m_text[k]].metrics.advance * converted_size;
					}

					position.y = i * (font.line_height * converted_size);

					return position;
				}*/

				f32_t get_font_size(fan::opengl::context_t* context, uintptr_t i) const {
					return *(f32_t*)m_glsl_buffer.get_instance(i * vertex_count, element_byte_size, offset_font_size);
				}
				void set_font_size(fan::opengl::context_t* context, uint32_t i, f32_t font_size) {
					for (uint32_t j = 0; j < vertex_count; j++) {
						m_glsl_buffer.edit_ram_instance(
							i * vertex_count + j,
							&font_size,
							element_byte_size,
							offset_font_size,
							sizeof(f32_t)
						);
					}
					m_queue_helper.edit(
						context,
						i * vertex_count * element_byte_size + offset_font_size,
						(i + 1) * (vertex_count)*element_byte_size - offset_font_size,
						&m_glsl_buffer
					);
				}

				f32_t get_angle(fan::opengl::context_t* context, uint32_t i) const {
					return *(f32_t*)m_glsl_buffer.get_instance(i * vertex_count, element_byte_size, offset_angle);
				}
				void set_angle(fan::opengl::context_t* context, uint32_t i, f32_t angle) {
					for (uint32_t j = 0; j < vertex_count; j++) {
						m_glsl_buffer.edit_ram_instance(
							i * vertex_count + j,
							&angle,
							element_byte_size,
							offset_angle,
							sizeof(f32_t)
						);
					}
					m_queue_helper.edit(
						context,
						i * vertex_count * element_byte_size + offset_angle,
						(i + 1) * (vertex_count)*element_byte_size - offset_angle,
						&m_glsl_buffer
					);
				}

				f32_t get_rotation_point(fan::opengl::context_t* context, uint32_t i) const {
					return *(f32_t*)m_glsl_buffer.get_instance(i * vertex_count, element_byte_size, offset_rotation_point);
				}
				void set_rotation_point(fan::opengl::context_t* context, uint32_t i, const fan::vec2& rotation_point) {
					for (uint32_t j = 0; j < vertex_count; j++) {
						m_glsl_buffer.edit_ram_instance(
							i * vertex_count + j,
							&rotation_point,
							element_byte_size,
							offset_rotation_point,
							sizeof(f32_t)
						);
					}
					m_queue_helper.edit(
						context,
						i * vertex_count * element_byte_size + offset_rotation_point,
						(i + 1) * (vertex_count)*element_byte_size - offset_rotation_point,
						&m_glsl_buffer
					);
				}

				fan::vec3 get_rotation_vector(fan::opengl::context_t* context, uint32_t i) const {
					return *(fan::vec3*)m_glsl_buffer.get_instance(i * vertex_count, element_byte_size, offset_rotation_point);
				}
				void set_rotation_vector(fan::opengl::context_t* context, uint32_t i, const fan::vec2& rotation_vector) {
					for (uint32_t j = 0; j < vertex_count; j++) {
						m_glsl_buffer.edit_ram_instance(
							i * vertex_count + j,
							&rotation_vector,
							element_byte_size,
							offset_rotation_vector,
							sizeof(fan::vec3)
						);
					}
					m_queue_helper.edit(
						context,
						i * vertex_count * element_byte_size + offset_rotation_vector,
						(i + 1) * (vertex_count)*element_byte_size - offset_rotation_vector,
						&m_glsl_buffer
					);
				}

				fan::color get_outline_color(fan::opengl::context_t* context, uint32_t i) const {
					return *(fan::color*)m_glsl_buffer.get_instance(i * vertex_count, element_byte_size, offset_outline_color);
				}
				void set_outline_color(fan::opengl::context_t* context, uint32_t i, const fan::color& outline_color) {
					for (uint32_t j = 0; j < vertex_count; j++) {
						m_glsl_buffer.edit_ram_instance(
							i * vertex_count + j,
							&outline_color,
							element_byte_size,
							offset_outline_color,
							sizeof(fan::color)
						);
					}
					m_queue_helper.edit(
						context,
						i * vertex_count * element_byte_size + offset_outline_color,
						(i + 1) * (vertex_count)*element_byte_size - offset_outline_color,
						&m_glsl_buffer
					);
				}

				f32_t get_outline_size(fan::opengl::context_t* context, uint32_t i) const {
					return *(f32_t*)m_glsl_buffer.get_instance(i * vertex_count, element_byte_size, offset_outline_size);
				}
				void set_outline_size(fan::opengl::context_t* context, uint32_t i, f32_t outline_size) {
					for (uint32_t j = 0; j < vertex_count; j++) {
						m_glsl_buffer.edit_ram_instance(
							i * vertex_count + j,
							&outline_size,
							element_byte_size,
							offset_outline_size,
							sizeof(fan::color)
						);
					}
					m_queue_helper.edit(
						context,
						i * vertex_count * element_byte_size + offset_outline_size,
						(i + 1) * (vertex_count)*element_byte_size - offset_outline_size,
						&m_glsl_buffer
					);
				}

				void erase(fan::opengl::context_t* context, uintptr_t i) {

					uint32_t src = i;
					uint32_t dst = i + 1;

					m_glsl_buffer.erase(src * vertex_count * element_byte_size, dst * vertex_count * element_byte_size);

					m_queue_helper.edit(
						context,
						src * vertex_count * element_byte_size,
						m_glsl_buffer.m_buffer.size(),
						&m_glsl_buffer
					);

					m_store.erase(src, dst);
				}

				void erase(fan::opengl::context_t* context, uintptr_t begin, uintptr_t end) {

					uint32_t src = begin;
					uint32_t dst = end;

					m_glsl_buffer.erase(src * vertex_count * element_byte_size, dst * vertex_count * element_byte_size);

					m_queue_helper.edit(
						context,
						src * vertex_count * element_byte_size,
						m_glsl_buffer.m_buffer.size(),
						&m_glsl_buffer
					);

					m_store.erase(src, dst);
				}

				void clear(fan::opengl::context_t* context) {

					m_glsl_buffer.clear_ram();

					m_queue_helper.edit(
						context,
						0,
						m_glsl_buffer.m_buffer.size(),
						&m_glsl_buffer
					);

					m_store.clear();
				}

				wchar_t get_character(fan::opengl::context_t* context, uint32_t i) const {
					return m_text[i];
				}
				/*void set_character(fan::opengl::context_t* context, fan::character_t character) {

					auto font_size = this->get_font_size(context, i);
					auto position = this->get_position(context, i);
					auto color = this->get_text_color(context, i);

					const auto outline_color = get_outline_color(context, i);
					const auto outline_size = get_outline_size(context, i);

					this->clear(context, i);

					text_renderer::properties_t properties;

					properties.text = text;
					properties.font_size = font_size;
					properties.position = position;
					properties.text_color = color;
					properties.outline_color = outline_color;
					properties.outline_size = outline_size;

					this->insert(context, i, properties);
				}*/

				fan::color get_text_color(fan::opengl::context_t* context, uint32_t i, uint32_t j = 0) const {
					return *(fan::color*)m_glsl_buffer.get_instance(i * vertex_count + j, element_byte_size, offset_color);
				}
				void set_text_color(fan::opengl::context_t* context, uint32_t i, const fan::color& color) {
					for (int j = i * vertex_count; j < vertex_count; j++) {
						m_glsl_buffer.edit_ram_instance(
							j,
							&color,
							element_byte_size,
							offset_color,
							sizeof(fan::color)
						);
					}
					m_queue_helper.edit(
						context,
						i * vertex_count * element_byte_size + offset_color,
						(i + 1) * (vertex_count)*element_byte_size - (element_byte_size - offset_color),
						&m_glsl_buffer
					);
				}
				void set_text_color(fan::opengl::context_t* context, uint32_t i, uint32_t j, const fan::color& color);

				f32_t get_original_font_size(fan::opengl::context_t* context) {
					return font.size;
				}

				static uint64_t get_new_lines(fan::opengl::context_t* context, const fan::utf16_string& str)
				{
					uint64_t new_lines = 0;
					const wchar_t* p = str.data();
					for (int i = 0; i < str.size(); i++) {
						if (p[i] == '\n') {
							new_lines++;
						}
					}

					return new_lines;
				}

				void enable_draw(fan::opengl::context_t* context) {
					m_draw_node_reference = context->enable_draw(this, [](fan::opengl::context_t* c, void* d) { ((decltype(this))d)->draw(c); });
				}
				void disable_draw(fan::opengl::context_t* context) {
					context->disable_draw(m_draw_node_reference);
				}

				void draw(fan::opengl::context_t* context) {
					m_shader.use();
					m_shader.set_int("texture_sampler", 0);
					glActiveTexture(GL_TEXTURE0);
					glBindTexture(GL_TEXTURE_2D, font_image->texture);
					m_glsl_buffer.draw(
						context,
						m_shader,
						0,
						letter_vertex_size() * vertex_count
					);
				}

			protected:

				struct letter_t {
					fan::vec2 texture_position = 0;
					fan::vec2 texture_size = 0;
					std::array<fan::vec2, 4> texture_coordinates;

					fan::vec2 size = 0;
					fan::vec2 offset = 0;
				};

				letter_t get_letter(fan::opengl::context_t* context, fan::font::single_info_t* letter_info, f32_t font_size) {

					letter_t letter;
					letter.texture_position = letter_info->glyph.position / font_image->size;
					letter.texture_size = letter_info->glyph.size / font_image->size;

					fan::vec2 src = letter.texture_position;
					fan::vec2 dst = src + letter.texture_size;

					letter.texture_coordinates = { {
							fan::vec2(src.x, src.y),
							fan::vec2(dst.x, src.y),
							fan::vec2(dst.x, dst.y),
							fan::vec2(src.x, dst.y)
						} };

					letter.size = letter_info->metrics.size / 2;
					letter.offset = letter_info->metrics.offset;

					return letter;
				}

				void push_letter(fan::opengl::context_t* context, fan::font::single_info_t* letter_info, instance_t instance) {

					letter_t letter = get_letter(context, letter_info, instance.font_size);

					instance.position += fan::vec2(letter.size.x, -letter.size.y) + fan::vec2(letter.offset.x, -letter.offset.y);
					instance.size = letter.size;

					uint32_t from = m_glsl_buffer.m_buffer.size();

					for (int i = 0; i < vertex_count; i++) {
						instance.texture_coordinates = fan_2d::graphics::convert_tc_4_2_6(&letter.texture_coordinates, i);
						m_glsl_buffer.push_ram_instance(&instance, element_byte_size);
					}

					m_queue_helper.edit(
						context,
						from,
						m_glsl_buffer.m_buffer.size(),
						&m_glsl_buffer
					);

					m_store.resize(m_store.size() + 1);

					m_store[m_store.size() - 1].m_texture = font_image->texture;
				}

				uint32_t letter_vertex_size() const {
					return m_glsl_buffer.m_buffer.size() / element_byte_size / vertex_count;
				}

				fan::utf16_string m_text;
				fan::vec2 m_position;
				uint32_t m_new_lines;

				struct text_renderer_store_t {
					uint16_t m_font_index;
				};

				struct store_sprite_t {
					uint32_t m_texture;
				};

				fan::hector_t<store_sprite_t> m_store;
				fan::hector_t<text_renderer_store_t> m_text_renderer_store;

				fan::shader_t m_shader;
				fan::graphics::core::glsl_buffer_t m_glsl_buffer;
				fan::graphics::core::queue_helper_t m_queue_helper;
				uint32_t m_draw_node_reference;

				fan_2d::graphics::image_t font_image;

			public:
				fan::font::font_t font;

			};

		}
	}
}