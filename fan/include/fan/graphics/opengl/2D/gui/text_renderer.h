#pragma once

#include <fan/graphics/opengl/gl_core.h>
#include <fan/graphics/opengl/gl_shader.h>
#include <fan/graphics/shared_graphics.h>
#include <fan/physics/collision/rectangle.h>
#include <fan/font.h>

namespace fan_2d {
	namespace opengl {
		namespace gui {

			struct text_renderer_t {

				text_renderer_t() = default;

				struct properties_t {
					fan::utf16_string text;
					f32_t font_size;
					fan::vec2 position = 0;
					fan::color text_color;
					fan::color outline_color = fan::color(0, 0, 0, 0);
					f32_t outline_size = 0.8;
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

				void open(fan::opengl::context_t* context) {

					m_shader.open(context);

					m_shader.set_vertex(
						context, 
						#include <fan/graphics/glsl/opengl/2D/gui/text.vs>
					);

					m_shader.set_fragment(
						context, 
						#include <fan/graphics/glsl/opengl/2D/gui/text.fs>
					);

					m_shader.compile(context);

					m_store_sprite.open();
					m_glsl_buffer.open(context);
					m_glsl_buffer.init(context, m_shader.id, element_byte_size);
					m_queue_helper.open();

					m_draw_node_reference = fan::uninitialized;

					m_store.open();

					static constexpr const char* font_name = "bitter";

					if (!font_image) {
						font_image = fan::opengl::load_image(context, std::string("fonts/") + font_name + ".webp");
					}

					font = fan::font::parse_font(std::string("fonts/") + font_name + "_metrics.txt");
				}
				void close(fan::opengl::context_t* context) {

					m_glsl_buffer.close(context);
					m_queue_helper.close(context);
					m_shader.close(context);

					if (m_draw_node_reference == fan::uninitialized) {
						return;
					}

					context->disable_draw(m_draw_node_reference);
					m_draw_node_reference = fan::uninitialized;

					m_store_sprite.close();

					for (int i = 0; i < m_store.size(); i++) {
						m_store[i].m_text.close();
					}
					m_store.close();
				}


				void push_back(fan::opengl::context_t* context, properties_t properties) {

					if (properties.text.empty()) {
						throw std::runtime_error("text cannot be empty");
					}

					store_t store;
					store.m_text.open();

					store.m_position = properties.position;
					*store.m_text = properties.text;
					store.m_indices = m_store.empty() ? properties.text.size() : m_store[m_store.size() - 1].m_indices + properties.text.size();

					fan::vec2 text_size = get_text_size(context, properties.text, properties.font_size);

					f32_t left = properties.position.x - text_size.x / 2;

					uint64_t new_lines = get_new_lines(context, properties.text);

					properties.position.y += font.size * convert_font_size(context, properties.font_size) / 2;
					if (new_lines) {
						properties.position.y -= (get_line_height(context, properties.font_size) * (new_lines - 1)) / 2;
					}

					f32_t average_height = 0;

					store.m_new_lines = new_lines;

					m_store.push_back(store);

					for (int i = 0; i < properties.text.size(); i++) {

						if (properties.text[i] == '\n') {
							left = properties.position.x - text_size.x / 2;
							properties.position.y += get_line_height(context, properties.font_size);
						}

						auto letter = get_letter_info(context, properties.text[i], properties.font_size);

						average_height += letter.metrics.size.y;

						instance_t letter_properties;
						letter_properties.position = fan::vec2(left - letter.metrics.offset.x, properties.position.y);
						letter_properties.font_size = properties.font_size;
						letter_properties.color = properties.text_color;
						letter_properties.outline_color = properties.outline_color;
						letter_properties.outline_size = properties.outline_size;
						letter_properties.angle = properties.angle;
						letter_properties.rotation_vector = fan::vec3(0, 0, 1);
						letter_properties.rotation_point = (properties.position - fan::vec2(0, text_size.y / 2 - (average_height / properties.text.size()) / 2)) + properties.rotation_point;

						push_letter(context, properties.text[i], letter_properties);
						left += letter.metrics.advance;
					}
				}

				void insert(fan::opengl::context_t* context, uint32_t i, properties_t properties) {
					if (properties.text.empty()) {
						throw std::runtime_error("text cannot be empty");
					}

					store_t store;
					store.m_text.open();

					store.m_position = properties.position;
					*store.m_text = properties.text;

					fan::vec2 text_size = get_text_size(context, properties.text, properties.font_size);

					f32_t left = properties.position.x - text_size.x / 2;

					uint64_t new_lines = get_new_lines(context, properties.text);

					properties.position.y += font.size * convert_font_size(context, properties.font_size) / 2;
					if (new_lines) {
						properties.position.y -= (get_line_height(context, properties.font_size) * (new_lines - 1)) / 2;
					}

					f32_t average_height = 0;

					store.m_new_lines = new_lines;
					m_store.insert(i, store);

					for (int j = 0; j < properties.text.size(); j++) {

						if (properties.text[j] == '\n') {
							left = properties.position.x - text_size.x / 2;
							properties.position.y += font.get_line_height(properties.font_size);
						}

						auto letter = get_letter_info(context, properties.text[j], properties.font_size);

						average_height += letter.metrics.size.y;

						instance_t letter_properties;
						letter_properties.position = fan::vec2(left - letter.metrics.offset.x, properties.position.y);
						letter_properties.font_size = properties.font_size;
						letter_properties.color = properties.text_color;
						letter_properties.outline_color = properties.outline_color;
						letter_properties.outline_size = properties.outline_size;
						letter_properties.angle = properties.angle;
						letter_properties.rotation_vector = fan::vec3(0, 0, 1);
						letter_properties.rotation_point = properties.position - fan::vec2(0, text_size.y / 2 - (average_height / properties.text.size()) / 2) + properties.rotation_point;

						insert_letter(context, get_index(i) + j, properties.text[j], letter_properties);
						left += letter.metrics.advance;

					}
					regenerate_indices();
				}

				fan::vec2 get_position(fan::opengl::context_t* context, uint32_t i) const {
					return m_store[i].m_position;
				}
				void set_position(fan::opengl::context_t* context, uint32_t i, const fan::vec2& position) {
					const uint32_t index = i == 0 ? 0 : m_store[i - 1].m_indices;

					const fan::vec2 offset = position - get_position(context, i);

					m_store[i].m_position = position;

					for (int j = 0; j < m_store[i].m_text->size(); j++) {

						set_letter_position(context, index + j, get_letter_position(context, index + j) + offset);

						for (uint32_t j = 0; j < vertex_count; j++) {
							m_glsl_buffer.edit_ram_instance(
								context, 
								i * vertex_count + j,
								&position,
								element_byte_size,
								offset_rotation_point,
								sizeof(fan::vec2)
							);
						}

					}
				}

				uint32_t size(fan::opengl::context_t* context) const {
					return m_store.size();
				}

				static fan::font::single_info_t get_letter_info(fan::opengl::context_t* context, wchar_t c, f32_t font_size) {
					auto found = font.characters.find(c);

					if (found == font.characters.end()) {
						throw std::runtime_error("failed to find character: " + std::to_string((int)c));
					}

					f32_t converted_size = font.convert_font_size(font_size);

					fan::font::single_info_t font_info;
					font_info.metrics.size = found->second.metrics.size * converted_size;
					font_info.metrics.offset = found->second.metrics.offset * converted_size;
					font_info.metrics.advance = (found->second.metrics.advance * converted_size);

					font_info.glyph = found->second.glyph;
					font_info.mapping = found->second.mapping;

					return font_info;
				}

				static fan::font::single_info_t get_letter_info(fan::opengl::context_t* context, uint8_t* c, f32_t font_size) {

					auto found = font.characters.find(fan::utf16_string(c).data()[0]);

					if (found == font.characters.end()) {
						throw std::runtime_error("failed to find character: " + std::to_string(fan::utf16_string(c).data()[0]));
					}

					f32_t converted_size = font.convert_font_size(font_size);

					fan::font::single_info_t font_info;
					font_info.metrics.size = found->second.metrics.size * converted_size;
					font_info.metrics.offset = found->second.metrics.offset * converted_size;
					font_info.metrics.advance = (found->second.metrics.advance * converted_size);

					font_info.glyph = found->second.glyph;
					font_info.mapping = found->second.mapping;

					return font_info;
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
					return *(f32_t*)m_glsl_buffer.get_instance(context, get_index(i) * vertex_count, element_byte_size, offset_font_size);
				}
				void set_font_size(fan::opengl::context_t* context, uint32_t i, f32_t font_size) {
					const auto text = get_text(context, i);

					const auto position = get_position(context, i);

					const auto color = get_text_color(context, i);

					const auto outline_color = get_outline_color(context, i);
					const auto outline_size = get_outline_size(context, i);

					this->erase(context, i);

					properties_t properties;
					properties.text = text;
					properties.font_size = font_size;
					properties.position = position;
					properties.text_color = color;
					properties.outline_color = outline_color;
					properties.outline_size = outline_size;

					this->insert(context, i, properties);
				}

				f32_t get_angle(fan::opengl::context_t* context, uint32_t i) const {
					return *(f32_t*)m_glsl_buffer.get_instance(context, get_index(i) * vertex_count, element_byte_size, offset_angle);
				}
				void set_angle(fan::opengl::context_t* context, uint32_t i, f32_t angle) {
					for (int j = get_index(i) * vertex_count; j < get_index(i + 1) * vertex_count; j += vertex_count) {
						m_glsl_buffer.edit_ram_instance(
							context, 
							j,
							&angle,
							element_byte_size,
							offset_angle,
							sizeof(properties_t::angle)
						);
					}
					m_queue_helper.edit(
						context,
						get_index(i) * vertex_count * element_byte_size + offset_angle,
						get_index(i + 1) * (vertex_count)*element_byte_size - offset_angle,
						&m_glsl_buffer
					);
				}

				f32_t get_rotation_point(fan::opengl::context_t* context, uint32_t i) const {
					return *(f32_t*)m_glsl_buffer.get_instance(context, get_index(i) * vertex_count, element_byte_size, offset_rotation_point);
				}
				void set_rotation_point(fan::opengl::context_t* context, uint32_t i, const fan::vec2& rotation_point) {
					for (int j = get_index(i) * vertex_count; j < get_index(i + 1) * vertex_count; j += vertex_count) {
						m_glsl_buffer.edit_ram_instance(
							context, 
							j,
							&rotation_point,
							element_byte_size,
							offset_rotation_point,
							sizeof(properties_t::rotation_point)
						);
					}
					m_queue_helper.edit(
						context,
						get_index(i) * vertex_count * element_byte_size + offset_rotation_point,
						get_index(i + 1) * (vertex_count)*element_byte_size - offset_rotation_point,
						&m_glsl_buffer
					);
				}

				fan::color get_outline_color(fan::opengl::context_t* context, uint32_t i) const {
					return *(fan::color*)m_glsl_buffer.get_instance(context, get_index(i) * vertex_count, element_byte_size, offset_outline_color);
				}
				void set_outline_color(fan::opengl::context_t* context, uint32_t i, const fan::color& outline_color) {
					for (int j = get_index(i) * vertex_count; j < get_index(i + 1) * vertex_count; j += vertex_count) {
						m_glsl_buffer.edit_ram_instance(
							context, 
							j,
							&outline_color,
							element_byte_size,
							offset_outline_color,
							sizeof(properties_t::outline_color)
						);
					}
					m_queue_helper.edit(
						context,
						get_index(i) * vertex_count * element_byte_size + offset_outline_color,
						get_index(i + 1) * (vertex_count)*element_byte_size - offset_outline_color,
						&m_glsl_buffer
					);
				}

				f32_t get_outline_size(fan::opengl::context_t* context, uint32_t i) const {
					return *(f32_t*)m_glsl_buffer.get_instance(context, get_index(i) * vertex_count, element_byte_size, offset_outline_size);
				}
				void set_outline_size(fan::opengl::context_t* context, uint32_t i, f32_t outline_size) {
					for (int j = get_index(i) * vertex_count; j < get_index(i + 1) * vertex_count; j += vertex_count) {
						m_glsl_buffer.edit_ram_instance(
							context, 
							j,
							&outline_size,
							element_byte_size,
							offset_outline_size,
							sizeof(properties_t::outline_color)
						);
					}
					m_queue_helper.edit(
						context,
						get_index(i) * vertex_count * element_byte_size + offset_outline_size,
						get_index(i + 1) * (vertex_count)*element_byte_size - offset_outline_size,
						&m_glsl_buffer
					);
				}

				static f32_t convert_font_size(fan::opengl::context_t* context, f32_t font_size) {
					return font_size / font.size;
				}

				void erase(fan::opengl::context_t* context, uintptr_t i) {

					uint32_t src = get_index(i);
					uint32_t dst = get_index(i + 1);

					m_glsl_buffer.erase(context, src * vertex_count * element_byte_size, dst * vertex_count * element_byte_size);

					m_store_sprite.erase(src, dst);

					m_store.erase(i);

					this->regenerate_indices();

					m_queue_helper.edit(
						context,
						src * vertex_count * element_byte_size,
						m_glsl_buffer.m_buffer.size(),
						&m_glsl_buffer
					);
				}

				void erase(fan::opengl::context_t* context, uintptr_t begin, uintptr_t end) {

					uint32_t src = get_index(begin);
					uint32_t dst = get_index(end);

					m_glsl_buffer.erase(context, src * vertex_count * element_byte_size, dst * vertex_count * element_byte_size);

					m_store_sprite.erase(src, dst);

					m_store.erase(begin, end);

					this->regenerate_indices();

					m_queue_helper.edit(
						context,
						src * vertex_count * element_byte_size,
						m_glsl_buffer.m_buffer.size(),
						&m_glsl_buffer
					);
				}

				void clear(fan::opengl::context_t* context) {

					m_glsl_buffer.clear_ram(context);
					m_store.clear();
					m_store_sprite.clear();

					m_queue_helper.edit(
						context,
						0,
						(this->size(context)) * vertex_count * element_byte_size,
						&m_glsl_buffer
					);
				}


				static f32_t get_line_height(fan::opengl::context_t* context, f32_t font_size) {
					return font.line_height * convert_font_size(context, font_size);
				}

				fan::utf16_string get_text(fan::opengl::context_t* context, uint32_t i) const {
					return *m_store[i].m_text;
				}
				void set_text(fan::opengl::context_t* context, uint32_t i, const fan::utf16_string& text) {
					fan::utf16_string str = text;

					if (str.empty()) {
						str.resize(1);
					}

					auto font_size = this->get_font_size(context, i);
					auto position = this->get_position(context, i);
					auto color = this->get_text_color(context, i);

					const auto outline_color = get_outline_color(context, i);
					const auto outline_size = get_outline_size(context, i);

					this->erase(context, i);

					properties_t properties;

					properties.text = text;
					properties.font_size = font_size;
					properties.position = position;
					properties.text_color = color;
					properties.outline_color = outline_color;
					properties.outline_size = outline_size;

					this->insert(context, i, properties);
				}

				fan::color get_text_color(fan::opengl::context_t* context, uint32_t i, uint32_t j = 0) const {
					return *(fan::color*)m_glsl_buffer.get_instance(context, get_index(i) * vertex_count + j, element_byte_size, offset_color);
				}
				void set_text_color(fan::opengl::context_t* context, uint32_t i, const fan::color& color) {
					for (int j = get_index(i) * vertex_count; j < get_index(i + 1) * vertex_count; j++) {
						m_glsl_buffer.edit_ram_instance(
							context, 
							j,
							&color,
							element_byte_size,
							offset_color,
							sizeof(fan::color)
						);
					}
					m_queue_helper.edit(
						context,
						get_index(i) * vertex_count * element_byte_size + offset_color,
						get_index(i + 1) * (vertex_count)*element_byte_size - (element_byte_size - offset_color),
						&m_glsl_buffer
					);
				}
				void set_text_color(fan::opengl::context_t* context, uint32_t i, uint32_t j, const fan::color& color);

				f32_t get_text_alpha(fan::opengl::context_t* context, uint32_t i) {
					return ((fan::color*)m_glsl_buffer.get_instance(context, get_index(i) * vertex_count, element_byte_size, offset_color))->a;
				}
				void set_text_alpha(fan::opengl::context_t* context, uint32_t i, f32_t alpha) {
					for (int j = get_index(i) * vertex_count; j < get_index(i + 1) * vertex_count; j++) {
						m_glsl_buffer.edit_ram_instance(
							context, 
							j,
							&alpha,
							element_byte_size,
							offset_color + sizeof(fan::color::value_type) * 3,
							sizeof(f32_t)
						);
					}
					m_queue_helper.edit(
						context,
						get_index(i) * vertex_count * element_byte_size + offset_color + sizeof(fan::color::value_type) * 3,
						get_index(i + 1) * (vertex_count)*element_byte_size - (element_byte_size - offset_color - + sizeof(fan::color::value_type) * 3),
						&m_glsl_buffer
					);
				}

				fan::vec2 get_text_size(fan::opengl::context_t* context, uint32_t i) {

					uint32_t begin = 0;
					uint32_t end = 0;

					for (int j = 0; j < i; j++) {
						begin += m_store[j].m_text->size();
					}

					end = begin + m_store[i].m_text->size() - 1;

					auto p_first = get_letter_position(context, begin);
					auto p_last = get_letter_position(context, end);

					auto s_first = get_letter_position(context, begin);
					auto s_last = get_letter_position(context, end);

					return fan::vec2((p_last.x + s_last.x) - (p_first.x - s_first.x), font.line_height);
				}

				static fan::vec2 get_text_size(fan::opengl::context_t* context, const fan::utf16_string& text, f32_t font_size) {
					fan::vec2 text_size = 0;

					text_size.y = font.line_height;

					f32_t width = 0;

					for (int i = 0; i < text.size(); i++) {

						switch (text[i]) {
						case '\n': {
							text_size.x = std::max(width, text_size.x);
							text_size.y += font.line_height;
							width = 0;
							continue;
						}
						}

						auto letter = font.characters[text[i]];

						if (i == text.size() - 1) {
							width += letter.glyph.size.x;
						}
						else {
							width += letter.metrics.advance;
						}
					}

					text_size.x = std::max(width, text_size.x);

					return text_size * convert_font_size(context, font_size);
				}

				static f32_t get_original_font_size(fan::opengl::context_t* context) {
					return font.size;
				}

				inline static fan::font::font_t font;
				inline static fan::opengl::image_t* font_image = nullptr;

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

				void draw(fan::opengl::context_t* context, uint32_t begin = 0, uint32_t end = fan::uninitialized) {
					m_shader.use(context);
					m_shader.set_int(context, "texture_sampler", 0);
					context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
					context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, font_image->texture);
					const fan::vec2 viewport_size = context->viewport_size;

					fan::mat4 projection(1);
					projection = fan::math::ortho<fan::mat4>(
						(f32_t)viewport_size.x * 0.5,
						((f32_t)viewport_size.x + (f32_t)viewport_size.x * 0.5), 
						((f32_t)viewport_size.y + (f32_t)viewport_size.y * 0.5), 
						((f32_t)viewport_size.y * 0.5), 
						0.01,
						1000.0
					);

					fan::mat4 view(1);
					view = context->camera.get_view_matrix(view.translate(fan::vec3((f_t)viewport_size.x * 0.5, (f_t)viewport_size.y * 0.5, -700.0f)));

					uint32_t begin_ = get_index(begin) * vertex_count;
					uint32_t end_;
					if (end == fan::uninitialized) {
						end_ = letter_vertex_size() * vertex_count;
					}
					else {
						end_ = get_index(end) * vertex_count;
					}

					m_shader.use(context);
					m_shader.set_view(context, view);
					m_shader.set_projection(context, projection);
					m_glsl_buffer.draw(
						context,
						begin_,
						end_
					);
				}

			protected:

				void regenerate_indices() {

					for (int i = 0; i < m_store.size(); i++) {
						if (i == 0) {
							m_store[i].m_indices = m_store[i].m_text->size();
						}
						else {
							m_store[i].m_indices = m_store[i - 1].m_indices + m_store[i].m_text->size();
						}
					}
				}

				struct letter_t {
					fan::vec2 texture_position = 0;
					fan::vec2 texture_size = 0;
					std::array<fan::vec2, 4> texture_coordinates;

					fan::vec2 size = 0;
					fan::vec2 offset = 0;
				};

				letter_t get_letter(fan::opengl::context_t* context, wchar_t character, f32_t font_size) {

					auto letter_info = get_letter_info(context, character, font_size);

					letter_t letter;
					letter.texture_position = letter_info.glyph.position / font_image->size;
					letter.texture_size = letter_info.glyph.size / font_image->size;

					fan::vec2 src = letter.texture_position;
					fan::vec2 dst = src + letter.texture_size;

					letter.texture_coordinates = { {
							fan::vec2(src.x, src.y),
							fan::vec2(dst.x, src.y),
							fan::vec2(dst.x, dst.y),
							fan::vec2(src.x, dst.y)
						} };

					letter.size = letter_info.metrics.size / 2;
					letter.offset = letter_info.metrics.offset;

					return letter;
				}

				void push_letter(fan::opengl::context_t* context, wchar_t character, instance_t instance) {

					letter_t letter = get_letter(context, character, instance.font_size);

					instance.position += fan::vec2(letter.size.x, -letter.size.y) + fan::vec2(letter.offset.x, -letter.offset.y);
					instance.size = letter.size;

					uint32_t from = m_glsl_buffer.m_buffer.size();

					for (int i = 0; i < vertex_count; i++) {
						instance.texture_coordinates = fan_2d::opengl::convert_tc_4_2_6(&letter.texture_coordinates, i);
						m_glsl_buffer.push_ram_instance(context, &instance, element_byte_size);
					}

					uint32_t to = m_glsl_buffer.m_buffer.size();

					m_queue_helper.edit(
						context,
						from,
						to,
						&m_glsl_buffer
					);

					m_store_sprite.resize(m_store_sprite.size() + 1);

					m_store_sprite[m_store_sprite.size() - 1].m_texture = font_image->texture;
				}

				void insert_letter(fan::opengl::context_t* context, uint32_t i, wchar_t character, instance_t instance) {

					letter_t letter = get_letter(context, character, instance.font_size);

					instance.position += fan::vec2(letter.size.x, -letter.size.y) + fan::vec2(letter.offset.x, -letter.offset.y);
					instance.size = letter.size;

					uint32_t from = m_glsl_buffer.m_buffer.size();

					for (int j = 0; j < vertex_count; j++) {
						instance.texture_coordinates = fan_2d::opengl::convert_tc_4_2_6(&letter.texture_coordinates, j);
						m_glsl_buffer.insert_ram_instance(context, i * vertex_count + j, &instance, element_byte_size);
					}

					uint32_t to = m_glsl_buffer.m_buffer.size();

					m_queue_helper.edit(
						context,
						from,
						to,
						&m_glsl_buffer
					);

					store_sprite_t sst;
					sst.m_texture = font_image->texture;

					m_store_sprite.insert(i, sst);
				}

				constexpr uint32_t get_index(uint32_t i) const {
					return i == 0 ? 0 : m_store[i - 1].m_indices;
				}

				fan::vec2 get_letter_position(fan::opengl::context_t* context, uint32_t i) const {
					return *(fan::vec2*)m_glsl_buffer.get_instance(context, i * vertex_count, element_byte_size, offset_position);
				}
				void set_letter_position(fan::opengl::context_t* context, uint32_t i, const fan::vec2& position) {
					for (int j = 0; j < vertex_count; j++) {
						m_glsl_buffer.edit_ram_instance(
							context, 
							i * vertex_count + j,
							&position,
							element_byte_size,
							offset_position,
							sizeof(properties_t::position)
						);
					}
					m_queue_helper.edit(
						context,
						i * vertex_count * element_byte_size + offset_position,
						(i + 1) * (vertex_count)*element_byte_size - offset_position,
						&m_glsl_buffer
					);
				}
				uint32_t letter_vertex_size() const {
					return m_glsl_buffer.m_buffer.size() / element_byte_size / vertex_count;
				}

				struct store_t {
					fan::utf16_string_ptr_t m_text;
					fan::vec2 m_position;
					uint32_t m_indices;
					uint32_t m_new_lines;
				};

				struct store_sprite_t {
					uint32_t m_texture;
				};

				fan::hector_t<store_t> m_store;
				fan::hector_t<store_sprite_t> m_store_sprite;

				fan::shader_t m_shader;
				fan::opengl::core::glsl_buffer_t m_glsl_buffer;
				fan::opengl::core::queue_helper_t m_queue_helper;
				uint32_t m_draw_node_reference;

			};

		}
	}
}