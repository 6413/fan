#pragma once

#include <fan/graphics/shared_graphics.hpp>

#include <fan/graphics/graphics.hpp>

namespace fan_2d {

	namespace graphics {

		namespace gui {

			static fan::vec2 get_resize_movement_offset(fan::window* window)
			{
				return fan::cast<f32_t>(window->get_size() - window->get_previous_size());
			}

			static void add_resize_callback(fan::window* window, fan::vec2& position) {
				window->add_resize_callback([&](const fan::vec2i&) {
					position += fan_2d::graphics::gui::get_resize_movement_offset(window);
					});
			}

			struct rectangle : public fan_2d::graphics::rectangle {

				rectangle(fan::camera* camera)
					: fan_2d::graphics::rectangle(camera)
				{
					this->m_camera->m_window->add_resize_callback([&](const fan::vec2i&) {

						auto offset = this->m_camera->m_window->get_size() - this->m_camera->m_window->get_previous_size();

						bool write_after = !fan::gpu_queue;

						auto size_scale = fan::cast<f32_t>(this->m_camera->m_window->get_size()) / this->m_camera->m_window->get_previous_size();

						fan::begin_queue();

						for (int i = 0; i < this->size(); i++) {

							auto new_scale = this->get_size(i) * size_scale;

							if (new_scale.x > 0 && new_scale.y > 0 && !std::isnan(new_scale.x) && !std::isnan(new_scale.y)
								&& new_scale.x != INFINITY && new_scale.y != INFINITY) {
								this->set_size(i, this->get_size(i) * size_scale);
								this->set_position(i, this->get_position(i) * size_scale);
							}
							else {
								this->set_size(i, 1);
							}

						}

						if (write_after) {
							fan::end_queue();
						}

						this->release_queue();

					});
				}

		};

			class text_renderer 
#if fan_renderer == fan_renderer_opengl

				:
				protected fan_2d::graphics::sprite,
				protected fan::buffer_object<f32_t, 99>,
				protected fan::buffer_object<fan::vec2, 99>

#elif fan_renderer == fan_renderer_vulkan

#endif
			
			{
			public:

				text_renderer(fan::camera* camera);

				void draw();

				void push_back(const fan::fstring& text, f32_t font_size, fan::vec2 position, const fan::color& text_color);

				void insert(uint32_t i, const fan::fstring& text, f32_t font_size, fan::vec2 position, const fan::color& text_color);

				fan::vec2 get_position(uint32_t i) const {
					return m_position[i];
				}
				void set_position(uint32_t i, const fan::vec2& position);

				uint32_t size() const;

				static fan::io::file::font_t get_letter_info(fan::fstring::value_type c, f32_t font_size) {
					auto found = font_info.font.find(c);

					if (found == font_info.font.end()) {
						throw std::runtime_error("failed to find character: " + std::to_string(c));
					}

					f32_t converted_size = text_renderer::convert_font_size(font_size);

					return fan::io::file::font_t{
						found->second.position * converted_size,
						found->second.size * converted_size,
						found->second.offset * converted_size,
						(found->second.advance * converted_size)
					};
				}

				f32_t get_lowest(f32_t font_size) const {
					auto found = font_info.font.find(font_info.lowest);

					return (found->second.offset.y + found->second.size.y) * this->convert_font_size(font_size);
				}

				static f32_t get_highest(f32_t font_size) {
					return std::abs(font_info.font.find(font_info.highest)->second.offset.y * text_renderer::convert_font_size(font_size));
				}

				static f32_t get_highest_size(f32_t font_size) {
					return font_info.font.find(font_info.highest)->second.size.y * text_renderer::convert_font_size(font_size);
				}

				static f32_t get_lowest_size(f32_t font_size) {
					return font_info.font.find(font_info.lowest)->second.size.y * text_renderer::convert_font_size(font_size);
				}

				fan::vec2 get_character_position(uint32_t i, uint32_t j, f32_t font_size) const {
					fan::vec2 position = text_renderer::get_position(i);

					auto converted_size = convert_font_size(font_size);

					for (int k = 0; k < j; k++) {
						position.x += font_info.font[m_text[i][k]].advance * converted_size;
					}

					position.y = i * (font_info.line_height * converted_size);

					return position;
				}

				f32_t get_font_size(uint_t i) const;
				void set_font_size(uint32_t i, f32_t font_size);

				void set_angle(uint32_t i, f32_t angle);

				static f32_t convert_font_size(f32_t font_size) {
					return font_size / font_info.size;
				}

				void erase(uint_t i);

				void erase(uint_t begin, uint_t end);

				void clear();

				static f32_t get_line_height(f32_t font_size) {
					return font_info.line_height * convert_font_size(font_size);
				}

				fan::fstring get_text(uint32_t i) const {
					return m_text[i];
				}
				void set_text(uint32_t i, const fan::fstring& text);

				fan::color get_text_color(uint32_t i, uint32_t j = 0);
				void set_text_color(uint32_t i, const fan::color& color);
				void set_text_color(uint32_t i, uint32_t j, const fan::color& color);

				static fan::vec2 get_text_size(const fan::fstring& text, f32_t font_size) {
					fan::vec2 length;

					f32_t current = 0;

					int new_lines = 0;

					uint32_t last_n = 0;

					for (int i = 0; i < text.size(); i++) {

						if (text[i] == '\n') {
							length.x = std::max((f32_t)length.x, current);
							length.y += font_info.line_height;
							new_lines++;
							current = 0;
							last_n = i;
							continue;
						}

						auto found = font_info.font.find(text[i]);
						if (found == font_info.font.end()) {
							throw std::runtime_error("failed to find character: " + std::to_string(text[i]));
						}

						current += found->second.advance;
						length.y = std::max((f32_t)length.y, font_info.line_height * new_lines + (f32_t)found->second.size.y);
					}

					length.x = std::max((f32_t)length.x, current);

					if (text.size()) {
						auto found = font_info.font.find(text[text.size() - 1]);
						if (found != font_info.font.end()) {
							length.x -= found->second.offset.x;
						}
					}

					length.y -= font_info.line_height * convert_font_size(font_size);

					f32_t average = 0;

					for (int i = last_n; i < text.size(); i++) {

						auto found = font_info.font.find(text[i]);
						if (found == font_info.font.end()) {
							throw std::runtime_error("failed to find character: " + std::to_string(text[i]));
						}

						average += found->second.size.y + found->second.offset.y;
					}

					average /= text.size() - last_n;

					length.y += average;

					return length * convert_font_size(font_size);
				}

				void release_queue(uint16_t avoid_flags = 0);

				static f32_t get_original_font_size() {
					return font_info.size;
				}

				inline static fan::io::file::font_info font_info;

				inline static std::unique_ptr<fan_2d::graphics::image_info> image;

				fan::camera* m_camera = nullptr;

			protected:

				void insert_letter(uint32_t i, uint32_t j, fan::fstring::value_type letter, f32_t font_size, fan::vec2& position, const fan::color& color, f32_t& advance);
				void push_back_letter(fan::fstring::value_type letter, f32_t font_size, fan::vec2& position, const fan::color& color, f32_t& advance);

				std::vector<fan::fstring> m_text;
				std::vector<fan::vec2> m_position;

				void regenerate_indices() {
					m_indices.clear();

					for (int i = 0; i < m_text.size(); i++) {
						if (m_indices.empty()) {
							m_indices.emplace_back(m_text[i].size());
						}
						else {
							m_indices.emplace_back(m_indices[i - 1] + m_text[i].size());
						}
					}
				}

				std::vector<uint32_t> m_indices;

#if fan_renderer == fan_renderer_opengl

				using font_size_t = fan::buffer_object<f32_t, 99>;
				using rotation_point_t = fan::buffer_object<fan::vec2, 99>;

				static constexpr auto location_font_size = "layout_font_size";
				static constexpr auto location_rotation_point = "layout_rotation_point";

#elif fan_renderer == fan_renderer_vulkan

				struct instance_t {

					fan::vec2 position;
					fan::vec2 size;
					f32_t angle;
					fan::color color;
					f32_t font_size;
					fan::vec2 text_rotation_point;
					std::array<fan::vec2, 6> texture_coordinate;

				};

				view_projection_t view_projection{};

				using instance_buffer_t = fan::gpu_memory::buffer_object<instance_t, fan::gpu_memory::buffer_type::buffer>;

				instance_buffer_t* instance_buffer = nullptr;

				uint32_t m_begin = 0;
				uint32_t m_end = 0;

				bool realloc_buffer = false;

				fan::gpu_memory::uniform_handler* uniform_handler = nullptr;

				VkDeviceSize descriptor_offset = 0;

#endif

			};

		}

	}

}