#pragma once

#include <fan/graphics/shared_graphics.hpp>

#include <fan/graphics/graphics.hpp>

#include <fan/font.hpp>

#include <fan/utf_string.hpp>

#include <fan/physics/collision/rectangle.hpp>
#include <fan/physics/collision/circle.hpp>

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

						auto size_scale = fan::cast<f32_t>(this->m_camera->m_window->get_size()) / this->m_camera->m_window->get_previous_size();

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

				~text_renderer();

				void draw();

				void push_back(const fan::utf16_string& text, f32_t font_size, fan::vec2 position, const fan::color& text_color);

				void insert(uint32_t i, const fan::utf16_string& text, f32_t font_size, fan::vec2 position, const fan::color& text_color);

				fan::vec2 get_position(uint32_t i) const {
					return m_position[i];
				}
				void set_position(uint32_t i, const fan::vec2& position);

				uint32_t size() const;

				static fan::font::font_t get_letter_info(uint32_t c, f32_t font_size) {

					auto found = font_info.font.find(fan::utf16_string(c).data()[0]);

					if (found == font_info.font.end()) {
						throw std::runtime_error("failed to find character: " + std::to_string(fan::utf16_string(c).data()[0]));
					}

					f32_t converted_size = text_renderer::convert_font_size(font_size);

					return fan::font::font_t{
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

				f32_t get_font_size(uintptr_t i) const;
				void set_font_size(uint32_t i, f32_t font_size);

				void set_angle(uint32_t i, f32_t angle);

				static f32_t convert_font_size(f32_t font_size) {
					return font_size / font_info.size;
				}

				void erase(uintptr_t i);

				void erase(uintptr_t begin, uintptr_t end);

				void clear();

				static f32_t get_line_height(f32_t font_size) {
					return font_info.line_height * convert_font_size(font_size);
				}

				fan::utf16_string get_text(uint32_t i) const {
					return m_text[i];
				}
				void set_text(uint32_t i, const fan::utf16_string& text);

				fan::color get_text_color(uint32_t i, uint32_t j = 0);
				void set_text_color(uint32_t i, const fan::color& color);
				void set_text_color(uint32_t i, uint32_t j, const fan::color& color);

				static fan::vec2 get_text_size(const fan::utf16_string& text, f32_t font_size) {
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

				static f32_t get_average_text_height(const fan::utf16_string& text, f32_t font_size)
				{
					f32_t height = 0;

					for (const auto& i : text) {

						auto found = font_info.font.find(i);
						if (found == font_info.font.end()) {
							throw std::runtime_error("failed to find character: " + std::to_string(i));
						}

						height += (f_t)found->second.size.y;
					}

					return (height / text.size()) * convert_font_size(font_size);
				}


				static f32_t get_original_font_size() {
					return font_info.size;
				}

				inline static fan::font::font_info font_info;

				inline static fan_2d::graphics::image_t image;

				fan::camera* m_camera = nullptr;

				static int64_t get_new_lines(const fan::utf16_string& str)
				{
					int64_t new_lines = 0;
					const wchar_t* p = str.data();
					for (int i = 0; i < str.size(); i++) {
						if (p[i] == '\n') {
							new_lines++;
						}
					}

					return new_lines;
				}

				void write_data();

				void edit_data(uint32_t i);

				void edit_data(uint32_t begin, uint32_t end);

			protected:

				void insert_letter(uint32_t i, uint32_t j, wchar_t letter, f32_t font_size, fan::vec2& position, const fan::color& color, f32_t& advance);
				void push_back_letter(wchar_t letter, f32_t font_size, fan::vec2& position, const fan::color& color, f32_t& advance);

				std::vector<fan::utf16_string> m_text;
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

				fan::gpu_memory::uniform_handler* uniform_handler = nullptr;

				std::vector<VkDeviceSize> descriptor_offsets;

#endif

			};

			namespace base {

				struct button_metrics {

					std::vector<int64_t> m_new_lines;

				};

				// requires to have functions: 
				// get_camera() which returns camera, 
				// size() returns amount of objects, 
				// inside() if mouse inside
				struct mouse {

				protected:

					template <typename T>
					mouse(T& object) {

						std::memset(m_hover_button_id, -1, sizeof(m_hover_button_id));
						std::memset(m_held_button_id, -1, sizeof(m_held_button_id));


						// ------------------------------ key press

						m_key_press_it[0] = object.get_camera()->m_window->add_key_callback(fan::mouse_left, fan::key_state::press, [&] {
							for (uintptr_t i = 0; i < object.size(); i++) {
								if (object.inside(i)) {
									m_held_button_id[0] = i;
									if (m_on_click[0]) {
										m_on_click[0](i);
									}
								}
							}
							});

						m_key_press_it[1] = object.get_camera()->m_window->add_key_callback(fan::mouse_left, fan::key_state::press, [&] {
							for (uintptr_t i = 0; i < object.size(); i++) {
								if (object.inside(i)) {
									m_held_button_id[1] = i;
									if (m_on_click[1]) {
										m_on_click[1](i);
									}
								}
							}
							});

						// ------------------------------ key release

						m_key_release_it[0] = object.get_camera()->m_window->add_key_callback(fan::mouse_left, fan::key_state::release, [&] {

							if (m_held_button_id[0] != (uint32_t)fan::uninitialized && object.inside(m_held_button_id[0])) {
								if (m_on_release[0]) {
									m_on_release[0](m_held_button_id[0]);
								}

								m_held_button_id[0] = fan::uninitialized;
							}
							else if (m_held_button_id[0] != (uint32_t)fan::uninitialized && !object.inside(m_held_button_id[0])) {
								if (m_on_outside_release[0]) {
									m_on_outside_release[0](m_held_button_id[0]);
								}

								m_held_button_id[0] = fan::uninitialized;
							}

							});

						m_key_release_it[1] = object.get_camera()->m_window->add_key_callback(fan::mouse_left, fan::key_state::release, [&] {

							if (m_held_button_id[1] != (uint32_t)fan::uninitialized && object.inside(m_held_button_id[1])) {
								if (m_on_release[1]) {
									m_on_release[1](m_held_button_id[1]);
								}

								m_held_button_id[1] = fan::uninitialized;
							}
							else if (m_held_button_id[1] != (uint32_t)fan::uninitialized && !object.inside(m_held_button_id[1])) {
								if (m_on_outside_release[1]) {
									m_on_outside_release[1](m_held_button_id[1]);
								}

								m_held_button_id[1] = fan::uninitialized;
							}

							});

						// ------------------------------ hover

						object.get_camera()->m_window->add_mouse_move_callback([&](const fan::vec2&) {
							for (int k = 0; k < 2; k++) {
								for (uintptr_t i = 0; i < object.size(); i++) {
									if (object.inside(i) && i != m_hover_button_id[k] && m_hover_button_id[k] == (uint32_t)fan::uninitialized) {
										if (m_on_hover[k]) {
											m_on_hover[k](i);
										}

										m_hover_button_id[k] = i;
									}
								}
							}
							});

						// ------------------------------ exit

						object.get_camera()->m_window->add_mouse_move_callback([&](const fan::vec2i& position) {
							for (int k = 0; k < 2; k++) {
								if (m_hover_button_id[k] != (uint32_t)fan::uninitialized && !object.inside(m_hover_button_id[k]) && object.inside(m_hover_button_id[k], object.get_camera()->m_window->get_previous_mouse_position())) {
									if (m_on_exit[k]) {
										m_on_exit[k](m_hover_button_id[k]);
									}

									m_hover_button_id[k] = (uint32_t)fan::uninitialized;
								}
							}
							});

					}

					// keys must be same for click and release

					template <bool user_side>
					void on_click(std::function<void(uint32_t i)> function, uint16_t key = fan::mouse_left) {
						m_key_press_key[user_side] = key;
						m_key_press_it[user_side]->key = key;
						m_on_click[user_side] = function;

						m_key_release_it[user_side]->key = key;
						m_on_release[user_side] = function;
					}

					template <bool user_side>
					void on_release(std::function<void(uint32_t i)> function, uint16_t key = fan::mouse_left) {
						m_key_release_key[user_side] = key;
						m_key_release_it[user_side]->key = key;
						m_on_release[user_side] = function;

						m_key_press_key[user_side] = key;
						m_key_press_it[user_side]->key = key;
					}

					template <bool user_side>
					void on_outside_release(std::function<void(uint32_t i)> function, uint16_t key = fan::mouse_left) {
						m_key_release_key[user_side] = key;
						m_key_release_it[user_side]->key = key;
						m_on_outside_release[user_side] = function;

						m_key_press_key[user_side] = key;
						m_key_press_it[user_side]->key = key;
					}

					template <bool user_side>
					void on_hover(std::function<void(uint32_t i)> function) {
						m_on_hover[user_side] = function;
					}

					template <bool user_side>
					void on_exit(std::function<void(uint32_t i)> function) {
						m_on_exit[user_side] = function;
					}


				public:

					void on_click(std::function<void(uint32_t i)> function, uint16_t key = fan::mouse_left) {
						base::mouse::on_click<1>(function, key);
					}

					void on_release(std::function<void(uint32_t i)> function, uint16_t key = fan::mouse_left) {
						base::mouse::on_release<1>(function, key);
					}

					void on_outside_release(std::function<void(uint32_t i)> function, uint16_t key = fan::mouse_left) {
						base::mouse::on_outside_release<1>(function, key);
					}

					void on_hover(std::function<void(uint32_t i)> function) {
						base::mouse::on_hover<1>(function);
					}

					void on_exit(std::function<void(uint32_t i)> function) {
						base::mouse::on_exit<1>(function);
					}

				protected:

					std::function<void(uint32_t)> m_on_click[2];
					std::function<void(uint32_t)> m_on_release[2];
					std::function<void(uint32_t)> m_on_outside_release[2];
					std::function<void(uint32_t)> m_on_hover[2];
					std::function<void(uint32_t)> m_on_exit[2];

					uint32_t m_held_button_id[2];
					uint32_t m_hover_button_id[2];

					uint32_t m_key_press_key[2];
					uint32_t m_key_release_key[2];

					fan::window::key_callback_t* m_key_press_it[2];
					fan::window::key_callback_t* m_key_release_it[2];

				};

#define define_get_button_size \
							fan::vec2 get_button_size(uint32_t i) const \
				{ \
					const f32_t font_size = fan_2d::graphics::gui::text_renderer::get_font_size(i); \
																								  \
					f32_t h = (std::abs(fan_2d::graphics::gui::text_renderer::get_highest(font_size) + fan_2d::graphics::gui::text_renderer::get_lowest(font_size))); \
																																									\
					if (i < m_new_lines.size() && m_new_lines[i]) { \
						h += text_renderer::font_info.line_height * fan_2d::graphics::gui::text_renderer::convert_font_size(font_size) * m_new_lines[i]; \
					} \
					\
					return (fan_2d::graphics::gui::text_renderer::get_text(i).empty() ? fan::vec2(0, h) : fan::vec2(fan_2d::graphics::gui::text_renderer::get_text_size(fan_2d::graphics::gui::text_renderer::get_text(i), font_size).x, h)) + m_properties[i].border_size; \
				}

#define define_get_property_size \
				fan::vec2 get_size(properties_t properties) \
				{ \
					f32_t h = (std::abs(fan_2d::graphics::gui::text_renderer::get_highest(properties.font_size) + fan_2d::graphics::gui::text_renderer::get_lowest(properties.font_size))); \
																																															\
					int64_t new_lines = fan_2d::graphics::gui::text_renderer::get_new_lines(properties.text); \
						\
					if (new_lines) { \
						h += text_renderer::font_info.line_height * fan_2d::graphics::gui::text_renderer::convert_font_size(properties.font_size) * new_lines; \
					} \
						\
					return (properties.text.empty() ? fan::vec2(0, h) : fan::vec2(fan_2d::graphics::gui::text_renderer::get_text_size(properties.text, properties.font_size).x, h)) + properties.border_size; \
				}
			}

			enum class text_position_e {
				left,
				middle
			};

			struct button_properties {

				button_properties() {}

				button_properties(
					const fan::utf16_string& text,
					const fan::vec2& position
				) : text(text), position(position) {}

				fan::utf16_string text;

				fan::vec2 position;
				fan::vec2 border_size;

				f32_t font_size = fan_2d::graphics::gui::defaults::font_size;

				text_position_e text_position = text_position_e::middle;
			};

			struct rectangle_button_properties : public button_properties {

			};

			struct sprite_button_properties : public button_properties {

			};

			struct rectangle_text_box : 
				protected fan::class_duplicator<fan_2d::graphics::rectangle, 0>,
				protected fan::class_duplicator<fan_2d::graphics::rectangle, 1>,
				public fan_2d::graphics::gui::base::button_metrics,
				protected graphics::gui::text_renderer
			{

				using properties_t = rectangle_button_properties;

				using inner_rect_t = fan::class_duplicator<fan_2d::graphics::rectangle, 0>;
				using outer_rect_t = fan::class_duplicator<fan_2d::graphics::rectangle, 1>;

				rectangle_text_box(fan::camera* camera, fan_2d::graphics::gui::theme theme = fan_2d::graphics::gui::themes::deep_blue());

				void push_back(const rectangle_button_properties& properties);

				void draw(uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized);

				using fan_2d::graphics::rectangle::get_size;

				define_get_property_size;

				bool inside(uintptr_t i, const fan::vec2& position = fan::math::inf) const;

				fan::camera* get_camera();

				uintptr_t size() const;

				void write_data();

				void edit_data(uint32_t i);

				void edit_data(uint32_t begin, uint32_t end);

				using inner_rect_t::get_color;
				using inner_rect_t::set_color;

			protected:

				define_get_button_size;

				std::vector<rectangle_button_properties> m_properties;

				fan_2d::graphics::gui::theme theme;
			};

			struct rectangle_text_button :
				public fan_2d::graphics::gui::rectangle_text_box,
				public fan_2d::graphics::gui::base::mouse {

				rectangle_text_button(fan::camera* camera, fan_2d::graphics::gui::theme theme = fan_2d::graphics::gui::themes::deep_blue());

			};

			namespace text_box_properties {

				inline int blink_speed(500); // ms
				inline fan::color cursor_color(fan::colors::white);
				inline fan::color select_color(fan::colors::blue - fan::color(0, 0, 0, 0.5));

			}

			namespace font_properties {

				inline f32_t space_width(15); // remove pls

			}

			class sprite_text_box :
				protected fan::class_duplicator<fan_2d::graphics::sprite, 0>,
				public base::button_metrics,
				protected fan_2d::graphics::gui::text_renderer {

			public:

				using properties_t = sprite_button_properties;

				using sprite_t = fan::class_duplicator<fan_2d::graphics::sprite, 0>;

				sprite_text_box(fan::camera* camera, const std::string& path);

				void push_back(const sprite_button_properties& properties);

				void draw(uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized);

				define_get_property_size

					fan::camera* get_camera();

				uint64_t size() const;
				bool inside(uint32_t i, const fan::vec2& position = fan::math::inf) const;

				fan_2d::graphics::image_t image;

			protected:

				std::vector<sprite_button_properties> m_properties;

				define_get_button_size

			};

			struct sprite_text_button :
				public fan_2d::graphics::gui::sprite_text_box,
				public base::mouse {

				sprite_text_button(fan::camera* camera, const std::string& path);

			};

			template <typename T>
			class circle_slider : protected fan_2d::graphics::circle, protected fan_2d::graphics::rounded_rectangle {
			public:

				struct property_t {

					T min;
					T max;

					T current;

					fan::vec2 position;
					fan::vec2 box_size;
					f32_t box_radius;
					f32_t button_radius;
					fan::color box_color;
					fan::color button_color;

					fan::color text_color = fan_2d::graphics::gui::defaults::text_color;

					f32_t font_size = fan_2d::graphics::gui::defaults::font_size;
				};

				circle_slider(fan::camera* camera)
					: fan_2d::graphics::circle(camera), fan_2d::graphics::rounded_rectangle(camera), m_click_begin(fan::uninitialized), m_moving_id(fan::uninitialized)
				{
					camera->m_window->add_key_callback(fan::mouse_left, fan::key_state::press, [&] {

						// last ones are on the bottom
						for (uint32_t i = fan_2d::graphics::circle::size(); i-- ; ) {
							if (fan_2d::graphics::circle::inside(i)) {

								m_click_begin = fan_2d::graphics::rounded_rectangle::m_camera->m_window->get_mouse_position();
								m_moving_id = i;

								return;
							}
						}

						const fan::vec2 mouse_position = fan_2d::graphics::rounded_rectangle::m_camera->m_window->get_mouse_position();

						for (uint32_t i = fan_2d::graphics::circle::size(); i-- ; ) {

							const fan::vec2 box_position = fan_2d::graphics::rounded_rectangle::get_position(i);
							const fan::vec2 box_size = fan_2d::graphics::rounded_rectangle::get_size(i);

							const f32_t circle_diameter = fan_2d::graphics::circle::get_radius(i) * 2;

							const bool horizontal = box_size.x > box_size.y;

							if (fan_2d::collision::rectangle::point_inside_no_rotation(mouse_position, box_position - fan::vec2(horizontal ? 0 : circle_diameter / 2 - 2, horizontal ? circle_diameter / 2 - 2 : 0), fan::vec2(horizontal ? box_size.x : circle_diameter, horizontal ? circle_diameter : box_size.y))) {

								m_click_begin = fan_2d::graphics::rounded_rectangle::m_camera->m_window->get_mouse_position();
								m_moving_id = i;

								fan::vec2 circle_position = fan_2d::graphics::circle::get_position(i);

								if (horizontal) {
									fan_2d::graphics::circle::set_position(m_moving_id, fan::vec2(mouse_position.x, circle_position.y));
								}
								else {
									fan_2d::graphics::circle::set_position(m_moving_id, fan::vec2(circle_position.x, mouse_position.y));
								}

								circle_position = fan_2d::graphics::circle::get_position(i);

								f32_t min = get_min_value(m_moving_id);
								f32_t max = get_max_value(m_moving_id);

								f32_t length = box_size[!horizontal];

								T new_value = min + (((circle_position[!horizontal] - box_position[!horizontal]) / length) * (max - min));

								if (new_value == get_current_value(m_moving_id)) {
									return;
								}

								set_current_value(m_moving_id, new_value);

								for (int i = 0; i < 2; i++) {
									if (m_on_click[i]) {
										m_on_click[i](m_moving_id);
									}
									if (m_on_drag[i]) {
										m_on_drag[i](m_moving_id);
									}
								}

								this->edit_data(m_moving_id);
							}
						}
					});

					camera->m_window->add_key_callback(fan::mouse_left, fan::key_state::release, [&] {

						m_click_begin = fan::uninitialized;
						m_moving_id = fan::uninitialized;

					});

					camera->m_window->add_mouse_move_callback([&](const fan::vec2& position) {

						if (m_click_begin == fan::uninitialized) {
							return;
						}

						const fan::vec2 box_position = fan_2d::graphics::rounded_rectangle::get_position(m_moving_id);
						const fan::vec2 box_size = fan_2d::graphics::rounded_rectangle::get_size(m_moving_id);

						fan::vec2 circle_position = fan_2d::graphics::circle::get_position(m_moving_id);

						const bool horizontal = box_size.x > box_size.y;

						if (horizontal) {
						}
						else {
							circle_position.y = m_click_begin.y + (position.y - m_click_begin.y);
						}

						f32_t length = box_size[!horizontal];

						f32_t min = get_min_value(m_moving_id);
						f32_t max = get_max_value(m_moving_id);

						circle_position[!horizontal] = m_click_begin[!horizontal] + (position[!horizontal] - m_click_begin[!horizontal]);

						circle_position = circle_position.clamp(
							fan::vec2(box_position.x, box_position.x + box_size.x),
							fan::vec2(box_position.y, box_position.y + box_size.y)
						);

						T new_value = min + (((circle_position[!horizontal] - box_position[!horizontal]) / length) * (max - min));

						if (new_value == get_current_value(m_moving_id)) {
							return;
						}

						set_current_value(m_moving_id, new_value);

						fan_2d::graphics::circle::set_position(m_moving_id, circle_position);

						for (int i = 0; i < 2; i++) {
							if (m_on_drag[i]) {
								m_on_drag[i](m_moving_id);
							}
						}

						this->edit_data(m_moving_id);
					});
				}

				void push_back(const circle_slider<T>::property_t& property)
				{
					m_properties.emplace_back(property);
					m_properties[m_properties.size() - 1].current = fan::clamp(m_properties[m_properties.size() - 1].current, property.min, property.max);

					rounded_rectangle::properties_t properties;
					properties.position = property.position;
					properties.size = property.box_size;
					properties.radius = property.box_radius;
					properties.color = property.box_color;
					properties.angle = 0;

					fan_2d::graphics::rounded_rectangle::push_back(properties);

					if (property.box_size.x > property.box_size.y) {

						f32_t min = property.position.x;
						f32_t max = property.position.x + property.box_size.x;

						f32_t new_x = (f32_t(property.current - property.min) / (property.max - property.min)) * (max - min);

						fan_2d::graphics::circle::push_back(property.position + fan::vec2(new_x, property.box_size.y / 2), property.button_radius, property.button_color);
					}
					else {

						f32_t min = property.position.y;
						f32_t max = property.position.y + property.box_size.y;

						f32_t new_y = (f32_t(property.current - property.min) / (property.max - property.min)) * (max - min);

						fan_2d::graphics::circle::push_back(property.position + fan::vec2(property.box_size.x / 2, new_y), property.button_radius, property.button_color);
					}
				}

				void draw()
				{
					// depth test
					//fan_2d::graphics::draw([&] {
						fan_2d::graphics::rounded_rectangle::draw();
						fan_2d::graphics::circle::draw();
					//});
				}

				auto get_min_value(uint32_t i) const {
					return m_properties[i].min;
				}
				void set_min_value(uint32_t i, T value) {
					m_properties[i].min = value;
				}
				auto get_max_value(uint32_t i) const {
					return m_properties[i].max;
				}
				void set_max_value(uint32_t i, T value) {
					m_properties[i].max = value;
				}
				auto get_current_value(uint32_t i) const {
					return m_properties[i].current;
				}
				void set_current_value(uint32_t i, T value) {
					m_properties[i].current = value;
				}

				void on_drag(const std::function<void(uint32_t)>& function) {
					on_drag(true, function);
					m_on_drag[1] = function;
				}

				void write_data() {
					fan_2d::graphics::rounded_rectangle::write_data();
					fan_2d::graphics::circle::write_data();
				}
				void edit_data(uint32_t i) {
					fan_2d::graphics::rounded_rectangle::edit_data(i);
					fan_2d::graphics::circle::edit_data(i);
				}
				void edit_data(uint32_t begin, uint32_t end) {
					fan_2d::graphics::rounded_rectangle::edit_data(begin, end);
					fan_2d::graphics::circle::edit_data(begin, end);
				}

			protected:

				void on_drag(bool user, const std::function<void(uint32_t)>& function) {
					m_on_drag[user] = function;
				}

				void on_click(bool user, const std::function<void(uint32_t)>& function) {
					m_on_click[user] = function;
				}

				std::deque<circle_slider<T>::property_t> m_properties;

				fan::vec2 m_click_begin;

				uint32_t m_moving_id;

				std::function<void(uint32_t i)> m_on_drag[2]; // 0 lib, 1 user
				std::function<void(uint32_t)> m_on_click[2]; // 0 lib, 1 user

			};

			template <typename T>
			class circle_text_slider : public circle_slider<T>, protected fan_2d::graphics::gui::text_renderer {
			public:

				using value_type_t = T;

				static constexpr f32_t text_gap_multiplier = 1.5;

				circle_text_slider(fan::camera* camera) : circle_slider<T>(camera), fan_2d::graphics::gui::text_renderer(camera) {
					circle_text_slider::circle_slider::on_drag(false, [&](uint32_t i) {
						auto new_string = fan::to_wstring(this->get_current_value(i));

						bool resize = text_renderer::get_text(i * 3 + 2).size() != new_string.size();

						fan_2d::graphics::gui::text_renderer::set_text(i * 3 + 2, new_string);

						const fan::vec2 middle_text_size = fan_2d::graphics::gui::text_renderer::get_text_size(fan::to_wstring(this->get_current_value(i)), circle_text_slider::circle_slider::m_properties[i].font_size);

						const fan::vec2 position = circle_text_slider::circle_slider::rounded_rectangle::get_position(i);
						const fan::vec2 box_size = circle_text_slider::circle_slider::rounded_rectangle::get_size(i);

						const f32_t button_radius = circle_text_slider::circle_slider::circle::get_radius(i);

						fan::vec2 middle;

						if (box_size.x > box_size.y) {
							middle = position + fan::vec2(box_size.x / 2 - middle_text_size.x / 2, -middle_text_size.y - button_radius * text_gap_multiplier);
						}
						else {
							middle = position + fan::vec2(box_size.x / 2 + button_radius * text_gap_multiplier, box_size.y / 2 - middle_text_size.y / 2);
						}

						fan_2d::graphics::gui::text_renderer::set_position(i * 3 + 2, middle);

						if (resize) {
							circle_text_slider::text_renderer::write_data();
							circle_text_slider::circle_slider::edit_data(i);
						}
						else {
							this->edit_data(i);
						}
					});

					circle_text_slider::circle_slider::on_click(false, [&] (uint32_t i) {

						auto new_string = fan::to_wstring(this->get_current_value(i));

						bool resize = text_renderer::get_text(i * 3 + 2).size() != new_string.size();

						fan_2d::graphics::gui::text_renderer::set_text(i * 3 + 2, new_string);

						if (resize) {
							circle_text_slider::text_renderer::write_data();
							circle_text_slider::circle_slider::edit_data(i);
						}
						else {
							this->edit_data(i);
						}
					});
				}

				void push_back(const typename circle_slider<T>::property_t& property) {
					circle_text_slider::circle_slider::push_back(property);

					const fan::vec2 left_text_size = fan_2d::graphics::gui::text_renderer::get_text_size(fan::to_wstring(property.min), property.font_size);

					fan::vec2 left_or_up;

					if (property.box_size.x > property.box_size.y) {
						left_or_up = property.position - fan::vec2(left_text_size.x + property.button_radius * text_gap_multiplier, left_text_size.y / 2);
					}
					else {
						left_or_up = property.position - fan::vec2(left_text_size.x / 2, left_text_size.y + property.button_radius * text_gap_multiplier);
					}

					fan_2d::graphics::gui::text_renderer::push_back(
						fan::to_wstring(property.min), 
						property.font_size,
						left_or_up, 
						fan_2d::graphics::gui::defaults::text_color
					);

					const fan::vec2 right_text_size = fan_2d::graphics::gui::text_renderer::get_text_size(fan::to_wstring(property.min), property.font_size);

					fan::vec2 right_or_down;

					if (property.box_size.x > property.box_size.y) {
						right_or_down = property.position + fan::vec2(property.box_size.x + property.button_radius * text_gap_multiplier, -right_text_size.y / 2);
					}
					else {
						right_or_down = property.position + fan::vec2(-right_text_size.x / 2, property.box_size.y + property.button_radius * text_gap_multiplier);
					}

					fan_2d::graphics::gui::text_renderer::push_back(
						fan::to_wstring(property.max),
						property.font_size,
						right_or_down, 
						fan_2d::graphics::gui::defaults::text_color
					);

					const fan::vec2 middle_text_size = fan_2d::graphics::gui::text_renderer::get_text_size(fan::to_wstring(property.current), property.font_size);

					fan::vec2 middle;

					if (property.box_size.x > property.box_size.y) {
						middle = property.position + fan::vec2(property.box_size.x / 2 - middle_text_size.x / 2, -middle_text_size.y - property.button_radius * text_gap_multiplier);
					}
					else {
						middle = property.position + fan::vec2(property.box_size.x / 2 + property.button_radius * text_gap_multiplier, property.box_size.y / 2 - middle_text_size.y / 2);
					}

					fan_2d::graphics::gui::text_renderer::push_back(
						fan::to_wstring(property.current), 
						property.font_size,
						middle, 
						fan_2d::graphics::gui::defaults::text_color
					);
				}

				void draw() {
					// depth test
					//fan_2d::graphics::draw([&] {
						fan_2d::graphics::gui::circle_slider<value_type_t>::draw();
						fan_2d::graphics::gui::text_renderer::draw();
					//});
				}

				void write_data() {
					circle_text_slider::circle_slider::write_data();
					circle_text_slider::text_renderer::write_data();
				}
				void edit_data(uint32_t i) {
					circle_text_slider::circle_slider::edit_data(i);
					circle_text_slider::text_renderer::edit_data(i * 3, i * 3 + 2);
				}
				void edit_data(uint32_t begin, uint32_t end) {
					circle_text_slider::circle_slider::edit_data(begin, end);
					circle_text_slider::text_renderer::edit_data(begin * 3, end * 3 + 3); // ?
				}

			private:

			};

			class checkbox : 
				protected fan::class_duplicator<fan_2d::graphics::line, 99>,
				protected fan::class_duplicator<fan_2d::graphics::rectangle, 99>, 
				protected fan::class_duplicator<fan_2d::graphics::gui::text_renderer, 99>,
				public base::mouse {

			protected:

				using line_t = fan::class_duplicator<fan_2d::graphics::line, 99>;
				using rectangle_t = fan::class_duplicator<fan_2d::graphics::rectangle, 99>;
				using text_renderer_t = fan::class_duplicator<fan_2d::graphics::gui::text_renderer, 99>;

			public:

				struct properties_t {
					fan::vec2 position;

					f32_t font_size;

					fan::utf16_string text;

					uint8_t line_thickness = 2;

					f32_t box_size_multiplier = 1.5;

					bool checked = false;
				};

				checkbox(fan::camera* camera, fan_2d::graphics::gui::theme theme = fan_2d::graphics::gui::themes::deep_blue());

				void push_back(const checkbox::properties_t& property);

				void draw();

				void on_check(std::function<void(uint32_t i)> function);
				void on_uncheck(std::function<void(uint32_t i)> function);

				uint32_t size() const;
				bool inside(uint32_t i, const fan::vec2& position = fan::math::inf) const;

				fan::camera* get_camera();

				void write_data();
				void edit_data(uint32_t i);
				void edit_data(uint32_t begin, uint32_t end);


			protected:

				std::function<void(uint32_t)> m_on_check;
				std::function<void(uint32_t)> m_on_uncheck;

				fan_2d::graphics::gui::theme m_theme;

				std::deque<bool> m_visible;
				std::deque<checkbox::properties_t> m_properties;

			};

		}

	}

}