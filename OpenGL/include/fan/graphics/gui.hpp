#pragma once

#include <fan/graphics/graphics.hpp>


namespace fan_2d {

	namespace graphics {

		namespace gui {

			fan::vec2 get_resize_movement_offset(fan::window* window);

			void add_resize_callback(fan::window* window, fan::vec2& position);

			struct rectangle : public fan_2d::graphics::rectangle {

				rectangle(fan::camera* camera);

			};

			struct circle : public fan_2d::graphics::circle {

				circle(fan::camera* camera);

			};

			//struct sprite : public fan_2d::graphics::sprite {

			//	sprite(fan::camera* camera);
			//	// scale with default is sprite size
			//	sprite(fan::camera* camera, const std::string& path, const fan::vec2& position, const fan::vec2& size = 0, f_t transparency = 1);
			//	sprite(fan::camera* camera, unsigned char* pixels, const fan::vec2& position, const fan::vec2i& size = 0, f_t transparency = 1);

			//};


			namespace font_properties {

				inline f_t new_line(50);

				inline fan::color default_text_color(1);

				inline f_t space_width(15);

				inline f_t get_new_line(f_t font_size) {
					return new_line * font_size;
				}

			}

			inline std::unordered_map<fan::window_t, uint_t> current_focus;
			inline std::unordered_map<fan::window_t, uint_t> focus_counter;

			class text_renderer : 
				protected fan::basic_shape_color_vector_vector<3, fan::opengl_buffer_type::buffer_object, true>, 
				protected fan::basic_shape_color_vector_vector<3, fan::opengl_buffer_type::shader_storage_buffer_object, true>,
				public fan::texture_handler<>,
				public fan::vao_handler<>,
				public fan::glsl_location_handler<2, fan::opengl_buffer_type::buffer_object, true>,
				public fan::glsl_location_handler<0, fan::opengl_buffer_type::buffer_object, true>,
				public fan::glsl_location_handler<1, fan::opengl_buffer_type::buffer_object, true>{
			public:

				using text_color_t = fan::basic_shape_color_vector_vector<3, fan::opengl_buffer_type::buffer_object, true>;
				using outline_color_t = fan::basic_shape_color_vector_vector<3, fan::opengl_buffer_type::shader_storage_buffer_object, true>;

				using font_sizes_ssbo_t = fan::glsl_location_handler<2, fan::opengl_buffer_type::buffer_object, true>;

				using vertex_vbo_t = fan::glsl_location_handler<0, fan::opengl_buffer_type::buffer_object, true>;
				using texture_vbo_t = fan::glsl_location_handler<1, fan::opengl_buffer_type::buffer_object, true>;

				static constexpr auto vertex_location_name = "vertex";
				static constexpr auto text_color_location_name = "text_colors";
				static constexpr auto texture_coordinates_location_name = "texture_coordinate";
				static constexpr auto font_sizes_location_name = "font_sizes";

				text_renderer(fan::camera* camera);
				text_renderer(fan::camera* camera, const fan::fstring& text, const fan::vec2& position, const fan::color& text_color, f_t font_size, const fan::color& outline_color = fan::color(-1, -1, -1, 0), bool queue = false);

				text_renderer(const text_renderer& tr);
				text_renderer(text_renderer&& tr);

				text_renderer& operator=(const text_renderer& tr);
				text_renderer& operator=(text_renderer&& tr);

				fan::vec2 get_position(uint_t i) const;

				void set_position(uint_t i, const fan::vec2& position, bool queue = false);

				f_t get_font_size(uint_t i) const;
				void set_font_size(uint_t i, f_t font_size, bool queue = false);
				void set_text(uint_t i, const fan::fstring& text, bool queue = false);
				void set_text_color(uint_t i, const fan::color& color, bool queue = false);
				void set_outline_color(uint_t i, const fan::color& color, bool queue = false);

				fan::io::file::font_t get_letter_info(fan::fstring::value_type c, f_t font_size) const;
				fan::vec2 get_text_size(uint_t i) const {
					return get_text_size(get_text(i), get_font_size(i));
				}

				f_t get_longest_text() const {

					f_t longest = -fan::inf;

					for (uint_t i = 0; i < this->size(); i++) {
						longest = std::max(longest, (f_t)get_text_size(i).x);
					}

					return longest;
				}

				f_t get_highest_text() const {

					f_t highest = -fan::inf;

					for (uint_t i = 0; i < this->size(); i++) {
						highest = std::max(highest, (f_t)get_text_size(i).y);
					}

					return highest;
				}

				fan::vec2 get_text_size(const fan::fstring& text, f_t font_size) const;

				f_t get_lowest(f_t font_size) const;
				f_t get_highest(f_t font_size) const;

				f_t get_highest_size(f_t font_size) const;
				f_t get_lowest_size(f_t font_size) const;

				// i = string[i], j = string[i][j] (fan::fstring::value_type)
				fan::color get_color(uint_t i, uint_t j = 0) const;

				fan::fstring get_text(uint_t i) const;

				f_t convert_font_size(f_t font_size) const;

				void free_queue();
				void insert(uint_t i, const fan::fstring& text, const fan::vec2& position, const fan::color& text_color, f_t font_size, const fan::color& outline_color = fan::uninitialized, bool queue = false);
				void push_back(const fan::fstring& text, const fan::vec2& position, const fan::color& text_color, f_t font_size, const fan::color& outline_color = fan::uninitialized, bool queue = false);

				void draw() const;

				void erase(uint_t i, bool queue = false);
				void erase(uint_t begin, uint_t end, bool queue = false);

				uint_t size() const;

				void write_data();

				void release_queue(bool vertices, bool texture_coordinates, bool font_sizes);

				fan::io::file::font_info m_font_info;

				fan::camera* m_camera;

			private:

				void initialize_buffers();

				uint_t get_character_offset(uint_t i, bool special);

				std::vector<fan::vec2> get_vertices(uint_t i);
				std::vector<fan::vec2> get_texture_coordinates(uint_t i);

				void load_characters(uint_t i, fan::vec2 position, const fan::fstring& text, bool edit, bool insert);

				void edit_letter_data(uint_t i, uint_t j, const fan::fstring::value_type letter, const fan::vec2& position, int& advance, f_t converted_font_size);
				void insert_letter_data(uint_t i, const fan::fstring::value_type letter, const fan::vec2& position, int& advance, f_t converted_font_size);
				void write_letter_data(uint_t i, const fan::fstring::value_type letter, const fan::vec2& position, int& advance, f_t converted_font_size);

				void write_vertices();
				void write_texture_coordinates();
				void write_font_sizes();

				fan::shader m_shader;

				fan::vec2ui m_original_image_size;

				std::vector<fan::fstring> m_text;

				std::vector<fan::vec2> m_position;

				std::vector<std::vector<f_t>> m_font_size;
				std::vector<std::vector<fan::vec2>> m_vertices;
				std::vector<std::vector<fan::vec2>> m_texture_coordinates;

			};

			namespace text_box_properties {

				inline int blink_speed(500); // ms
				inline fan::color cursor_color(fan::colors::white);
				inline fan::color select_color(fan::colors::blue - fan::color(0, 0, 0, 0.5));

			}
			enum class e_text_position {
				left,
				middle
			};

			template <typename T>
			class base_box {
			public:

				using value_type = T;

				base_box(fan::camera* camera)
					: m_focus_begin(0), m_focus_end(fan::uninitialized), m_tr(camera), m_rv(camera) { }

				base_box(fan::camera* camera, const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& text_color)
					: m_focus_begin(0), m_focus_end(fan::uninitialized) , m_tr(camera, text, position, text_color, font_size), m_rv(camera){ }

				f_t get_highest(f_t font_size) const {
					return m_tr.get_highest(font_size);
				}

				f_t get_lowest(f_t font_size) const {
					return m_tr.get_lowest(font_size);
				}

				fan::fstring get_text(uint_t i) const {
					return m_tr.get_text(i);
				}

				fan::vec2 get_position(uint_t i) const {
					return m_rv.get_position(i);
				}

				fan::vec2 get_size(uint_t i) const {
					return m_rv.get_size(i);
				}

				void set_text(uint_t i, const fan::fstring& text, bool queue = false) {
					m_tr.set_text(i, text, queue);
				}

				fan::color get_box_color(uint_t i) const {
					return m_rv.get_color(i);
				}

				void set_box_color(uint_t i, const fan::color& color, bool queue = false) {
					m_rv.set_color(i, color, queue);
				}

				fan::color get_text_color(uint_t i) const {
					return m_tr.get_color(i);
				}

				void set_text_color(uint_t i, const fan::color& color, bool queue = false) {
					m_tr.set_text_color(i, color, queue);
				}

				f_t get_font_size(uint_t i) const {
					return m_tr.get_font_size(i);
				}

				void draw() {
					m_rv.draw();
					m_tr.draw();
				}

				void erase(uint_t i, bool queue = false) {
					base_box<T>::m_tr.erase(i, queue);
					base_box<T>::m_rv.erase(i, queue);
					base_box<T>::m_new_lines.erase(base_box<T>::m_new_lines.begin() + i);
					m_focus_id.erase(m_focus_id.begin() + i);
					m_border_size.erase(m_border_size.begin() + i);
				}

				void erase(uint_t begin, uint_t end, bool queue = false) {
					base_box<T>::m_tr.erase(begin, end, queue);
					base_box<T>::m_rv.erase(begin, end, queue);
					base_box<T>::m_new_lines.erase(base_box<T>::m_new_lines.begin() + begin, base_box<T>::m_new_lines.begin() + end);
					m_focus_id.erase(m_focus_id.begin() + begin, m_focus_id.begin() + end);
					m_border_size.erase(m_border_size.begin() + begin, m_border_size.begin() + end);
				}

				uint_t get_focus_id(uint_t i) const {
					return m_focus_id[i];
				}

				void set_focus_id(uint_t i, uint_t id) {
					m_focus_id[i] = id;
				}

				void set_focus_begin(uint_t begin) {
					m_focus_begin = begin;
				}

				void set_focus_end(uint_t end) {
					m_focus_end = end;
				}

				uint_t size() const {
					return base_box<T>::m_rv.size();
				}

			public:

				std::vector<uint_t> m_focus_id;

				uint_t m_focus_begin;
				uint_t m_focus_end;

				fan_2d::graphics::gui::text_renderer m_tr;
				T m_rv;

				std::vector<int64_t> m_new_lines;

				std::vector<fan::vec2> m_border_size;

			};

			template <typename T>
			class basic_sized_text_box : public base_box<T> {
			public:

				basic_sized_text_box(fan::camera* camera) 
					: base_box<T>(camera) { }

				basic_sized_text_box(fan::camera* camera, const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& text_color) 
					: base_box<T>(camera, text, font_size, position, text_color) { }

				fan::vec2 get_updated_box_size(uint_t i) {

					const f_t font_size = base_box<T>::m_tr.get_font_size(i);

					f_t h = font_size + (std::abs(base_box<T>::m_tr.get_highest(font_size) + base_box<T>::m_tr.get_lowest(font_size)));

					if (base_box<T>::m_new_lines[i]) {
						h += fan_2d::graphics::gui::font_properties::new_line * base_box<T>::m_tr.convert_font_size(font_size) * base_box<T>::m_new_lines[i];
					}

					return m_size[i] + base_box<T>::m_border_size[i];
				}

				// center, left
				void set_position(uint_t i, const fan::vec2& position, bool queue = false) {
					base_box<T>::m_rv.set_position(i, position, queue);
					auto h = (std::abs(this->get_highest(base_box<T>::m_tr.get_font_size(i)) + this->get_lowest(base_box<T>::m_tr.get_font_size(i)))) / 2;
					base_box<T>::m_tr.set_position(i, fan::vec2(position.x, position.y + m_size[i].y * 0.5 - h) +base_box<T>::m_border_size[i] * 0.5, queue);
				}

				fan::vec2 get_text_position(uint_t i, const fan::vec2& position) {
					auto h = (std::abs(this->get_highest(base_box<T>::m_tr.get_font_size(i)) + this->get_lowest(base_box<T>::m_tr.get_font_size(i)))) / 2;
					return fan::vec2(position.x, position.y + m_size[i].y * 0.5 - h + base_box<T>::m_border_size[i].y * 0.5);
				}

				/*void set_font_size(uint_t i, f_t font_size, bool queue = false) {
				base_box<T>::m_tr.set_font_size(i, font_size);

				auto h = (std::abs(this->get_highest(base_box<T>::get_font_size(i)) - this->get_lowest(base_box<T>::get_font_size(i)))) * 0.5;

				base_box<T>::m_tr.set_position(i, fan::vec2(base_box<T>::m_rv.get_position(i).x + m_border_size[i].x * 0.5, base_box<T>::m_rv.get_position(i).y + h + m_border_size[i].y * 0.5));

				update_box(i, queue);
				}*/

				fan::vec2 get_border_size(uint_t i) const {
					return base_box<T>::m_border_size[i];
				}

				void erase(uint_t i, bool queue = false) {
					m_size.erase(m_size.begin() + i);
					base_box<T>::erase(i, queue);
				}

				void erase(uint_t begin, uint_t end, bool queue = false) {
					m_size.erase(m_size.begin() + begin, m_size.begin() + end);
					base_box<T>::erase(begin, end, queue);
				}

				void update_box_size(uint_t i) {

					//base_box<T>::m_rv.set_size(i, get_updated_box_size(i, border_size));
				}

			protected:

				std::vector<fan::vec2> m_size;

			};

			template <typename T>
			class basic_text_box : public base_box<T> {
			public:

				basic_text_box(fan::camera* camera) 
					: base_box<T>(camera) { }

				basic_text_box(fan::camera* camera, const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& text_color) 
					: base_box<T>(camera, text, font_size, position, text_color) { }

				fan::vec2 get_updated_box_size(uint_t i) {

					const f_t font_size = base_box<T>::m_tr.get_font_size(i);

					f_t h = (std::abs(base_box<T>::m_tr.get_highest(font_size) + base_box<T>::m_tr.get_lowest(font_size)));

					if (i < base_box<T>::m_new_lines.size() && base_box<T>::m_new_lines[i]) {
						h += fan_2d::graphics::gui::font_properties::new_line * base_box<T>::m_tr.convert_font_size(font_size) * base_box<T>::m_new_lines[i];
					}

					return (base_box<T>::m_tr.get_text(i).empty() ? fan::vec2(0, h) : fan::vec2(base_box<T>::m_tr.get_text_size(base_box<T>::m_tr.get_text(i), font_size).x, h)) + base_box<T>::m_border_size[i];
				}

				void set_position(uint_t i, const fan::vec2& position, bool queue = false) {
					base_box<T>::m_rv.set_position(i, position, queue);
					base_box<T>::m_tr.set_position(i, fan::vec2(position.x, position.y) + base_box<T>::m_border_size[i] * 0.5, queue);
				}

				/*void set_font_size(uint_t i, f_t font_size, bool queue = false) {
				base_box<T>::m_tr.set_font_size(i, font_size);

				auto h = (std::abs(this->get_highest(base_box<T>::get_font_size(i)) - this->get_lowest(base_box<T>::get_font_size(i)))) * 0.5;

				base_box<T>::m_tr.set_position(i, fan::vec2(base_box<T>::m_rv.get_position(i).x + m_border_size[i].x * 0.5, base_box<T>::m_rv.get_position(i).y + h + m_border_size[i].y * 0.5));

				update_box(i, queue);
				}*/

				void erase(uint_t i, bool queue = false) {
					base_box<T>::erase(i, queue);
					base_box<T>::m_border_size.erase(base_box<T>::m_border_size.begin() + i);
				}

				fan::vec2 get_border_size(uint_t i) const {
					return base_box<T>::m_border_size[i];
				}

				void update_box_size(uint_t i) {
					base_box<T>::m_rv.set_size(i, get_updated_box_size(i));
				}

			};

			template <typename T>
			class text_box_mouse_input {
			public:

				text_box_mouse_input(T& rv, std::vector<uint_t>& focus_id) : m_rv(rv), m_focus_id(focus_id) {}

				bool inside(uint_t i) const {
					return m_rv.inside(i);
				}

				void on_touch(std::function<void(uint_t j)> function) {
					m_on_touch = function;

					m_rv.m_camera->m_window->add_mouse_move_callback([&] {
						for (uint_t i = 0; i < m_rv.size(); i++) {
							if (m_rv.inside(i)) {
								m_on_touch(i);
							}
						}
					});
				}

				void on_exit(std::function<void(uint_t j)> function) {
					m_on_exit = function;

					m_rv.m_camera->m_window->add_mouse_move_callback([&] {
						for (uint_t i = 0; i < m_rv.size(); i++) {
							if (!inside(i)) {
								m_on_exit(i);
							}
						}
					});
				}

				void on_click(std::function<void(uint_t i)> function, fan::input key = fan::mouse_left) {
					if (m_on_click) {
						m_on_click = function;
					}
					else {
						m_on_click = function;

						m_rv.m_camera->m_window->add_key_callback(key, [&] {
							for (uint_t i = 0; i < m_rv.size(); i++) {
								if (m_rv.inside(i)) {

									current_focus[m_rv.m_camera->m_window->get_handle()] = m_focus_id[i];

									m_on_click(i);
									break;
								}
							}
						});
					}
				}

				void on_release(std::function<void(uint_t i)> function, uint16_t key = fan::mouse_left) {
					m_on_release = function;

					m_rv.m_camera->m_window->add_key_callback(key, [&] {
						for (uint_t i = 0; i < m_rv.size(); i++) {
							m_on_release(i);
						}
					}, true);
				}

			protected:

				std::function<void(uint_t i)> m_on_touch;
				std::function<void(uint_t i)> m_on_exit;
				std::function<void(uint_t i)> m_on_click;
				std::function<void(uint_t i)> m_on_release;

			protected:

				T& m_rv;

				std::vector<uint_t>& m_focus_id;

			};

			template <typename box_type>
			class text_box_keyboard_input : public box_type {
			public:

				text_box_keyboard_input(fan::camera* camera)
					:	box_type(camera), m_text_visual_input(camera) {

					box_type::m_tr.m_camera->m_window->add_keys_callback([&](fan::fstring::value_type key) {
						for (uint_t j = 0; j < m_callable.size(); j++) {
							bool break_loop = handle_input(m_callable[j], key);
							if (break_loop) {
								break;
							}
						}
					});
				}

				text_box_keyboard_input(fan::camera* camera, const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& text_color) 
					:	box_type(camera, text, font_size, position, text_color), m_text_visual_input(camera) {

					box_type::m_tr.m_camera->m_window->add_keys_callback([&](fan::fstring::value_type key) {
						for (uint_t j = 0; j < m_callable.size(); j++) {
							bool break_loop = handle_input(m_callable[j], key);
							if (break_loop) {
								break;
							}
						}
					});
				}

				void set_text(uint_t i, const fan::fstring& text, bool queue = false) {
					box_type::set_text(i, text, queue);
					update_box(i);
				}

				void set_cursor_visible(uint_t i, bool state = true) {
					m_text_visual_input.m_visible[i] = state;
					m_text_visual_input.m_timer[i].restart();
					update_cursor_position(i);
				}

				void push_back(uint_t i) {

					const auto& str = box_type::m_tr.get_text(i);

					m_text_visual_input.m_cursor.push_back(fan::vec2(), fan::vec2(), text_box_properties::cursor_color);

					m_text_visual_input.m_timer.emplace_back(fan::timer<>(fan::timer<>::start(), text_box_properties::blink_speed));
					m_text_visual_input.m_visible.emplace_back(false);
					m_line_offset.resize(i + 1);
					m_line_offset[i].emplace_back(0);
					m_starting_line.resize(i + 1);

					m_characters_per_line.resize(i + 1);

					uint_t new_lines = 0;
					uint_t characters_per_line = 0;

					for (std::size_t j = 0; j < str.size(); j++) {
						characters_per_line++;
						if (str[j] == '\n') {
							m_characters_per_line[i].emplace_back(characters_per_line);
							m_line_offset[i].emplace_back(characters_per_line);
							new_lines++;
							characters_per_line = 0;
						}
					}

					m_characters_per_line[i].emplace_back(characters_per_line);
					m_current_character.emplace_back(characters_per_line);

					text_box_keyboard_input::base_box::m_new_lines.emplace_back(new_lines);
					m_current_line.emplace_back(new_lines);

					m_text_visual_input.m_select.resize(new_lines + 1, text_box_properties::select_color);
					uint32_t previous_size = m_starting_select_character.size();
					m_starting_select_character.resize(new_lines + 1);
					std::fill(m_starting_select_character.begin() + previous_size, m_starting_select_character.end(), INT64_MAX);
				}

				// returns begin and end of cursor points
				fan::mat2 get_cursor_position(uint_t i, uint_t beg = fan::uninitialized, uint_t n = fan::uninitialized) const {

					const auto& str = box_type::m_tr.get_text(i);

					const auto font_size = box_type::m_tr.get_font_size(i);

					auto converted = this->box_type::m_tr.convert_font_size(font_size);

					f_t x = 0;
					f_t y = 0;

					const std::size_t b = beg == (std::size_t)-1 ? 0 : beg;

					const std::size_t n_ = n == (std::size_t)-1 ? str.size() : n;

					for (std::size_t j = b; j < b + n_ && j < str.size(); j++) {

						if (str[j] == '\n') {
							continue;
						}

						auto found = box_type::m_tr.m_font_info.m_font.find(str[j]);
						if (found != box_type::m_tr.m_font_info.m_font.end()) {
							x += found->second.m_advance * converted;
						}

					}

					const f_t new_line_size = font_properties::get_new_line(box_type::m_tr.convert_font_size(font_size));

					y += m_current_line[i] * new_line_size;

					return fan::mat2(
						fan::vec2(x, y), 
						fan::vec2(x, y + new_line_size)
					) + box_type::m_rv.get_position(i) + box_type::m_border_size[i] * 0.5;
				}

				void update_cursor_position(uint_t i) {
					if (i >= m_text_visual_input.m_cursor.size()) {
						return;
					}

					const fan::mat2 cursor_position = this->get_cursor_position(i, m_line_offset[i][m_current_line[i]], m_current_character[i]);

					m_text_visual_input.m_cursor.set_line(i, cursor_position[0], cursor_position[1]);
				}

				void draw() {

					if (!m_text_visual_input.m_cursor.size()) {
						return;
					}

					bool found = false;
					uint_t draw_id = 0;

					for (uint_t i = 0; i < box_type::m_focus_id.size(); i++) {
						if (box_type::m_focus_id[i] == current_focus[box_type::m_tr.m_camera->m_window->get_handle()]) {
							found = true;
							draw_id = i;
							break;
						}
					}

					auto cfound = std::find(m_callable.begin(), m_callable.end(), draw_id);

					if (!found || cfound == m_callable.end()) {
						return;
					}

					//for (uint_t i = 0; i < m_text_visual_input.m_cursor.size(); i++) {

					if (m_text_visual_input.m_timer[draw_id].finished()) {
						m_text_visual_input.m_visible[draw_id] = !m_text_visual_input.m_visible[draw_id];
						m_text_visual_input.m_timer[draw_id].restart();
					}
					//}

					//if (m_callable.size() == m_text_visual_input.m_cursor.size()) {
					//	if (m_text_visual_input.m_visible[draw_id]) {
					//		m_text_visual_input.m_cursor.draw(draw_id);
					//	}	
					//}
					//else { // in case we dont want to draw input for some window
					//	for (std::size_t i = 0; i < m_callable.size(); i++) {
					if (m_text_visual_input.m_visible[draw_id]) {
						m_text_visual_input.m_cursor.draw(draw_id);
					}

					m_text_visual_input.m_select.draw();

					//if (m_text_visual_input.m_select.get_size(draw_id) != 0) {
					//}
					/*	}
					}*/
				}

				void update_box(uint_t i) {
					m_characters_per_line[i].clear();
					m_line_offset[i].clear();

					m_line_offset[i].emplace_back(0);
					m_characters_per_line[i].clear();

					uint_t new_lines = 0;
					uint_t characters_per_line = 0;

					auto str = box_type::m_tr.get_text(i);

					for (std::size_t j = 0; j < str.size(); j++) {
						characters_per_line++;
						if (str[j] == '\n') {
							m_characters_per_line[i].emplace_back(characters_per_line);
							m_line_offset[i].emplace_back(characters_per_line);
							new_lines++;
							characters_per_line = 0;
						}
					}

					m_characters_per_line[i].emplace_back(characters_per_line);

					text_box_keyboard_input::base_box::m_new_lines[i] = new_lines;
					m_current_line[i] = new_lines;

					box_type::update_box_size(i);
					update_cursor_position(i);
				}

				bool key_press(fan::input key) {
					return box_type::m_tr.m_camera->m_window->key_press(key);
				}

				void set_selected_size(uint_t i) {

					int diff = m_current_character[i] - m_starting_select_character[m_current_line[i]];

					if (diff) {
						//	fan::print("a", m_text_visual_input.m_select.get_position(0), m_text_visual_input.m_select.get_position(1));
					}

					int character_min = std::min(m_current_character[i], m_starting_select_character[m_current_line[i]]);
					int character_max = std::max(m_current_character[i], m_starting_select_character[m_current_line[i]]);

					int line_min = std::min(m_starting_line[i], m_current_line[i]);
					int line_max = std::max(m_starting_line[i], m_current_line[i]) + 1;

					for (int j = line_min; j < line_max; j++) {
						if (m_starting_line[i] <= m_current_line[i]) {
							if (j != line_max - 1) {
								m_text_visual_input.m_select.set_size(
									j,
									fan::vec2(
									box_type::m_tr.get_text_size(box_type::m_tr.get_text(i).substr(m_line_offset[i][j] + character_min, m_characters_per_line[i][j]), box_type::m_tr.get_font_size(i)).x,
									font_properties::get_new_line(box_type::m_tr.convert_font_size(box_type::m_tr.get_font_size(i)))
								)
								);
							}
							else {

								m_text_visual_input.m_select.set_size(
									j, 
									fan::vec2(
									(diff < 0 ? -1 : 1) * box_type::m_tr.get_text_size(box_type::m_tr.get_text(i).substr(m_line_offset[i][j] + character_min, character_max - character_min), box_type::m_tr.get_font_size(i)).x,
									font_properties::get_new_line(box_type::m_tr.convert_font_size(box_type::m_tr.get_font_size(i)))
								)
								);
							}
						}
						else {
							m_text_visual_input.m_select.set_size(
								j, 
								fan::vec2(
								(diff < 0 ? -1 : 1) * box_type::m_tr.get_text_size(box_type::m_tr.get_text(i).substr(m_line_offset[i][m_current_line[i]] + line_min, fan::abs(diff)), box_type::m_tr.get_font_size(i)).x,
								font_properties::get_new_line(box_type::m_tr.convert_font_size(box_type::m_tr.get_font_size(i)))
							)
							);
						}

					}

				}

				void update_selected_position(uint_t i) {
					m_text_visual_input.m_select.set_position(i, get_cursor_position(i, m_line_offset[i][m_current_line[i]], m_current_character[i])[0]);
				}

				bool handle_input(uint_t i, uint_t key) {
					if (current_focus[box_type::m_tr.m_camera->m_window->get_handle()] != box_type::m_focus_id[i]) {
						return false;
					}

					auto str = box_type::m_tr.get_text(i);

					auto current_key = box_type::m_tr.m_camera->m_window->get_current_key();

					bool replace_selected_text = false;
					bool paste = false;

					switch (current_key) {
						case fan::key_v: 
						{
							paste = true;
							goto g_delete;
g_paste:
							paste = false;

							disable_select_and_reset(m_current_line[i]);

							if (this->key_press(fan::key_control)) {

								str = fan::io::get_clipboard_text(box_type::m_tr.m_camera->m_window->get_handle());

								auto old_text = box_type::m_tr.get_text(i);

								old_text.insert(old_text.begin() + m_current_character[i], str.begin(), str.end());

								box_type::m_tr.set_text(i, old_text);

								m_current_character[i] += str.size();

								update_box(i);

							}
							else {
								goto g_add_key;
							}

							break;
						}
						case fan::key_delete: {

g_delete:

							if (str.size()) {

								uint32_t count = m_starting_select_character[m_current_line[i]] == INT64_MAX ? 1 : std::abs(m_starting_select_character[m_current_line[i]] - m_current_character[i]);

								if (m_starting_select_character[m_current_line[i]] < m_current_character[i]) {
									m_current_character[i] = m_starting_select_character[m_current_line[i]];
								}

								for (uint32_t k = 0; k < count; k++) {
									if (str[m_line_offset[i][m_current_line[i]] + m_current_character[i]] == '\n') {
										m_characters_per_line[i][m_current_line[i]] += m_characters_per_line[i][m_current_line[i] + 1] - 1;
										m_characters_per_line[i].erase(m_characters_per_line[i].begin() + m_current_line[i] + 1);
										m_line_offset[i][m_current_line[i] + 1] = m_line_offset[i][m_current_line[i]];
										m_line_offset[i].erase(m_line_offset[i].begin() + m_current_line[i]);
										text_box_keyboard_input::base_box::m_new_lines[i]--;
									}
									else {
										m_characters_per_line[i][m_current_line[i]]--;
									}

									if ((uint_t)(m_line_offset[i][m_current_line[i]] + m_current_character[i]) >= str.size()) {
										break;
									}

									str.erase(str.begin() + m_line_offset[i][m_current_line[i]] + m_current_character[i]);

									box_type::m_tr.set_text(i, str);

									for (int j = m_current_line[i] + 1; j <= text_box_keyboard_input::base_box::m_new_lines[i]; j++) {
										m_line_offset[i][j]--;
									}
								}

								disable_select_and_reset(m_current_line[i]);

								update_cursor_position(i);
								box_type::update_box_size(i);
							}

							if (paste) {
								goto g_paste;
							}

							if (replace_selected_text) {
								goto g_add_key;
							}

							break;
						}
						case fan::key_backspace:
						{

							if ((m_current_character[i] || m_current_line[i]) || m_starting_select_character[m_current_line[i]] != INT64_MAX) {

								if (m_starting_select_character[m_current_line[i]] != INT64_MAX && m_starting_select_character[m_current_line[i]] < m_current_character[i]) {
									goto g_delete;
								}

								uint32_t count = m_starting_select_character[m_current_line[i]] == INT64_MAX ? 1 : std::abs(m_starting_select_character[m_current_line[i]] - m_current_character[i]);

								m_current_character[i] = m_starting_select_character[m_current_line[i]] == INT64_MAX ? m_current_character[i] : m_starting_select_character[m_current_line[i]];

								for (uint32_t j = 0; j < count; j++) {
									m_current_character[i]--;

									str.erase(str.begin() + m_line_offset[i][m_current_line[i]] + m_current_character[i]);

									box_type::m_tr.set_text(i, str);

									// previous line
									if (m_current_line[i] && m_current_character[i] == -1) {
										m_current_character[i] = m_characters_per_line[i][m_current_line[i] - 1] - 1;
										m_characters_per_line[i][m_current_line[i] - 1] += m_characters_per_line[i][m_current_line[i]];
										m_characters_per_line[i].erase(m_characters_per_line[i].begin() + m_current_line[i]);
										m_line_offset[i].erase(m_line_offset[i].begin() + m_current_line[i]);
										m_current_line[i]--;
										text_box_keyboard_input::base_box::m_new_lines[i]--;
									}
									else if (m_current_character[i] == -1) {
										m_current_character[i] = 0;
									}

									if (m_characters_per_line[i][m_current_line[i]]) {
										m_characters_per_line[i][m_current_line[i]]--;
									}

									if (m_current_line[i] || m_characters_per_line[i][m_current_line[i]]) {
										for (int j = m_current_line[i] + 1; j <= text_box_keyboard_input::base_box::m_new_lines[i]; j++) {
											m_line_offset[i][j]--;
										}
									}
								}

								disable_select_and_reset(m_current_line[i]);

								update_cursor_position(i);
								box_type::update_box_size(i);
							}

							break;
						}
						case fan::key_left:
						{

							m_text_visual_input.m_visible[i] = true;

							bool go_once = false;

							if (key_press(fan::key_shift) && m_starting_select_character[m_current_line[i]] == INT64_MAX) {

								m_starting_select_character[m_current_line[i]] = m_current_character[i];

								update_selected_position(i);
							}

							if (this->key_press(fan::key_control)) {

								std::size_t found = -1;

								const auto offset = m_line_offset[i][m_current_line[i]];

								if (m_current_character[i] - 1 >= 0 && (uint_t)offset < str.size()) {
									auto str_ = str.substr(offset, m_current_character[i] ? m_current_character[i] - 1 : m_current_character[i]);
									found = str_.find_last_of(L' ');
								}

								if (found != std::string::npos) {
									m_current_character[i] = found + 1;

									if (str[m_line_offset[i][m_current_line[i]] + m_current_character[i]] == ' ') {
										while (str[m_line_offset[i][m_current_line[i]] + m_current_character[i] - 1] == ' ') {
											m_current_character[i]--;
										}
										while (m_current_character[i] > 0 && str[m_line_offset[i][m_current_line[i]] + m_current_character[i] - 1] != ' ') {
											m_current_character[i]--;
										}
									}
								}
								else if (
									m_current_character[i] - 1 <= -1) {
									go_once = true;
								}
								else {
									m_current_character[i] = 0;
								}
							}
							else {
								go_once = true;
							}

							if (go_once) {
								m_current_character[i]--;
								if (m_current_line[i] && m_current_character[i] == -1) {
									m_current_line[i]--;
									m_current_character[i] = m_characters_per_line[i][m_current_line[i]] - 1;
								}
								else if (m_current_character[i] == -1) {
									m_current_character[i] = 0;
								}
							}

							if (!this->key_press(fan::key_shift)) {
								disable_select_and_reset(m_current_line[i]);
							}
							else {
								set_selected_size(i);
							}

							update_cursor_position(i);

							break;
						}
						case fan::key_right: {

							if (key_press(fan::key_shift) && m_starting_select_character[m_current_line[i]] == INT64_MAX) {
								m_starting_select_character[m_current_line[i]] = m_current_character[i];
								auto cursor_position = get_cursor_position(i, m_line_offset[i][m_current_line[i]], m_current_character[i]);

								m_text_visual_input.m_select.set_position(i, cursor_position[0]);
							}

							m_text_visual_input.m_visible[i] = true;

							if (m_current_line[i] == *(text_box_keyboard_input::base_box::m_new_lines.end() - 1) && m_characters_per_line[i][m_current_line[i]] <= m_current_character[i]) {
								break;
							}

							bool go_once = false;

							if (box_type::m_tr.m_camera->m_window->key_press(fan::key_control)) {

								std::size_t found = -1;

								const auto offset = m_line_offset[i][m_current_line[i]] + m_current_character[i] + 1;

								if ((uint_t)offset < str.size()) {
									found = str.substr(offset, m_characters_per_line[i][m_current_line[i]] - m_current_character[i]).find_first_of(L' ');
								}

								if (found != std::string::npos) {
									m_current_character[i] = m_current_character[i] + found + 1;
									if (str[m_line_offset[i][m_current_line[i]] + m_current_character[i] - 1] == ' ') {
										while (str[m_line_offset[i][m_current_line[i]] + m_current_character[i]] == ' ') {
											m_current_character[i]++;
										}
										while (m_current_character[i] < m_characters_per_line[i][m_current_line[i]] && str[m_line_offset[i][m_current_line[i]] + m_current_character[i]] != ' ') {
											m_current_character[i]++;
										}
									}
								}
								else if (m_current_character[i] + 1 >= m_characters_per_line[i][m_current_line[i]]) {
									go_once = true;
								}
								else {
									m_current_character[i] = m_characters_per_line[i][m_current_line[i]];
								}
							}
							else {
								go_once = true;
							}

							if (go_once) {
								const auto offset = m_line_offset[i][m_current_line[i]] + m_current_character[i];

								if (m_current_character[i] >= m_characters_per_line[i][m_current_line[i]] || (*(str.begin() + offset) == '\n')) {
									m_current_character[i] = 0;
									m_current_line[i]++;
								}
								else {
									m_current_character[i] = std::clamp(++m_current_character[i], (int64_t)0, (int64_t)m_characters_per_line[i][m_current_line[i]]);
								}
							}

							if (!this->key_press(fan::key_shift)) {
								disable_select_and_reset(m_current_line[i]);
							}
							else {
								set_selected_size(i);
							}

							update_cursor_position(i);

							break;
						}
						case fan::key_home:
						{
							if (key_press(fan::key_shift) && m_starting_select_character[m_current_line[i]] == INT64_MAX) {
								m_starting_select_character[m_current_line[i]] = m_current_character[i];
								auto cursor_position = get_cursor_position(i, m_line_offset[i][m_current_line[i]], m_current_character[i]);

								m_text_visual_input.m_select.set_position(i, cursor_position[0]);
							}

							m_text_visual_input.m_visible[i] = true;

							if (box_type::m_tr.m_camera->m_window->key_press(fan::key_control)) {
								m_current_line[i] = 0;
							}

							m_current_character[i] = 0;

							if (!this->key_press(fan::key_shift)) {
								disable_select_and_reset(m_current_line[i]);
							}
							else {
								set_selected_size(i);
							}

							update_cursor_position(i);

							break;
						}
						case fan::key_end:
						{
							if (key_press(fan::key_shift) && m_starting_select_character[m_current_line[i]] == INT64_MAX) {
								m_starting_select_character[m_current_line[i]] = m_current_character[i];
								auto cursor_position = get_cursor_position(i, m_line_offset[i][m_current_line[i]], m_current_character[i]);

								m_text_visual_input.m_select.set_position(i, cursor_position[0]);
							}

							m_text_visual_input.m_visible[i] = true;

							if (box_type::m_tr.m_camera->m_window->key_press(fan::key_control)) {
								m_current_line[i] = text_box_keyboard_input::base_box::m_new_lines[i];
							}

							auto b = text_box_keyboard_input::base_box::m_new_lines[i] && m_characters_per_line[i][m_current_line[i]] && str[m_line_offset[i][m_current_line[i]] + m_characters_per_line[i][m_current_line[i]] - 1] == '\n';

							m_current_character[i] = m_characters_per_line[i][m_current_line[i]] - b;

							if (!this->key_press(fan::key_shift)) {
								disable_select_and_reset(m_current_line[i]);
							}
							else {
								set_selected_size(i);
							}

							update_cursor_position(i);

							break;
						}
						case fan::key_up:
						{
							disable_select_and_reset(m_current_line[i]);

							m_text_visual_input.m_visible[i] = true;

							if (m_current_line[i] > 0) {

								f_t fclosest = fan::inf;

								f_t current = 0;

								for (int j = 0; j < m_current_character[i]; j++) {
									auto c = *(str.begin() + m_line_offset[i][m_current_line[i]] + j);

									if (c == '\n') {
										continue;
									}

									auto l_info = box_type::m_tr.get_letter_info(c, box_type::m_tr.get_font_size(i));
									current += l_info.m_advance;
								}

								m_current_line[i]--;

								f_t new_current = 0;
								std::size_t iclosest = 0;

								if (fan::distance(new_current, current) < fclosest) {
									fclosest = fan::distance(new_current, current);
									iclosest = 0;
								}

								for (int j = m_line_offset[i][m_current_line[i]]; j < m_line_offset[i][m_current_line[i]] + m_characters_per_line[i][m_current_line[i]]; j++) {

									auto c = str[j];

									if (c == '\n') {
										continue;
									}

									auto l_info = box_type::m_tr.get_letter_info(c, box_type::m_tr.get_font_size(i));

									new_current += l_info.m_advance;

									if (fan::distance(new_current, current) < fclosest) {
										fclosest = fan::distance(new_current, current);
										iclosest = j - m_line_offset[i][m_current_line[i]];
									}
								}

								if (!new_current) {
									m_current_character[i] = m_characters_per_line[i][m_current_line[i]];
									if (str[m_line_offset[i][m_current_line[i]] + m_current_character[i] - (bool)(m_line_offset[i][m_current_line[i]] + m_current_character[i])] == '\n') {
										m_current_character[i] = 0;
									}
								}
								else {
									m_current_character[i] = iclosest + (bool)m_current_character[i];
								}

								update_cursor_position(i);
							}

							break;
						}
						case fan::key_down:
						{
							disable_select_and_reset(m_current_line[i]);

							m_text_visual_input.m_visible[i] = true;

							if (m_current_line[i] < text_box_keyboard_input::base_box::m_new_lines[i]) {

								f_t fclosest = fan::inf;
								std::size_t iclosest = 0;

								f_t current = 0;

								for (int j = 0; j < m_current_character[i]; j++) {

									auto c = *(str.begin() + m_line_offset[i][m_current_line[i]] + j);

									if (c == '\n') {
										continue;
									}

									auto l_info = box_type::m_tr.get_letter_info(c, box_type::m_tr.get_font_size(i));
									current += l_info.m_advance;
								}

								m_current_line[i]++;

								f_t new_current = 0;

								if (fan::distance(new_current, current) < fclosest) {
									fclosest = fan::distance(new_current, current);
									iclosest = 0;
								}

								for (int j = 0; j < m_characters_per_line[i][m_current_line[i]]; j++) {
									auto c = *(str.begin() + m_line_offset[i][m_current_line[i]] - 1 + j);

									if (c == '\n') {
										continue;
									}

									auto l_info = box_type::m_tr.get_letter_info(c, box_type::m_tr.get_font_size(i));

									new_current += l_info.m_advance;

									if (fan::distance(new_current, current) < fclosest) {
										fclosest = fan::distance(new_current, current);
										iclosest = j;
									}
								}

								if (!new_current) {
									m_current_character[i] = m_characters_per_line[i][m_current_line[i]];
									if (str[m_line_offset[i][m_current_line[i]] + m_current_character[i] - 1] == '\n') {
										m_current_character[i] = 0;
									}
								}
								else {
									m_current_character[i] = iclosest;
								}


								update_cursor_position(i);
							}

							break;
						}
						case fan::key_enter: {
							disable_select_and_reset(m_current_line[i]);

							/*str.insert(str.begin() + m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i], '\n');

							const auto characters_left = m_characters_per_line[i][m_current_cursor_line[i]] - m_current_character[i];

							m_line_offset[i].insert(m_line_offset[i].begin() + m_current_cursor_line[i] + 1, (m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i] + 1));

							m_characters_per_line[i][m_current_cursor_line[i]] -= characters_left;

							m_characters_per_line[i][m_current_cursor_line[i]]++;

							base_box::m_new_lines[i]++;
							m_current_cursor_line[i]++;
							m_current_character[i] = 0;

							m_characters_per_line[i].insert(m_characters_per_line[i].begin() + m_current_cursor_line[i], characters_left);

							box_type::m_tr.set_text(i, str);

							update_box_size(i, border_size);
							update_cursor_position(i, border_size);

							for (int j = m_current_cursor_line[i] + 1; j <= base_box::m_new_lines[i]; j++) {
							m_line_offset[i][j]++;
							}*/

							break;
						}
						case fan::key_tab:
						{
							disable_select_and_reset(m_current_line[i]);

							m_text_visual_input.m_visible[i] = false;

							// begin not working properly yet
							int64_t min = box_type::m_focus_begin;
							int64_t max = (box_type::m_focus_end == (uint_t)fan::uninitialized ? m_text_visual_input.m_visible.size() : box_type::m_focus_end) - box_type::m_focus_begin;

							if (box_type::m_tr.m_camera->m_window->key_press(fan::key_shift)) {

								m_text_visual_input.m_visible[(fan::modi((int64_t)(i - 1), max)) + min] = true;
								m_text_visual_input.m_timer[(fan::modi((int64_t)(i - 1), max)) + min].restart();
								current_focus[box_type::m_tr.m_camera->m_window->get_handle()] = box_type::m_focus_id[(fan::modi((int64_t)(i - 1), max)) + min];
								update_cursor_position((fan::modi((int64_t)(i - 1), max)) + min);
							}
							else {
								m_text_visual_input.m_visible[((i + 1) % max) + min] = true;
								m_text_visual_input.m_timer[((i + 1) % max) + min].restart();
								current_focus[box_type::m_tr.m_camera->m_window->get_handle()] = box_type::m_focus_id[((i + 1) % max) + min];
								update_cursor_position(((i + 1) % max) + min);
							}

							current_focus[box_type::m_tr.m_camera->m_window->get_handle()] = fan::modi(
								(int64_t)current_focus[box_type::m_tr.m_camera->m_window->get_handle()], 
								(int64_t)focus_counter[box_type::m_tr.m_camera->m_window->get_handle()]
							);

							return true;
						}
						default:
						{
g_add_key:

							if (m_starting_select_character[m_current_line[i]] != INT64_MAX) {
								replace_selected_text = true;
								goto g_delete;
							}

							if (!this->key_press(fan::key_shift) && !this->key_press(fan::key_control)) {
								disable_select_and_reset(m_current_line[i]);
							}
							for (uint_t j = 0; j < box_type::m_tr.m_camera->m_window->m_key_exceptions.size(); j++) {
								if (current_key == box_type::m_tr.m_camera->m_window->m_key_exceptions[j]) {
									return false;
								}
							}

							str.insert(str.begin() + m_line_offset[i][m_current_line[i]] + m_current_character[i], key);
							m_characters_per_line[i][m_current_line[i]]++;
							m_current_character[i]++;

							for (int j = m_current_line[i] + 1; j <= text_box_keyboard_input::base_box::m_new_lines[i]; j++) {
								m_line_offset[i][j]++;
							}

							text_box_keyboard_input::base_box::m_tr.set_text(i, str);
							box_type::update_box_size(i);

							update_cursor_position(i);
						}
					}

					return false;
				}

				constexpr void disable_select() {
					std::fill(m_starting_select_character.begin(), m_starting_select_character.end(), INT64_MAX);
				}

				constexpr void disable_select(uint_t i) {
					m_starting_select_character[i] = INT64_MAX;
				}

				constexpr void disable_select_and_reset(uint_t i) {
					m_text_visual_input.m_select.set_size(i, 0);
					m_starting_select_character[i] = INT64_MAX;
				}

				constexpr void disable_select_and_reset() {
					for (uint_t i = 0; i < m_text_visual_input.m_select.size(); i++) {
						m_text_visual_input.m_select.set_size(i, 0);
					}
					for (uint_t i = 0; i < m_starting_select_character.size(); i++) {
						m_starting_select_character[i] = INT64_MAX;
					}
				}

				void get_mouse_cursor(uint_t i) {

					if ((!key_press(fan::mouse_left) || !box_type::m_rv.inside(i))) {
						return;
					}

					const fan::vec2& box_position = box_type::m_rv.get_position(i);
					const fan::vec2& box_size = box_type::m_rv.get_size(i);

					const auto& str = box_type::m_tr.get_text(i);

					m_text_visual_input.m_visible[i] = true;

					fan::vec2 mouse_position(box_type::m_tr.m_camera->m_window->get_mouse_position());

					const fan::vec2 border_size = box_type::get_border_size(i);

					mouse_position.x = fan::clamp(mouse_position.x, box_position.x + border_size.x * 0.5f, mouse_position.x);
					mouse_position.y = fan::clamp(mouse_position.y, box_position.y + border_size.y * 0.5f, mouse_position.y);

					f_t fclosest = fan::inf;
					int64_t iclosest = 0;

					f_t current = 0;

					for (int j = 0; j < m_characters_per_line[i][m_current_line[i]] + 1; j++) {

						fan::fstring::value_type c;

						if (j != m_characters_per_line[i][m_current_line[i]]) {
							c = *(str.begin() + m_line_offset[i][m_current_line[i]] + j);
						}
						else {
							c = ' ';
						}

						if (c == '\n') {
							continue;
						}

						auto l_info = box_type::m_tr.get_letter_info(c, box_type::m_tr.get_font_size(i));
						current += l_info.m_advance;

						if (fan::distance(mouse_position.x - (box_position.x + border_size.x * 0.5), current) < fclosest) {
							fclosest = fan::distance(mouse_position.x - (box_position.x + border_size.x * 0.5), current);
							iclosest = j;
						}
					}

					if (fan::distance(mouse_position.x, box_position.x + border_size.x * 0.5) < fclosest) {
						iclosest = -1;
					}

					m_current_line[i] = (mouse_position.y - (box_position.y + border_size.y * 0.5)) / fan_2d::graphics::gui::font_properties::get_new_line(box_type::m_tr.convert_font_size(box_type::m_tr.get_font_size(i)));

					m_current_line[i] = fan::clamp(m_current_line[i], (int64_t)0, text_box_keyboard_input::base_box::m_new_lines[i]);

					m_current_character[i] = fan::clamp(iclosest + 1, (int64_t)0, m_characters_per_line[i][m_current_line[i]]);

					if (m_current_character[i] > 0) {
						if (*(str.begin() + m_line_offset[i][m_current_line[i]] + m_current_character[i] - 1) == '\n') {
							m_current_character[i] = fan::clamp(m_current_character[i] - 1, (int64_t)0, m_characters_per_line[i][m_current_line[i]]);
						}
					}

					if (m_starting_select_character[m_current_line[i]] == INT64_MAX) {
						disable_select_and_reset();

						m_starting_line[i] = m_current_line[i];
						m_starting_select_character[m_current_line[i]] = m_current_character[i];

						for (int j = 0; j < m_current_line[i] + 1; j++) {
							m_text_visual_input.m_select.set_position(j, get_cursor_position(i, m_line_offset[i][j], j == m_current_line[i] ? m_current_character[i] : m_characters_per_line[i][j])[0]);
						}
					}


					set_selected_size(i);

					update_cursor_position(i);
				}

				fan::fstring get_line(uint_t i, uint_t line) {
					const auto& str = box_type::m_tr.get_text(i);

					if (box_type::m_tr.size() < i) {
						return L"";
					}
					else if (line > text_box_keyboard_input::base_box::m_new_lines[i]) {
						return L"";
					}
					return str.substr(m_line_offset[i][line], m_characters_per_line[i][line]);
				}

				void erase(uint_t i, bool queue = false) {
					m_text_visual_input.m_cursor.erase(i, queue);
					m_text_visual_input.m_timer.erase(i);
					m_text_visual_input.m_visible.erase(i);

					m_callable.erase(m_callable.begin() + i);
					m_current_line.erase(m_current_line.begin() + i);
					m_current_character.erase(m_current_character.begin() + i);
					m_characters_per_line.erase(m_characters_per_line.begin() + i);
					m_line_offset.erase(m_line_offset.begin() + i);
				}

				/*

				struct text_visual_input {

				text_visual_input(fan::camera* camera) : m_cursor(camera), m_select(camera) {}

				fan_2d::graphics::line m_cursor;
				fan_2d::graphics::rectangle m_select;
				std::vector<fan::timer<>> m_timer;
				std::vector<bool> m_visible;
				};

				text_visual_input m_text_visual_input;

				std::vector<int64_t> m_callable;

				std::vector<int64_t> m_current_line;
				std::vector<int64_t> m_current_character;
				std::vector<int64_t> m_starting_line;
				std::vector<std::vector<int64_t>> m_characters_per_line;
				std::vector<std::vector<int64_t>> m_line_offset;

				// INT64_MAX when not dragging
				std::vector<int64_t> m_starting_select_character;
				*/

				void erase(uint_t begin, uint_t end, bool queue = false) {
					m_text_visual_input.m_cursor.erase(begin, end, queue);
					m_text_visual_input.m_select.erase(0, m_text_visual_input.m_select.size(), queue);
					m_text_visual_input.m_timer.erase(m_text_visual_input.m_timer.begin() + begin, m_text_visual_input.m_timer.begin() + end);
					m_text_visual_input.m_visible.erase(m_text_visual_input.m_visible.begin() + begin, m_text_visual_input.m_visible.begin() + end);


					m_callable.erase(m_callable.begin() + begin, m_callable.begin() + end);
					m_current_line.erase(m_current_line.begin() + begin, m_current_line.begin() + end);
					m_current_character.erase(m_current_character.begin() + begin, m_current_character.begin() + end);
					m_characters_per_line.erase(m_characters_per_line.begin() + begin, m_characters_per_line.begin() + end);
					m_line_offset.erase(m_line_offset.begin() + begin, m_line_offset.begin() + end);

					m_starting_select_character.clear();

					box_type::erase(begin, end, queue);
				}

				void set_input_callback(uint_t i) {
					m_callable.emplace_back(i);
				}

			protected:

				struct text_visual_input {

					text_visual_input(fan::camera* camera) : m_cursor(camera), m_select(camera) {}

					fan_2d::graphics::line m_cursor;
					fan_2d::graphics::rectangle m_select;
					std::vector<fan::timer<>> m_timer;
					std::vector<bool> m_visible;
				};

				text_visual_input m_text_visual_input;

				std::vector<int64_t> m_callable;

				std::vector<int64_t> m_current_line;
				std::vector<int64_t> m_current_character;
				std::vector<int64_t> m_starting_line;
				std::vector<std::vector<int64_t>> m_characters_per_line;
				std::vector<std::vector<int64_t>> m_line_offset;

				// INT64_MAX when not dragging
				std::vector<int64_t> m_starting_select_character;

			};

			struct text_box : 
				public text_box_mouse_input<fan_2d::graphics::rectangle>,
				public text_box_keyboard_input<basic_text_box<fan_2d::graphics::rectangle>> {

				using value_type = fan_2d::graphics::rectangle;

				using basic_box = basic_text_box<value_type>;
				using mouse_input = text_box_mouse_input<value_type>;
				using keyboard_input = text_box_keyboard_input<basic_box>;

				text_box(fan::camera* camera, const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& box_color, const fan::vec2& border_size, const fan::color& text_color = fan::colors::white) 
					: mouse_input(basic_box::m_rv, keyboard_input::m_focus_id), keyboard_input(camera, text, font_size, position + border_size / 2, text_color)
				{ 
					camera->m_window->add_resize_callback([&] {

						for (uint_t i = 0; i < basic_box::m_rv.size(); i++) {
							const auto offset = fan_2d::graphics::gui::get_resize_movement_offset(basic_box::m_tr.m_camera->m_window);
							basic_box::m_rv.set_position(i, basic_box::m_rv.get_position(i) + offset);
							basic_box::m_tr.set_position(i, basic_box::m_tr.get_position(i) + offset);
							update_cursor_position(i);
						}

					});

					on_click([&](uint_t i) {
						disable_select();
					});

					basic_box::m_border_size.emplace_back(border_size);

					basic_box::m_tr.set_position(0, fan::vec2(position.x + border_size.x * 0.5, position.y + border_size.y * 0.5));

					keyboard_input::push_back(basic_box::m_border_size.size() - 1);

					basic_box::m_rv.push_back(position, basic_box::get_updated_box_size(basic_box::m_border_size.size() - 1), box_color);

					keyboard_input::update_box_size(this->size() - 1);
					update_cursor_position(this->size() - 1);

					auto found = focus_counter.find(basic_box::m_tr.m_camera->m_window->get_handle());

					if (found == focus_counter.end()) {
						fan_2d::graphics::gui::focus_counter.insert(std::make_pair(basic_box::m_tr.m_camera->m_window->get_handle(), 0));
					}

					auto new_found = focus_counter.find(basic_box::m_tr.m_camera->m_window->get_handle());

					keyboard_input::m_focus_id.emplace_back(new_found->second);
					focus_counter[new_found->first]++;
				}

				void push_back(const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& box_color, const fan::vec2& border_size, const fan::color& text_color = fan::colors::white);

				void draw() {
					fan::draw_2d([&] {
						basic_box::draw();
						keyboard_input::draw();
					});
				}

				void release_queue(bool rectangle, bool color, bool text) {
					basic_box::m_rv.release_queue(rectangle, color);
					if (text) {
						basic_box::m_tr.write_data();
					}
				}

			};

			struct rounded_text_box : 
				public text_box_mouse_input<fan_2d::graphics::rounded_rectangle>,
				public text_box_keyboard_input<basic_text_box<fan_2d::graphics::rounded_rectangle>> {

				using value_type = fan_2d::graphics::rounded_rectangle;

				using basic_box = basic_text_box<value_type>;
				using mouse_input = text_box_mouse_input<value_type>;
				using keyboard_input = text_box_keyboard_input<basic_box>;

				rounded_text_box(fan::camera* camera, const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& box_color, const fan::vec2& border_size, f_t radius, const fan::color& text_color = fan::colors::white)
					:	mouse_input(basic_box::m_rv, keyboard_input::m_focus_id), text_box_keyboard_input<basic_text_box<fan_2d::graphics::rounded_rectangle>>(camera, text, font_size, position, text_color)
				{
					camera->m_window->add_resize_callback([&] {
						for (uint_t i = 0; i < basic_box::m_rv.size(); i++) {
							const auto offset = fan_2d::graphics::gui::get_resize_movement_offset(camera->m_window);
							basic_box::m_rv.set_position(i, basic_box::m_rv.get_position(i) + offset);
							basic_box::m_tr.set_position(i, basic_box::m_tr.get_position(i) + offset);
							update_cursor_position(i);
						}
					});

					mouse_input::on_click([&] (uint_t i) {}); 

					basic_box::m_border_size.emplace_back(border_size);

					keyboard_input::push_back(basic_box::m_border_size.size() - 1);

					basic_box::m_tr.set_position(0, fan::vec2(position.x + border_size.x * 0.5, position.y + 0 + border_size.y * 0.5));
					const auto size = basic_box::get_updated_box_size(0 );

					auto h = (std::abs(this->get_highest(font_size) + this->get_lowest(font_size))) / 2;
					basic_box::m_tr.set_position(0, fan::vec2(position.x + border_size.x * 0.5, position.y + h + border_size.y * 0.5));


					basic_box::m_rv.push_back(position, size, radius, box_color);

					keyboard_input::m_focus_id.emplace_back(focus_counter[camera->m_window->get_handle()]++);

					keyboard_input::update_box_size(this->size() - 1);
					update_cursor_position(this->size() - 1);

				}

				void push_back(const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& box_color, const fan::vec2& border_size, f_t radius, const fan::color& text_color = fan::colors::white);

				void set_input_callback(uint_t i) {
					keyboard_input::set_input_callback(i);
				}

				void set_position(uint_t i, const fan::vec2& position, bool queue = false) {
					basic_box::set_position(i, position, queue);
					update_cursor_position(i);
				}

				void draw() {
					fan::draw_2d([&] {
						basic_box::draw();
						keyboard_input::draw();
					});
				}

			};

			struct sized_text_box : 
				public text_box_mouse_input<fan_2d::graphics::rectangle>,
				public text_box_keyboard_input<basic_sized_text_box<fan_2d::graphics::rectangle>> {

				using value_type = fan_2d::graphics::rectangle;

				using basic_box = basic_sized_text_box<value_type>;
				using mouse_input = text_box_mouse_input<value_type>;
				using keyboard_input = text_box_keyboard_input<basic_box>;

				sized_text_box(fan::camera* camera, e_text_position text_position) : mouse_input(basic_box::m_rv, keyboard_input::m_focus_id), keyboard_input(camera), m_text_position(text_position) { 
					on_click([&](uint_t i) { disable_select_and_reset(); });
				}

				sized_text_box(fan::camera* camera, const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::vec2& size, const fan::vec2& border_size, const fan::color& box_color, e_text_position text_position, const fan::color& text_color = fan::colors::white) 
					: mouse_input(basic_box::m_rv, keyboard_input::m_focus_id), keyboard_input(camera, text, font_size, position + border_size / 2, text_color), m_text_position(text_position)
				{ 
					camera->m_window->add_resize_callback([&] {
						for (uint_t i = 0; i < basic_box::m_rv.size(); i++) {
							const auto offset = fan_2d::graphics::gui::get_resize_movement_offset(keyboard_input::m_tr.m_camera->m_window);
							basic_box::m_rv.set_position(i, basic_box::m_rv.get_position(i) + offset);
							basic_box::m_tr.set_position(i, basic_box::m_tr.get_position(i) + offset);
							update_cursor_position(i);
						}
					});

					on_click([&](uint_t i) { disable_select_and_reset(); });

					m_size.emplace_back(size);

					basic_box::m_border_size.emplace_back(border_size);

					auto h = (std::abs(this->get_highest(get_font_size(0)) + this->get_lowest(get_font_size(0)))) / 2;

					basic_box::m_tr.set_position(0, fan::vec2(position.x + size.x * 0.5 - keyboard_input::m_tr.get_text_size(text, font_size).x * 0.5, position.y + size.y * 0.5 - h) + border_size * 0.5);

					keyboard_input::push_back(basic_box::m_border_size.size() - 1);

					basic_box::m_rv.push_back(position, size + border_size, box_color);

					update_cursor_position(this->size() - 1);

					auto found = focus_counter.find(basic_box::m_tr.m_camera->m_window->get_handle());

					if (found == focus_counter.end()) {
						fan_2d::graphics::gui::focus_counter.insert(std::make_pair(basic_box::m_tr.m_camera->m_window->get_handle(), 0));
					}

					auto new_found = focus_counter.find(basic_box::m_tr.m_camera->m_window->get_handle());

					keyboard_input::m_focus_id.emplace_back(new_found->second);
					focus_counter[new_found->first]++;
				}

				void push_back(const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::vec2& size, const fan::vec2& border_size, const fan::color& box_color, const fan::color& text_color = fan::colors::white) {

					m_size.emplace_back(size);

					basic_box::m_border_size.emplace_back(border_size);

					auto h = (std::abs(this->get_highest(font_size) + this->get_lowest(font_size))) / 2;

					switch (m_text_position) {
						case e_text_position::middle:
						{
							basic_box::m_tr.push_back(text, fan::vec2(position.x + size.x * 0.5 - keyboard_input::m_tr.get_text_size(text, font_size).x * 0.5, position.y + size.y * 0.5 - h) + border_size * 0.5, text_color, font_size);
							break;
						}
						case e_text_position::left:
						{
							basic_box::m_tr.push_back(text, fan::vec2(position.x, position.y + size.y * 0.5 - h) + border_size * 0.5, text_color, font_size);
							break;
						}
					}


					keyboard_input::push_back(basic_box::m_border_size.size() - 1);

					basic_box::m_rv.push_back(position, size + border_size, box_color);

					update_cursor_position(this->size() - 1);

					auto found = focus_counter.find(basic_box::m_tr.m_camera->m_window->get_handle());

					if (found != focus_counter.end()) {
						keyboard_input::m_focus_id.emplace_back(found->second);
						focus_counter[found->first]++;
					}
					else {
						keyboard_input::m_focus_id.emplace_back(0);
						focus_counter.insert(std::make_pair(keyboard_input::m_tr.m_camera->m_window->get_handle(), 0));
					}

				}

				void draw() {
					fan::draw_2d([&] {
						base_box::draw();
						keyboard_input::draw();
					});
				}

			private:

				e_text_position m_text_position;

			};

			class basic_selectable_box {
			public:

				basic_selectable_box() : m_selected(fan::uninitialized) { }

				int64_t get_selected() {
					return m_selected;
				}

				void set_selected(int64_t i) {
					m_selected = i;
				}

				/*void color_on_click(uint_t i, const fan::color& color) {
				mouse_input_t::on_click(i, [&] {
				box_t::set_box_color(i, color);
				this->set_selected(i);
				});
				}*/

			private:

				//	mouse_input_t& m_mouse_input;
				//box_t& m_box;

				int64_t m_selected;

			};

			struct selectable_text_box : public text_box, public basic_selectable_box {
				using text_box::text_box;
			};

			struct selectable_rounded_text_box : public rounded_text_box, public basic_selectable_box {
				using rounded_text_box::rounded_text_box;
			};

			struct selectable_sized_text_box : public sized_text_box, public basic_selectable_box {
				using sized_text_box::sized_text_box;
			};

			//class slider : public text_box {
			//public:

			//	slider(fan::camera* camera, f_t min, f_t max, f_t font_size, const fan::vec2& position, const fan::color& slider_color, const fan::color& box_color, const fan::vec2& border_size, const fan::color& text_color = fan::colors::white)
			//		: text_box(camera, fan::to_wstring(min), font_size, position, box_color, border_size, text_color) 
			//	{
			//		const auto size(text_box::m_rv.get_size(text_box::m_rv.size() - 1));

			//		m_min.push_back(min);
			//		m_max.push_back(max);
			//		m_value.push_back(0);

			//		text_box::m_rv.push_back(position + 5, fan::vec2(size.x / 20, size.y - 5 * 2), slider_color);

			//		m_moving.resize(m_moving.size() + 1);

			//	}

			//	f_t get_value(uint_t i) const {
			//		return m_value[i];
			//	}

			//	void set_value(uint_t i, f_t value) {
			//		m_value[i] = value;
			//	}

			//	void move()	{
			//		const bool left_press = m_rv.m_camera->m_window->key_press(fan::mouse_left);

			//		for (uint_t i = 1; i < m_rv.size(); i += 2) {
			//			if (m_rv.inside(i >> 1) && left_press) {
			//				m_moving[i >> 1] = true;
			//			}
			//			else if (!left_press) {
			//				m_moving[i >> 1] = false;
			//			}
			//			if (m_moving[i >> 1]) {
			//				const auto mouse_position(m_tr.m_camera->m_window->get_mouse_position());
			//				const auto s_size(m_rv.get_size(i));
			//				const auto b_size(m_rv.get_size(i >> 1));
			//				const auto position(m_rv.get_position(i >> 1));

			//				auto slider_position = std::clamp((mouse_position.x - s_size.x * 0.5), f_t(position.x + 5), f_t(position.x + b_size.x - s_size.x - 5));

			//				auto min = position.x + 5;
			//				auto max = position.x + b_size.x - s_size.x - 5;
			//				auto n = slider_position;

			//				m_rv.set_position(i, fan::vec2(slider_position, position.y + 5));

			//				this->set_value(i >> 1, ((n - min) / (max - min)) * (m_max[i >> i] - m_min[i >> 1]) + m_min[i >> 1]);

			//				const auto& str = fan::to_wstring(this->get_value(i >> 1));

			//				auto s = m_tr.get_text_size(str, get_font_size(i >> 1));
			//				fan::print(s);
			//				m_tr.set_text(i >> 1, str);

			//			}
			//		}
			//	}

			//private:

			//	std::vector<bool> m_moving;
			//	std::vector<f_t> m_min;
			//	std::vector<f_t> m_value;
			//	std::vector<f_t> m_max;
			//	std::vector<f_t> m_previous;

			//};

		}

	}
}