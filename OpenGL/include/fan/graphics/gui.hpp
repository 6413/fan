#pragma once

#include <fan/graphics/graphics.hpp>

#include <fan/graphics/themes.hpp>

#include <fan/physics/collision/rectangle.hpp>

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
			//	sprite(fan::camera* camera, const std::string& path, const fan::vec2& position, const fan::vec2& size = 0, f32_t transparency = 1);
			//	sprite(fan::camera* camera, unsigned char* pixels, const fan::vec2& position, const fan::vec2i& size = 0, f32_t transparency = 1);

			//};


			namespace font_properties {

				inline f32_t space_width(15);

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
				public fan::glsl_location_handler<1, fan::opengl_buffer_type::buffer_object, true> {
			public:
				using text_color_t = fan::basic_shape_color_vector_vector<3, fan::opengl_buffer_type::buffer_object, true>;
				using outline_color_t = fan::basic_shape_color_vector_vector<3, fan::opengl_buffer_type::shader_storage_buffer_object, true>;

				using font_sizes_ssbo_t = fan::glsl_location_handler<2, fan::opengl_buffer_type::buffer_object, true>;

				using vertex_vbo_t = fan::glsl_location_handler<0, fan::opengl_buffer_type::buffer_object, true>;
				using texture_vbo_t = fan::glsl_location_handler<1, fan::opengl_buffer_type::buffer_object, true>;

				text_renderer(fan::camera* camera);
				text_renderer(fan::camera* camera, const fan::fstring& text, const fan::vec2& position, const fan::color& text_color, f32_t font_size, const fan::color& outline_color = fan::color(-1, -1, -1, 0), bool queue = false);

				text_renderer(const text_renderer& tr);
				text_renderer(text_renderer&& tr);

				text_renderer& operator=(const text_renderer& tr);
				text_renderer& operator=(text_renderer&& tr);

				fan::vec2 get_position(uint_t i) const;

				void set_position(uint_t i, const fan::vec2& position, bool queue = false);

				f32_t get_font_size(uint_t i) const;
				void set_font_size(uint_t i, f32_t font_size, bool queue = false);
				void set_text(uint_t i, const fan::fstring& text, bool queue = false);
				void set_text_color(uint_t i, const fan::color& color, bool queue = false);
				void set_text_color(uint_t i, uint_t j, const fan::color& color, bool queue = false);
				void set_outline_color(uint_t i, const fan::color& color, bool queue = false);

				static fan::io::file::font_t get_letter_info(fan::fstring::value_type c, f32_t font_size);

				fan::vec2 get_text_size(uint_t i) const;
				static fan::vec2 get_text_size(const fan::fstring& text, f32_t font_size);

				f32_t get_average_text_height(const fan::fstring& text, f32_t font_size) const;

				f32_t get_longest_text() const;
				f32_t get_highest_text() const;

				f32_t get_lowest(f32_t font_size) const;
				f32_t get_highest(f32_t font_size) const;

				f32_t get_highest_size(f32_t font_size) const;
				f32_t get_lowest_size(f32_t font_size) const;

				// i = string[i], j = string[i][j] (fan::fstring::value_type)
				fan::color get_color(uint_t i, uint_t j = 0) const;

				fan::fstring get_text(uint_t i) const;

				static f32_t convert_font_size(f32_t font_size);

				void insert(uint_t i, const fan::fstring& text, const fan::vec2& position, const fan::color& text_color, f32_t font_size, const fan::color& outline_color = fan::uninitialized, bool queue = false);
				void push_back(const fan::fstring& text, const fan::vec2& position, const fan::color& text_color, f32_t font_size, const fan::color& outline_color = fan::uninitialized, bool queue = false);

				void draw() const;

				void erase(uint_t i, bool queue = false);
				void erase(uint_t begin, uint_t end, bool queue = false);

				uint_t size() const;

				void write_data();

				void release_queue(bool vertices, bool texture_coordinates, bool font_sizes);

				int64_t get_new_lines(uint32_t i);

				static int64_t get_new_lines(const fan::fstring& str);

				fan::vec2 get_character_position(uint32_t i, uint32_t j);

				f32_t get_line_height(f32_t font_size) const;

				static f32_t get_original_font_size();

				static inline fan::io::file::font_info font_info;

			private:

				fan::camera* m_camera;

				void initialize_buffers();

				void load_characters(uint_t i, fan::vec2 position, const fan::fstring& text, bool edit, bool insert);

				void edit_letter_data(uint_t i, uint_t j, const fan::fstring::value_type letter, const fan::vec2& position, int& advance, f32_t converted_font_size);
				void insert_letter_data(uint_t i, const fan::fstring::value_type letter, const fan::vec2& position, int& advance, f32_t converted_font_size);
				void write_letter_data(uint_t i, const fan::fstring::value_type letter, const fan::vec2& position, int& advance, f32_t converted_font_size);

				void write_vertices();
				void write_texture_coordinates();
				void write_font_sizes();

				fan::shader m_shader;

				fan::vec2ui m_original_image_size;

				std::vector<fan::fstring> m_text;

				std::vector<fan::_vec2<f32_t>> m_position;

				std::vector<std::vector<f32_t>> m_font_size;
				std::vector<std::vector<fan::_vec2<f32_t>>> m_vertices;
				std::vector<std::vector<fan::_vec2<f32_t>>> m_texture_coordinates;

				static constexpr auto vertex_location_name = "vertex";
				static constexpr auto text_color_location_name = "text_colors";
				static constexpr auto texture_coordinates_location_name = "texture_coordinate";
				static constexpr auto font_sizes_location_name = "font_sizes";

			};

		//	namespace text_box_properties {

		//		inline int blink_speed(500); // ms
		//		inline fan::color cursor_color(fan::colors::white);
		//		inline fan::color select_color(fan::colors::blue - fan::color(0, 0, 0, 0.5));

		//	}

		//	enum class text_position_e {
		//		left,
		//		middle
		//	};

		//	struct button_properties {

		//		button_properties() {}

		//		button_properties(
		//			const fan::fstring& text, 
		//			const fan::vec2& position
		//		) : text(text), position(position) {}

		//		fan::fstring text;

		//		fan::vec2 position;
		//		fan::vec2 border_size;

		//		f32_t font_size = fan_2d::graphics::gui::defaults::font_size;

		//		text_position_e text_position = text_position_e::middle;

		//		// queues push_back in opengl buffer. release_queue() writes to buffer
		//		bool queue = false;

		//	};

		//	struct rectangle_button_properties : public button_properties {

		//	};

		//	struct sprite_button_properties : public button_properties {
		//		uint32_t texture_id = fan::uninitialized;
		//	};

		//	namespace base {

		//		struct button_metrics {

		//			std::vector<int64_t> m_new_lines;

		//		};

		//		// requires to have functions: 
		//		// get_camera() which returns camera, 
		//		// size() returns amount of objects, 
		//		// inside() if mouse inside
		//		struct mouse {

		//		protected:

		//			template <typename T>
		//			mouse(T& object) {

		//				std::memset(m_hover_button_id, -1, sizeof(m_hover_button_id));
		//				std::memset(m_held_button_id, -1, sizeof(m_held_button_id));


		//				// ------------------------------ key press

		//				m_key_press_it[0] = object.get_camera()->m_window->add_key_callback(fan::mouse_left, [&] {
		//					for (uint_t i = 0; i < object.size(); i++) {
		//						if (object.inside(i)) {
		//							m_held_button_id[0] = i;
		//							if (m_on_click[0]) {
		//								m_on_click[0](i);
		//							}
		//						}
		//					}
		//				});

		//				m_key_press_it[1] = object.get_camera()->m_window->add_key_callback(fan::mouse_left, [&] {
		//					for (uint_t i = 0; i < object.size(); i++) {
		//						if (object.inside(i)) {
		//							m_held_button_id[1] = i;
		//							if (m_on_click[1]) {
		//								m_on_click[1](i);
		//							}
		//						}
		//					}
		//				});

		//				// ------------------------------ key release

		//				m_key_release_it[0] = object.get_camera()->m_window->add_key_callback(fan::mouse_left, [&] {

		//					if (m_held_button_id[0] != (uint32_t)fan::uninitialized && object.inside(m_held_button_id[0])) {
		//						if (m_on_release[0]) {
		//							m_on_release[0](m_held_button_id[0]);
		//						}

		//						m_held_button_id[0] = fan::uninitialized;
		//					}
		//					else if (m_held_button_id[0] != (uint32_t)fan::uninitialized && !object.inside(m_held_button_id[0])) {
		//							if (m_on_outside_release[0]) {
		//								m_on_outside_release[0](m_held_button_id[0]);
		//							}

		//							m_held_button_id[0] = fan::uninitialized;
		//					}

		//				}, true);

		//				m_key_release_it[1] = object.get_camera()->m_window->add_key_callback(fan::mouse_left, [&] {

		//					if (m_held_button_id[1] != (uint32_t)fan::uninitialized && object.inside(m_held_button_id[1])) {
		//						if (m_on_release[1]) {
		//							m_on_release[1](m_held_button_id[1]);
		//						}

		//						m_held_button_id[1] = fan::uninitialized;
		//					}
		//					else if (m_held_button_id[1] != (uint32_t)fan::uninitialized && !object.inside(m_held_button_id[1])) {
		//						if (m_on_outside_release[1]) {
		//							m_on_outside_release[1](m_held_button_id[1]);
		//						}

		//						m_held_button_id[1] = fan::uninitialized;
		//					}

		//				}, true);

		//				// ------------------------------ hover

		//				object.get_camera()->m_window->add_mouse_move_callback([&](fan::window*) {
		//					for (int k = 0; k < 2; k++) {
		//						for (uint_t i = 0; i < object.size(); i++) {
		//							if (object.inside(i) && i != m_hover_button_id[k] && m_hover_button_id[k] == (uint32_t)fan::uninitialized) {
		//								if (m_on_hover[k]) {
		//									m_on_hover[k](i);
		//								}

		//								m_hover_button_id[k] = i;
		//							}
		//						}
		//					}
		//				});

		//				// ------------------------------ exit

		//				object.get_camera()->m_window->add_mouse_move_callback([&](fan::window*) {
		//					for (int k = 0; k < 2; k++) {
		//						if (m_hover_button_id[k] != (uint32_t)fan::uninitialized && !object.inside(m_hover_button_id[k]) && object.inside(m_hover_button_id[k], object.get_camera()->m_window->get_previous_mouse_position())) {
		//							if (m_on_exit[k]) {
		//								m_on_exit[k](m_hover_button_id[k]);
		//							}

		//							m_hover_button_id[k] = (uint32_t)fan::uninitialized;
		//						}
		//					}
		//				});

		//			}

		//			// keys must be same for click and release

		//			template <bool user_side>
		//			void on_click(std::function<void(uint32_t i)> function, uint16_t key = fan::mouse_left) {
		//				m_key_press_key[user_side] = key;
		//				m_key_press_it[user_side]->key = key;
		//				m_on_click[user_side] = function;

		//				m_key_release_it[user_side]->key = key;
		//				m_on_release[user_side] = function;
		//			}

		//			template <bool user_side>
		//			void on_release(std::function<void(uint32_t i)> function, uint16_t key = fan::mouse_left) {
		//				m_key_release_key[user_side] = key;
		//				m_key_release_it[user_side]->key = key;
		//				m_on_release[user_side] = function;

		//				m_key_press_key[user_side] = key;
		//				m_key_press_it[user_side]->key = key;
		//			}

		//			template <bool user_side>
		//			void on_outside_release(std::function<void(uint32_t i)> function, uint16_t key = fan::mouse_left) {
		//				m_key_release_key[user_side] = key;
		//				m_key_release_it[user_side]->key = key;
		//				m_on_outside_release[user_side] = function;

		//				m_key_press_key[user_side] = key;
		//				m_key_press_it[user_side]->key = key;
		//			}

		//			template <bool user_side>
		//			void on_hover(std::function<void(uint32_t i)> function) {
		//				m_on_hover[user_side] = function;
		//			}

		//			template <bool user_side>
		//			void on_exit(std::function<void(uint32_t i)> function) {
		//				m_on_exit[user_side] = function;
		//			}


		//		public:

		//			void on_click(std::function<void(uint32_t i)> function, uint16_t key = fan::mouse_left) { 
		//				base::mouse::on_click<1>(function, key); 
		//			} 

		//			void on_release(std::function<void(uint32_t i)> function, uint16_t key = fan::mouse_left) { 
		//				base::mouse::on_release<1>(function, key); 
		//			}

		//			void on_outside_release(std::function<void(uint32_t i)> function, uint16_t key = fan::mouse_left) { 
		//				base::mouse::on_outside_release<1>(function, key); 
		//			}

		//			void on_hover(std::function<void(uint32_t i)> function) { 
		//				base::mouse::on_hover<1>(function); 
		//			} 

		//			void on_exit(std::function<void(uint32_t i)> function) { 
		//				base::mouse::on_exit<1>(function); 
		//			}

		//		protected:

		//			std::function<void(uint32_t)> m_on_click[2];
		//			std::function<void(uint32_t)> m_on_release[2];
		//			std::function<void(uint32_t)> m_on_outside_release[2];
		//			std::function<void(uint32_t)> m_on_hover[2];
		//			std::function<void(uint32_t)> m_on_exit[2];

		//			uint32_t m_held_button_id[2];
		//			uint32_t m_hover_button_id[2];

		//			uint32_t m_key_press_key[2];
		//			uint32_t m_key_release_key[2];

		//			std::deque<fan::window::key_callback_t>::iterator m_key_press_it[2];
		//			std::deque<fan::window::key_callback_t>::iterator m_key_release_it[2];

		//		};

		//		#define define_get_button_size \
		//		fan::vec2 get_button_size(uint32_t i) const \
		//		{ \
		//			const f32_t font_size = fan_2d::graphics::gui::text_renderer::get_font_size(i); \
		//																						  \
		//			f32_t h = (std::abs(fan_2d::graphics::gui::text_renderer::get_highest(font_size) + fan_2d::graphics::gui::text_renderer::get_lowest(font_size))); \
		//																																							\
		//			if (i < m_new_lines.size() && m_new_lines[i]) { \
		//				h += fan_2d::graphics::gui::font_properties::new_line * fan_2d::graphics::gui::text_renderer::convert_font_size(font_size) * m_new_lines[i]; \
		//			} \
		//			\
		//			return (fan_2d::graphics::gui::text_renderer::get_text(i).empty() ? fan::vec2(0, h) : fan::vec2(fan_2d::graphics::gui::text_renderer::get_text_size(fan_2d::graphics::gui::text_renderer::get_text(i), font_size).x, h)) + m_properties[i].border_size; \
		//		}

		//		#define define_get_property_size \
		//		fan::vec2 get_size(properties_t properties) \
		//		{ \
		//			f32_t h = (std::abs(fan_2d::graphics::gui::text_renderer::get_highest(properties.font_size) + fan_2d::graphics::gui::text_renderer::get_lowest(properties.font_size))); \
		//																																													\
		//			int64_t new_lines = fan_2d::graphics::gui::text_renderer::get_new_lines(properties.text); \
		//				\
		//			if (new_lines) { \
		//				h += fan_2d::graphics::gui::font_properties::new_line * fan_2d::graphics::gui::text_renderer::convert_font_size(properties.font_size) * new_lines; \
		//			} \
		//				\
		//			return (properties.text.empty() ? fan::vec2(0, h) : fan::vec2(fan_2d::graphics::gui::text_renderer::get_text_size(properties.text, properties.font_size).x, h)) + properties.border_size; \
		//		}

		//	}

		//	class rectangle_text_button : 
		//		protected fan::class_duplicator<fan_2d::graphics::rectangle, 0>, 
		//		protected fan::class_duplicator<fan_2d::graphics::rectangle, 1>,
		//		public base::mouse,
		//		public base::button_metrics,
		//		protected fan_2d::graphics::gui::text_renderer {

		//	public:

		//		using properties_t = rectangle_button_properties;

		//		rectangle_text_button(fan::camera* camera, fan_2d::graphics::gui::theme theme = fan_2d::graphics::gui::themes::deep_blue());

		//		void push_back(const rectangle_button_properties& properties);

		//		void draw(uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized) const;

		//		using fan_2d::graphics::rectangle::get_size;

		//		define_get_property_size;

		//		bool inside(uint_t i, const fan::vec2& position = fan::math::inf) const;

		//		fan::camera* get_camera();

		//		uint_t size() const;

		//		using class_duplicator<fan_2d::graphics::rectangle, 0>::get_color;
		//		using class_duplicator<fan_2d::graphics::rectangle, 0>::set_color;

		//	protected:

		//		define_get_button_size;

		//		std::vector<rectangle_button_properties> m_properties;

		//		fan_2d::graphics::gui::theme theme;

		//	};

		//	class sprite_text_button :
		//		protected fan_2d::graphics::sprite,
		//		public base::mouse,
		//		public base::button_metrics,
		//		protected fan_2d::graphics::gui::text_renderer {

		//	public:

		//		using properties_t = sprite_button_properties;

		//		sprite_text_button(fan::camera* camera, const std::string& path);

		//		void push_back(const sprite_button_properties& properties);

		//		void draw(uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized) const;

		//		define_get_property_size

		//		fan::camera* get_camera();

		//		using fan_2d::graphics::sprite::size;
		//		using fan_2d::graphics::sprite::inside;

		//	protected:

		//		std::vector<sprite_button_properties> m_properties;

		//		define_get_button_size

		//	};

		//	template <typename T>
		//	struct slider_property {

		//		T min;
		//		T max;

		//		T current;
		//		
		//		fan::vec2 position;
		//		fan::vec2 box_size;
		//		f32_t button_radius;
		//		fan::color box_color;
		//		fan::color button_color;

		//		fan::color text_color = fan_2d::graphics::gui::defaults::text_color;

		//		f32_t font_size;

		//		bool queue = false;
		//	};

		//	template <typename value_type>
		//	class circle_slider : protected fan_2d::graphics::circle, protected fan_2d::graphics::rounded_rectangle {
		//	public:

		//		circle_slider(fan::camera* camera)
		//			: fan_2d::graphics::circle(camera), fan_2d::graphics::rounded_rectangle(camera), m_click_begin(fan::uninitialized), m_moving_id(fan::uninitialized)
		//		{
		//			camera->m_window->add_key_callback(fan::mouse_left, [&] {

		//				// last ones are on the bottom
		//				for (uint32_t i = fan_2d::graphics::circle::size(); i-- ; ) {
		//					if (fan_2d::graphics::circle::inside(i)) {

		//						m_click_begin = fan_2d::graphics::rounded_rectangle::m_camera->m_window->get_mouse_position();
		//						m_moving_id = i;

		//						return;
		//					}
		//				}

		//				const fan::vec2 mouse_position = fan_2d::graphics::rounded_rectangle::m_camera->m_window->get_mouse_position();

		//				for (uint32_t i = fan_2d::graphics::circle::size(); i-- ; ) {

		//					const fan::vec2 box_position = fan_2d::graphics::rounded_rectangle::get_position(i);
		//					const fan::vec2 box_size = fan_2d::graphics::rounded_rectangle::get_size(i);

		//					const f32_t circle_diameter = fan_2d::graphics::circle::get_radius(i) * 2;

		//					const bool horizontal = box_size.x > box_size.y;

		//					if (fan_2d::collision::rectangle::point_inside_no_rotation(mouse_position, box_position - fan::vec2(horizontal ? 0 : circle_diameter / 2 - 2, horizontal ? circle_diameter / 2 - 2 : 0), fan::vec2(horizontal ? box_size.x : circle_diameter, horizontal ? circle_diameter : box_size.y))) {

		//						m_click_begin = fan_2d::graphics::rounded_rectangle::m_camera->m_window->get_mouse_position();
		//						m_moving_id = i;

		//						fan::vec2 circle_position = fan_2d::graphics::circle::get_position(i);

		//						if (horizontal) {
		//							fan_2d::graphics::circle::set_position(m_moving_id, fan::vec2(mouse_position.x, circle_position.y));
		//						}
		//						else {
		//							fan_2d::graphics::circle::set_position(m_moving_id, fan::vec2(circle_position.x, mouse_position.y));
		//						}

		//						circle_position = fan_2d::graphics::circle::get_position(i);

		//						f32_t min = get_min_value(m_moving_id);
		//						f32_t max = get_max_value(m_moving_id);

		//						f32_t length = box_size[!horizontal];

		//						set_current_value(m_moving_id, min + (((circle_position[!horizontal] - box_position[!horizontal]) / length) * (max - min)));

		//						for (int i = 0; i < 2; i++) {
		//							if (m_on_click[i]) {
		//								m_on_click[i](m_moving_id);
		//							}
		//							if (m_on_drag[i]) {
		//								m_on_drag[i](m_moving_id);
		//							}
		//						}

		//						return;
		//					}
		//				}

		//			});

		//			camera->m_window->add_key_callback(fan::mouse_left, [&] {

		//				m_click_begin = fan::uninitialized;
		//				m_moving_id = fan::uninitialized;

		//			}, true);

		//			camera->m_window->add_mouse_move_callback([&](const fan::vec2& position) {

		//				if (m_click_begin == fan::uninitialized) {
		//					return;
		//				}

		//				const fan::vec2 box_position = fan_2d::graphics::rounded_rectangle::get_position(m_moving_id);
		//				const fan::vec2 box_size = fan_2d::graphics::rounded_rectangle::get_size(m_moving_id);

		//				fan::vec2 circle_position = fan_2d::graphics::circle::get_position(m_moving_id);

		//				const bool horizontal = box_size.x > box_size.y;

		//				if (horizontal) {
		//				}
		//				else {
		//					circle_position.y = m_click_begin.y + (position.y - m_click_begin.y);
		//				}

		//				f32_t length = box_size[!horizontal];

		//				f32_t min = get_min_value(m_moving_id);
		//				f32_t max = get_max_value(m_moving_id);

		//				circle_position[!horizontal] = m_click_begin[!horizontal] + (position[!horizontal] - m_click_begin[!horizontal]);

		//				circle_position = circle_position.clamp(
		//					fan::vec2(box_position.x, box_position.x + box_size.x),
		//					fan::vec2(box_position.y, box_position.y + box_size.y)
		//				);

		//				set_current_value(m_moving_id, min + (((circle_position[!horizontal] - box_position[!horizontal]) / length) * (max - min)));

		//				fan_2d::graphics::circle::set_position(m_moving_id, circle_position);

		//				for (int i = 0; i < 2; i++) {
		//					if (m_on_drag[i]) {
		//						m_on_drag[i](m_moving_id);
		//					}
		//				}
		//			});
		//		}

		//		void push_back(const slider_property<value_type>& property)
		//		{
		//			m_properties.emplace_back(property);
		//			m_properties[m_properties.size() - 1].current = fan::clamp(m_properties[m_properties.size() - 1].current, property.min, property.max);

		//			fan_2d::graphics::rounded_rectangle::push_back(property.position, property.box_size, 30, property.box_color, property.queue);

		//			if (property.box_size.x > property.box_size.y) {
		//				
		//				f32_t min = property.position.x;
		//				f32_t max = property.position.x + property.box_size.x;

		//				f32_t new_x = (f32_t(property.current - property.min) / (property.max - property.min)) * (max - min);

		//				fan_2d::graphics::circle::push_back(property.position + fan::vec2(new_x, property.box_size.y / 2), property.button_radius, property.button_color, property.queue);
		//			}
		//			else {

		//				f32_t min = property.position.y;
		//				f32_t max = property.position.y + property.box_size.y;

		//				f32_t new_y = (f32_t(property.current - property.min) / (property.max - property.min)) * (max - min);

		//				fan_2d::graphics::circle::push_back(property.position + fan::vec2(property.box_size.x / 2, new_y), property.button_radius, property.button_color, property.queue);
		//			}

		//		}

		//		void draw() const
		//		{
		//			fan_2d::graphics::draw([&] {
		//				fan_2d::graphics::rounded_rectangle::draw();
		//				fan_2d::graphics::circle::draw();
		//			});
		//		}

		//		auto get_min_value(uint32_t i) const {
		//			return m_properties[i].min;
		//		}
		//		void set_min_value(uint32_t i, value_type value) {
		//			m_properties[i].min = value;
		//		}
		//		auto get_max_value(uint32_t i) const {
		//			return m_properties[i].max;
		//		}
		//		void set_max_value(uint32_t i, value_type value) {
		//			m_properties[i].max = value;
		//		}
		//		auto get_current_value(uint32_t i) const {
		//			return m_properties[i].current;
		//		}
		//		void set_current_value(uint32_t i, value_type value) {
		//			m_properties[i].current = value;
		//		}

		//		void on_drag(const std::function<void(uint32_t)>& function) {
		//			on_drag(true, function);
		//			m_on_drag[1] = function;
		//		}

		//	protected:

		//		void on_drag(bool user, const std::function<void(uint32_t)>& function) {
		//			m_on_drag[user] = function;
		//		}

		//		void on_click(bool user, const std::function<void(uint32_t)>& function) {
		//			m_on_click[user] = function;
		//		}

		//		std::deque<slider_property<value_type>> m_properties;

		//		fan::vec2 m_click_begin;

		//		uint32_t m_moving_id;

		//		std::function<void(uint32_t i)> m_on_drag[2]; // 0 lib, 1 user
		//		std::function<void(uint32_t)> m_on_click[2]; // 0 lib, 1 user

		//	};

		//	template <typename value_type>
		//	class circle_text_slider : public circle_slider<value_type>, protected fan_2d::graphics::gui::text_renderer {
		//	public:

		//		static constexpr f32_t text_gap_multiplier = 1.5;

		//		circle_text_slider(fan::camera* camera) : circle_slider<value_type>(camera), fan_2d::graphics::gui::text_renderer(camera) {
		//			circle_text_slider::circle_slider::on_drag(false, [&](uint32_t i) {
		//				fan_2d::graphics::gui::text_renderer::set_text(i * 3 + 2, fan::to_wstring(this->get_current_value(i)));

		//				const fan::vec2 middle_text_size = fan_2d::graphics::gui::text_renderer::get_text_size(fan::to_wstring(this->get_current_value(i)), circle_text_slider::circle_slider::m_properties[i].font_size);

		//				const fan::vec2 position = circle_text_slider::circle_slider::rounded_rectangle::get_position(i);
		//				const fan::vec2 box_size = circle_text_slider::circle_slider::rounded_rectangle::get_size(i);

		//				const f32_t button_radius = circle_text_slider::circle_slider::circle::get_radius(i);

		//				fan::vec2 middle;

		//				if (box_size.x > box_size.y) {
		//					middle = position + fan::vec2(box_size.x / 2 - middle_text_size.x / 2, -middle_text_size.y - button_radius * text_gap_multiplier);
		//				}
		//				else {
		//					middle = position + fan::vec2(box_size.x / 2 + button_radius * text_gap_multiplier, box_size.y / 2 - middle_text_size.y / 2);
		//				}

		//				fan_2d::graphics::gui::text_renderer::set_position(i * 3 + 2, middle);
		//			});

		//			circle_text_slider::circle_slider::on_click(false, [&] (uint32_t i) {
		//				fan_2d::graphics::gui::text_renderer::set_text(i * 3 + 2, fan::to_wstring(this->get_current_value(i)));
		//			});
		//		}

		//		void push_back(const slider_property<value_type>& property) {
		//			circle_text_slider::circle_slider::push_back(property);

		//			const fan::vec2 left_text_size = fan_2d::graphics::gui::text_renderer::get_text_size(fan::to_wstring(property.min), property.font_size);

		//			fan::vec2 left_or_up;

		//			if (property.box_size.x > property.box_size.y) {
		//				left_or_up = property.position - fan::vec2(left_text_size.x + property.button_radius * text_gap_multiplier, left_text_size.y / 2);
		//			}
		//			else {
		//				left_or_up = property.position - fan::vec2(left_text_size.x / 2, left_text_size.y + property.button_radius * text_gap_multiplier);
		//			}

		//			fan_2d::graphics::gui::text_renderer::push_back(
		//				fan::to_wstring(property.min), 
		//				left_or_up, 
		//				fan_2d::graphics::gui::defaults::text_color, 
		//				property.font_size
		//			);

		//			const fan::vec2 right_text_size = fan_2d::graphics::gui::text_renderer::get_text_size(fan::to_wstring(property.min), property.font_size);

		//			fan::vec2 right_or_down;

		//			if (property.box_size.x > property.box_size.y) {
		//				right_or_down = property.position + fan::vec2(property.box_size.x + property.button_radius * text_gap_multiplier, -right_text_size.y / 2);
		//			}
		//			else {
		//				right_or_down = property.position + fan::vec2(-right_text_size.x / 2, property.box_size.y + property.button_radius * text_gap_multiplier);
		//			}

		//			fan_2d::graphics::gui::text_renderer::push_back(
		//				fan::to_wstring(property.max),
		//				right_or_down, 
		//				fan_2d::graphics::gui::defaults::text_color, 
		//				property.font_size
		//			);

		//			const fan::vec2 middle_text_size = fan_2d::graphics::gui::text_renderer::get_text_size(fan::to_wstring(property.current), property.font_size);

		//			fan::vec2 middle;

		//			if (property.box_size.x > property.box_size.y) {
		//				middle = property.position + fan::vec2(property.box_size.x / 2 - middle_text_size.x / 2, -middle_text_size.y - property.button_radius * text_gap_multiplier);
		//			}
		//			else {
		//				middle = property.position + fan::vec2(property.box_size.x / 2 + property.button_radius * text_gap_multiplier, property.box_size.y / 2 - middle_text_size.y / 2);
		//			}

		//			fan_2d::graphics::gui::text_renderer::push_back(
		//				fan::to_wstring(property.current), 
		//				middle, 
		//				fan_2d::graphics::gui::defaults::text_color, 
		//				property.font_size
		//			);
		//		}

		//		void draw() const {
		//			fan_2d::graphics::draw([&] {
		//				fan_2d::graphics::gui::circle_slider<value_type>::draw();
		//				fan_2d::graphics::gui::text_renderer::draw();
		//			});
		//		}

		//	private:
		//	
		//	};

		//	struct checkbox_property {
		//		fan::vec2 position;

		//		f32_t font_size;

		//		fan::fstring text;

		//		uint8_t line_thickness = 2;

		//		f32_t box_size_multiplier = 1.5;

		//		bool checked = false;

		//		bool queue = false;
		//	};

		//	class checkbox : 
		//		protected fan_2d::graphics::rectangle, 
		//		protected fan_2d::graphics::line,
		//		protected fan_2d::graphics::gui::text_renderer,
		//		public base::mouse {

		//	public:

		//		checkbox(fan::camera* camera, fan_2d::graphics::gui::theme theme = fan_2d::graphics::gui::themes::deep_blue());

		//		void push_back(const checkbox_property& property);

		//		void draw() const;

		//		void on_check(std::function<void(uint32_t i)> function);
		//		void on_uncheck(std::function<void(uint32_t i)> function);

		//		fan::camera* get_camera();

		//		using fan_2d::graphics::rectangle::size;
		//		using fan_2d::graphics::rectangle::inside;

		//	protected:

		//		std::function<void(uint32_t)> m_on_check;
		//		std::function<void(uint32_t)> m_on_uncheck;

		//		fan_2d::graphics::gui::theme m_theme;

		//		std::deque<bool> m_visible;
		//		std::deque<checkbox_property> m_properties;

		//	};
		}

	}
}