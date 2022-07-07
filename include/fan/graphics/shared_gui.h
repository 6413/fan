#pragma once

#include _FAN_PATH(types/types.h)

#include _FAN_PATH(graphics/shared_graphics.h)

#include _FAN_PATH(graphics/graphics.h)

#include _FAN_PATH(font.h)

#include _FAN_PATH(types/utf_string.h)

#include _FAN_PATH(physics/collision/rectangle.h)
#include _FAN_PATH(physics/collision/circle.h)

#include _FAN_PATH(graphics/gui/button_event.h)
#include _FAN_PATH(graphics/gui/key_event.h)
#include _FAN_PATH(graphics/gui/themes.h)
#include _FAN_PATH(graphics/gui/focus.h)

#if fan_renderer == fan_renderer_opengl
	#include _FAN_PATH(graphics/gui/rectangle_text_box.h)
	#include _FAN_PATH(graphics/gui/rectangle_text_button.h)
#endif

//#include _FAN_PATH(graphics/gui/text_renderer_clickable.h)

//#include _FAN_PATH(graphics/gui/rectangle_text_button_sized.h)
//#include _FAN_PATH(graphics/gui/checkbox.h)
//#include _FAN_PATH(graphics/gui/select_box.h)

namespace fan_2d {

	namespace opengl {

		namespace gui {

			using namespace fan_2d::graphics::gui;

			static fan::vec2 get_resize_movement_offset(fan::window_t* window)
			{
				return fan::cast<f32_t>(window->get_size() - window->get_previous_size());
			}

			static void add_resize_callback(fan::window_t* window, fan::vec2* position) {
				window->add_resize_callback(position, [](fan::window_t* window, const fan::vec2i&, void* position) {
					*(fan::vec2*)position += fan_2d::opengl::gui::get_resize_movement_offset(window);
				});
			}

			namespace base {

			#define define_get_property_size \
				fan::vec2 get_size(properties_t properties) \
				{ \
					f32_t h = fan_2d::opengl::gui::text_renderer_t::font.line_height * fan_2d::opengl::gui::text_renderer_t::convert_font_size(properties.font_size); \
																																															\
					int64_t new_lines = fan_2d::opengl::gui::text_renderer_t::get_new_lines(properties.text); \
						\
					if (new_lines) { \
						h += text_renderer_t::font.line_height * fan_2d::opengl::gui::text_renderer_t::convert_font_size(properties.font_size) * new_lines; \
					} \
						\
					return (properties.text.empty() ? fan::vec2(0, h) : fan::vec2(fan_2d::opengl::gui::text_renderer_t::get_text_size(properties.text, properties.font_size).x, h)) + properties.padding; \
				}
			}

			//
			//			//struct editable_text_renderer : 
			//			//	public text_renderer_t,
			//			//	public base::mouse<editable_text_renderer>,
			//			//	public fan_2d::opengl::gui::text_input<editable_text_renderer>
			//			//{
			//
			//			//	editable_text_renderer(fan::camera* camera) :
			//			//		editable_text_renderer::text_renderer_t(camera), 
			//			//		editable_text_renderer::text_input(this),
			//			//		editable_text_renderer::mouse(this)
			//			//	{
			//
			//			//	}
			//
			//			//	void push_back(const fan::utf16_string& text, f32_t font_size, fan::vec2 position, const fan::color& text_color) {
			//			//		editable_text_renderer::text_renderer_t::push_back(text, font_size, position, text_color);
			//			//		editable_text_renderer::text_input::push_back(-1, -1, -1);
			//
			//			//		uint32_t index = this->size() - 1;
			//
			//			//		for (int i = 0; i < text.size(); i++) {
			//
			//			//			fan::utf8_string utf8;
			//			//			utf8.push_back(text[i]);
			//
			//			//			auto wc = utf8.to_utf16()[0];
			//
			//			//			FED_AddCharacterToCursor(&m_wed[index], cursor_reference[index], text[i], fan_2d::opengl::gui::text_renderer_t::font.font[wc].metrics.size.x * fan_2d::opengl::gui::text_renderer_t::convert_font_size(object->get_font_size(index)) * line_multiplier);
			//			//		}
			//
			//			//		editable_text_renderer::text_input::update_text(index);
			//			//	}
			//
			//			//	bool inside(uint32_t i) const {
			//
			//			//		auto box_size = editable_text_renderer::text_renderer_t::get_text_size(
			//			//			editable_text_renderer::text_renderer_t::get_text(i),
			//			//			editable_text_renderer::text_renderer_t::get_font_size(i)
			//			//		);
			//
			//			//		auto mouse_position = editable_text_renderer::text_renderer_t::m_camera->m_window->get_mouse_position();
			//
			//			//	
			//			//		f32_t converted = fan_2d::opengl::gui::text_renderer_t::convert_font_size(this->get_font_size(i));
			//			//		auto line_height = fan_2d::opengl::gui::text_renderer_t::font.font['\n'].metrics.size.y * converted;
			//
			//			//		return fan_2d::collision::rectangle::point_inside_no_rotation(
			//			//			mouse_position,
			//			//			editable_text_renderer::text_renderer_t::get_position(i),
			//			//			fan::vec2(box_size.x, line_height)
			//			//		);
			//			//	}
			//
			//			//	uint32_t size() const {
			//			//		return editable_text_renderer::text_renderer_t::size();
			//			//	}
			//
			//			//	// receives uint32_t box_i, uint32_t character number x, uint32_t character number y
			//			//	src_dst_t get_cursor(uint32_t i, uint32_t x, uint32_t y) {
			//			//		f32_t converted = fan_2d::opengl::gui::text_renderer_t::convert_font_size(this->get_font_size(i));
			//			//		auto line_height = fan_2d::opengl::gui::text_renderer_t::font.font['\n'].metrics.size.y * converted;
			//
			//			//		fan::vec2 src, dst;
			//
			//			//		src += text_renderer_t::get_position(i);
			//
			//			//		//src.y -= line_height / 2;
			//
			//			//		uint32_t offset = 0;
			//
			//			//		auto str = this->get_text(i);
			//
			//			//		for (int j = 0; j < y; j++) {
			//			//			while (str[offset++] != '\n') {
			//			//				if (offset >= str.size() - 1) {
			//			//					throw std::runtime_error("string didnt have endline");
			//			//				}
			//			//			}
			//			//		}
			//
			//			//		for (int j = 0; j < x; j++) {
			//			//			wchar_t letter = str[j + offset];
			//			//			if (letter == '\n') {
			//			//				continue;
			//			//			}
			//
			//			//			std::wstring wstr;
			//
			//			//			wstr.push_back(letter);
			//
			//			//			auto letter_info = fan_2d::opengl::gui::text_renderer_t::get_letter_info(fan::utf16_string(wstr).to_utf8().data(), this->get_font_size(i));
			//
			//			//			if (j == x - 1) {
			//			//				src.x += letter_info.metrics.size.x + (letter_info.metrics.advance - letter_info.metrics.size.x) / 2 - 1;
			//			//			}
			//			//			else {
			//			//				src.x += letter_info.metrics.advance;
			//			//			}
			//
			//			//		}
			//
			//			//		src.y += line_height * y;
			//
			//
			//			//		dst = src + fan::vec2(0, line_height);
			//
			//			//		dst = dst - src + fan::vec2(cursor_properties::line_thickness, 0);
			//
			//
			//			//		return { src, dst };
			//			//	}
			//
			//			//	fan::camera* get_camera() {
			//			//		return text_renderer_t::m_camera;
			//			//	}
			//
			//			//	void draw() {
			//			//		text_renderer_t::draw();
			//			//		text_input::draw();
			//			//	}
			//
			//			//	void backspace_callback(uint32_t i) override {}
			//			//	void text_callback(uint32_t i) override {}
			//
			//			//	void lib_add_on_input(uint32_t i, uint16_t key, fan::key_state state, fan_2d::opengl::gui::mouse_stage stage) {}
			//
			//			//	void lib_add_on_mouse_move(uint32_t i, fan_2d::opengl::gui::mouse_stage stage) {}
			//
			//			//	bool locked(uint32_t i) const { return false; }
			//
			//			//};
			//

			//
			//			struct button_properties_t {
			//
			//				button_properties_t() {}
			//
			//				button_properties_t(
			//					const fan::utf16_string& text,
			//					const fan::vec2& position
			//				) : text(text), position(position) {}
			//
			//				fan::utf16_string text = empty_string;
			//
			//				fan::utf16_string place_holder;
			//
			//				fan::vec2 position;
			//				fan::vec2 padding;
			//
			//				f32_t font_size = fan_2d::opengl::gui::defaults::font_size;
			//
			//			};
			//

			//
			//			struct rectangle_text_box : 
			//				protected fan::class_duplicator<fan_2d::opengl::rectangle, 0>,
			//				protected fan::class_duplicator<fan_2d::opengl::rectangle, 1>,
			//				protected graphics::gui::text_renderer_t
			//			{
			//
			//				using properties_t = button_properties_t;
			//
			//				using inner_rect_t = fan::class_duplicator<fan_2d::opengl::rectangle, 0>;
			//				using outer_rect_t = fan::class_duplicator<fan_2d::opengl::rectangle, 1>;
			//
			//				rectangle_text_box(fan::camera* camera, fan_2d::opengl::gui::theme theme);
			//
			//				void push_back(const properties_t& properties);
			//
			//				void draw(uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized);
			//
			//				void set_position(uint32_t i, const fan::vec2& position);
			//
			//				using fan_2d::opengl::rectangle::get_size;
			//
			//				define_get_property_size;
			//
			//				bool inside(uintptr_t i, const fan::vec2& position = fan::math::inf) const;
			//
			//				fan::utf16_string get_text(uint32_t i) const;
			//				void set_text(uint32_t i, const fan::utf16_string& text);
			//
			//				fan::color get_text_color(uint32_t i) const;
			//				void set_text_color(uint32_t i, const fan::color& color);
			//
			//				fan::vec2 get_position(uint32_t i) const;
			//				fan::vec2 get_size(uint32_t i) const;
			//
			//				fan::vec2 get_padding(uint32_t i) const;
			//
			//				f32_t get_font_size(uint32_t i) const;
			//
			//				properties_t get_property(uint32_t i) const;
			//
			//				fan::color get_color(uint32_t i) const;
			//
			//				// receives uint32_t box_i, uint32_t character number x, uint32_t character number y
			//				src_dst_t get_cursor(uint32_t i, uint32_t x, uint32_t y);
			//
			//				fan::vec2 get_text_starting_point(uint32_t i) const;
			//
			//				fan::camera* get_camera();
			//
			//				uintptr_t size() const;
			//
			//				void write_data();
			//
			//				void edit_data(uint32_t i);
			//
			//				void edit_data(uint32_t begin, uint32_t end);
			//
			//				void erase(uint32_t i);
			//				void erase(uint32_t begin, uint32_t end);
			//
			//				void clear();
			//
			//				void set_theme(fan_2d::opengl::gui::theme theme_);
			//
			//				void set_theme(uint32_t i, fan_2d::opengl::gui::theme theme_);
			//
			//				void enable_draw();
			//				void disable_draw();
			//
			//				using inner_rect_t::get_color;
			//				using inner_rect_t::set_color;
			//
			//				using graphics::gui::text_renderer_t::font;
			//
			//				fan_2d::opengl::gui::theme theme;
			//
			//			protected:
			//
			//				std::vector<button_properties_t> m_properties;
			//
			//			};
			//
			//
			//			struct rectangle_text_button :
			//				public fan_2d::opengl::gui::rectangle_text_box,
			//				public fan_2d::opengl::gui::base::mouse<rectangle_text_button>,
			//				public fan_2d::opengl::gui::text_input<rectangle_text_button>
			//			{
			//
			//				struct properties_t : public rectangle_text_box::properties_t{
			//					f32_t character_width = (f32_t)0xdfffffff / rectangle_text_button::text_input::line_multiplier;
			//					uint32_t character_limit = 99;
			//					uint32_t line_limit = 99;
			//				};
			//
			//				using input_instance_t = fan_2d::opengl::gui::text_input<rectangle_text_button>;
			//
			//				rectangle_text_button(fan::camera* camera, fan_2d::opengl::gui::theme theme);
			//
			//				void push_back(const properties_t& properties);
			//
			//				void set_place_holder(uint32_t i, const fan::utf16_string& place_holder);
			//
			//				void draw();
			//
			//				void backspace_callback(uint32_t i) override;
			//				void text_callback(uint32_t i) override;
			//
			//				void erase(uint32_t i);
			//				void erase(uint32_t begin, uint32_t end);
			//
			//				void clear();
			//
			//				void set_locked(uint32_t i);
			//
			//				bool locked(uint32_t i) const;
			//
			//				void lib_add_on_input(fan::window_t *window, uint32_t i, uint16_t key, fan::key_state state, fan_2d::opengl::gui::mouse_stage stage);
			//
			//				void lib_add_on_mouse_move(fan::window_t *window, uint32_t i, fan_2d::opengl::gui::mouse_stage stage);
			//
			//			};
			//

			/*	struct circle_button :
					protected fan_2d::opengl::circle,
					public fan_2d::opengl::gui::base::mouse<circle_button>
				{

					struct properties_t {

						properties_t() {}

						fan::vec2 position;

						f32_t radius;

						fan_2d::opengl::gui::theme theme;

						button_states_e button_state = button_states_e::clickable;
					};

					circle_button(fan::camera* camera);
					~circle_button();

					using fan_2d::opengl::circle::inside;
					using fan_2d::opengl::circle::size;

					void push_back(properties_t properties);

					void erase(uint32_t i);
					void erase(uint32_t begin, uint32_t end);
					void clear();

					void set_locked(uint32_t i, bool flag);

					bool locked(uint32_t i) const;

					void enable_draw();
					void disable_draw();

					void update_theme(uint32_t i);

					void lib_add_on_input(fan::window_t *window, fan::opengl::context_t* context, uint32_t i, uint16_t key, fan::key_state state, fan_2d::opengl::gui::mouse_stage stage);

					void lib_add_on_mouse_move(fan::window_t *window, fan::opengl::context_t* context, uint32_t i, fan_2d::opengl::gui::mouse_stage stage);

					fan::camera* get_camera();

				protected:

					std::vector<fan_2d::opengl::gui::theme> m_theme;
					std::vector<uint32_t> m_reserved;

				};*/

			//
			//			struct rectangle_selectable_button_sized : public sprite{
			//
			//				rectangle_selectable_button_sized(fan::camera* camera);
			//
			//				uint32_t get_selected(uint32_t i) const;
			//				void set_selected(uint32_t i);
			//
			//				void add_on_select(std::function<void(uint32_t i)> function);
			//
			//				void lib_add_on_input(fan::window_t *window, uint32_t i, uint16_t key, fan::key_state state, fan_2d::opengl::gui::mouse_stage stage) override;
			//
			//				void lib_add_on_mouse_move(fan::window_t *window, uint32_t i, fan_2d::opengl::gui::mouse_stage stage) override;
			//
			//			protected:
			//
			//				std::vector<std::function<void(uint32_t)>> m_on_select;
			//
			//				uint32_t m_selected = (uint32_t)fan::uninitialized;
			//
			//			};
			//
			//			namespace text_box_properties {
			//
			//				inline int blink_speed(500); // ms
			//				inline fan::color cursor_color(fan::colors::white);
			//				inline fan::color select_color(fan::colors::blue - fan::color(0, 0, 0, 0.5));
			//
			//			}
			//
			//			namespace font_properties {
			//
			//				inline f32_t space_width(15); // remove pls
			//
			//			}
			//
			//			class sprite_text_box :
			//				protected fan::class_duplicator<fan_2d::opengl::sprite_t, 0>,
			//				protected fan_2d::opengl::gui::text_renderer_t {
			//
			//			public:
			//
			//				using properties_t = button_properties_t;
			//
			//				using sprite_t = fan::class_duplicator<fan_2d::opengl::sprite_t, 0>;
			//
			//				sprite_text_box(fan::camera* camera, const std::string& path);
			//
			//				void push_back(const properties_t& properties);
			//
			//				void draw(uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized);
			//
			//				define_get_property_size
			//
			//					fan::camera* get_camera();
			//
			//				uint64_t size() const;
			//				bool inside(uint32_t i, const fan::vec2& position = fan::math::inf) const;
			//
			//				fan::opengl::image_t image;
			//
			//			protected:
			//
			//				std::vector<properties_t> m_properties;
			//			};
			//
			//			struct sprite_text_button :
			//				public fan_2d::opengl::gui::sprite_text_box,
			//				public base::mouse<sprite_text_button> {
			//
			//				sprite_text_button(fan::camera* camera, const std::string& path);
			//
			//				void lib_add_on_input(fan::window_t *window, uint32_t i, uint16_t key, fan::key_state state, fan_2d::opengl::gui::mouse_stage stage);
			//
			//				void lib_add_on_mouse_move(fan::window_t *window, uint32_t i, fan_2d::opengl::gui::mouse_stage stage);
			//
			//				bool locked(uint32_t i) const { return false; }
			//
			//			};
			//
			//			struct scrollbar : public fan_2d::opengl::rectangle, public fan_2d::opengl::gui::base::mouse<scrollbar> {
			//
			//				enum class scroll_direction_e {
			//					horizontal,
			//					vertical,
			//				};
			//
			//				struct properties_t {
			//					fan::vec2 position;
			//					fan::vec2 size;
			//					fan::color color;
			//					scroll_direction_e scroll_direction;
			//					f32_t current;
			//					f32_t length;
			//					uint32_t outline_thickness;
			//				};
			//
			//				using on_scroll_t = std::function<void(uint32_t i, f32_t current)>;
			//
			//				scrollbar(fan::camera* camera);
			//
			//				void push_back(const properties_t& instance);
			//
			//				void draw();
			//
			//				void write_data();
			//
			//				fan::camera* get_camera();
			//
			//				void add_on_scroll(on_scroll_t function);
			//
			//			protected:
			//
			//				struct scroll_properties_t {
			//					scroll_direction_e scroll_direction;
			//					f32_t length;
			//					f32_t current;
			//					uint32_t outline_thickness;
			//				};
			//
			//				std::vector<scroll_properties_t> m_properties;
			//
			//				std::vector<on_scroll_t> m_on_scroll;
			//
			//			};
			//
			//			// takes slider value type as parameter
			//			template <typename T>
			//			class circle_slider : protected fan_2d::opengl::circle, protected fan_2d::opengl::rounded_rectangle {
			//			public:
			//
			//				struct property_t {
			//
			//					T min;
			//					T max;
			//
			//					T current;
			//
			//					fan::vec2 position;
			//					fan::vec2 box_size;
			//					f32_t box_radius;
			//					f32_t button_radius;
			//					fan::color box_color;
			//					fan::color button_color;
			//
			//					fan::color text_color = fan_2d::opengl::gui::defaults::text_color;
			//
			//					f32_t font_size = fan_2d::opengl::gui::defaults::font_size;
			//				};
			//
			//				circle_slider(fan::camera* camera)
			//					: fan_2d::opengl::circle(camera), fan_2d::opengl::rounded_rectangle(camera), m_click_begin(fan::uninitialized), m_moving_id(fan::uninitialized)
			//				{
			//					camera->m_window->add_key_callback(fan::mouse_left, fan::key_state::press, [&] (fan::window_t*) {
			//
			//						// last ones are on the bottom
			//						for (uint32_t i = fan_2d::opengl::circle::size(); i-- ; ) {
			//							if (fan_2d::opengl::circle::inside(i)) {
			//
			//								m_click_begin = fan_2d::opengl::rounded_rectangle::m_camera->m_window->get_mouse_position();
			//								m_moving_id = i;
			//
			//								return;
			//							}
			//						}
			//
			//						const fan::vec2 mouse_position = fan_2d::opengl::rounded_rectangle::m_camera->m_window->get_mouse_position();
			//
			//						for (uint32_t i = fan_2d::opengl::circle::size(); i-- ; ) {
			//
			//							const fan::vec2 box_position = fan_2d::opengl::rounded_rectangle::get_position(i);
			//							const fan::vec2 box_size = fan_2d::opengl::rounded_rectangle::get_size(i);
			//
			//							const f32_t circle_diameter = fan_2d::opengl::circle::get_radius(i) * 2;
			//
			//							const bool horizontal = box_size.x > box_size.y;
			//
			//							if (fan_2d::collision::rectangle::point_inside_no_rotation(mouse_position, box_position - box_size - fan::vec2(horizontal ? 0 : circle_diameter / 2 - 2, horizontal ? circle_diameter / 2 - 2 : 0), box_position + fan::vec2(horizontal ? box_size.x : circle_diameter, horizontal ? circle_diameter : box_size.y))) {
			//
			//								m_click_begin = fan_2d::opengl::rounded_rectangle::m_camera->m_window->get_mouse_position();
			//								m_moving_id = i;
			//
			//								fan::vec2 circle_position = fan_2d::opengl::circle::get_position(i);
			//
			//								if (horizontal) {
			//									fan_2d::opengl::circle::set_position(m_moving_id, fan::vec2(mouse_position.x, circle_position.y));
			//								}
			//								else {
			//									fan_2d::opengl::circle::set_position(m_moving_id, fan::vec2(circle_position.x, mouse_position.y));
			//								}
			//
			//								circle_position = fan_2d::opengl::circle::get_position(i);
			//
			//								f32_t min = get_min_value(m_moving_id);
			//								f32_t max = get_max_value(m_moving_id);
			//
			//								f32_t length = box_size[!horizontal] * 2;
			//
			//								T new_value = min + (((circle_position[!horizontal] - (box_position[!horizontal] - box_size[!horizontal])) / length) * (max - min));
			//
			//								if (new_value == get_current_value(m_moving_id)) {
			//									return;
			//								}
			//
			//								set_current_value(m_moving_id, new_value);
			//
			//								for (int i = 0; i < 2; i++) {
			//									if (m_on_click[i]) {
			//										m_on_click[i](m_moving_id);
			//									}
			//									if (m_on_drag[i]) {
			//										m_on_drag[i](m_moving_id);
			//									}
			//								}
			//							}
			//						}
			//					});
			//
			//					camera->m_window->add_key_callback(fan::mouse_left, fan::key_state::release, [&] (fan::window_t*) {
			//
			//						m_click_begin = fan::uninitialized;
			//						m_moving_id = fan::uninitialized;
			//
			//					});
			//
			//					camera->m_window->add_mouse_move_callback([&](fan::window_t*, const fan::vec2& position) {
			//
			//						if (m_click_begin == fan::uninitialized) {
			//							return;
			//						}
			//
			//						const fan::vec2 box_position = fan_2d::opengl::rounded_rectangle::get_position(m_moving_id);
			//						const fan::vec2 box_size = fan_2d::opengl::rounded_rectangle::get_size(m_moving_id);
			//
			//						fan::vec2 circle_position = fan_2d::opengl::circle::get_position(m_moving_id);
			//
			//						const bool horizontal = box_size.x > box_size.y;
			//
			//						if (horizontal) {
			//						}
			//						else {
			//							circle_position.y = m_click_begin.y + (position.y - m_click_begin.y);
			//						}
			//
			//						f32_t length = box_size[!horizontal] * 2;
			//
			//						f32_t min = get_min_value(m_moving_id);
			//						f32_t max = get_max_value(m_moving_id);
			//
			//						circle_position[!horizontal] = m_click_begin[!horizontal] + (position[!horizontal] - m_click_begin[!horizontal]);
			//
			//						circle_position = circle_position.clamp(
			//							fan::vec2(box_position.x - box_size.x - circle::get_radius(m_moving_id), box_position.y - box_size.y - circle::get_radius(m_moving_id)),
			//							fan::vec2(box_position.x + box_size.x, box_position.y + box_size.y)
			//						);
			//
			//						T new_value = min + (((circle_position[!horizontal] - (box_position[!horizontal] - box_size[!horizontal] - circle::get_radius(m_moving_id) )) / length) * (max - min));
			//
			//						if (new_value == get_current_value(m_moving_id)) {
			//							return;
			//						}
			//
			//						new_value = fan::clamp(new_value, (T)min, (T)max);
			//
			//						set_current_value(m_moving_id, new_value);
			//
			//						fan_2d::opengl::circle::set_position(m_moving_id, circle_position);
			//
			//						for (int i = 0; i < 2; i++) {
			//							if (m_on_drag[i]) {
			//								m_on_drag[i](m_moving_id);
			//							}
			//						}
			//
			//						this->edit_data(m_moving_id);
			//					});
			//				}
			//
			//				void push_back(const circle_slider<T>::property_t& property)
			//				{
			//					m_properties.emplace_back(property);
			//					m_properties[m_properties.size() - 1].current = fan::clamp(m_properties[m_properties.size() - 1].current, property.min, property.max);
			//
			//					rounded_rectangle::properties_t properties;
			//					properties.position = property.position;
			//					properties.size = property.box_size;
			//					properties.radius = property.box_radius;
			//					properties.color = property.box_color;
			//					properties.angle = 0;
			//
			//					fan_2d::opengl::rounded_rectangle::push_back(properties);
			//
			//					if (property.box_size.x > property.box_size.y) {
			//
			//						f32_t min = property.position.x;
			//						f32_t max = property.position.x + property.box_size.x * 2;
			//
			//						f32_t new_x = (f32_t(property.current - property.min) / (property.max - property.min)) * (max - min);
			//
			//						fan_2d::opengl::circle::properties_t cp;
			//						cp.position = property.position + fan::vec2(new_x - property.box_size.x, 0);
			//						cp.radius = property.button_radius;
			//						cp.color = property.button_color;
			//
			//						fan_2d::opengl::circle::push_back(cp);
			//					}
			//					else {
			//
			//						f32_t min = property.position.y;
			//						f32_t max = property.position.y + property.box_size.y * 2;
			//
			//						f32_t new_y = (f32_t(property.current - property.min) / (property.max - property.min)) * (max - min);
			//
			//						fan_2d::opengl::circle::properties_t cp;
			//						cp.position = property.position + fan::vec2(0, new_y - property.box_size.y);
			//						cp.radius = property.button_radius;
			//						cp.color = property.button_color;
			//
			//						fan_2d::opengl::circle::push_back(cp);
			//					}
			//				}
			//
			//				auto get_min_value(uint32_t i) const {
			//					return m_properties[i].min;
			//				}
			//				void set_min_value(uint32_t i, T value) {
			//					m_properties[i].min = value;
			//				}
			//				auto get_max_value(uint32_t i) const {
			//					return m_properties[i].max;
			//				}
			//				void set_max_value(uint32_t i, T value) {
			//					m_properties[i].max = value;
			//				}
			//				auto get_current_value(uint32_t i) const {
			//					return m_properties[i].current;
			//				}
			//				void set_current_value(uint32_t i, T value) {
			//					m_properties[i].current = value;
			//				}
			//
			//				void on_drag(const std::function<void(uint32_t)>& function) {
			//					on_drag(true, function);
			//					m_on_drag[1] = function;
			//				}
			//
			//				void write_data() {
			//					fan_2d::opengl::rounded_rectangle::write_data();
			//					fan_2d::opengl::circle::write_data();
			//				}
			//				void edit_data(uint32_t i) {
			//					fan_2d::opengl::rounded_rectangle::edit_data(i);
			//					fan_2d::opengl::circle::edit_data(i);
			//				}
			//				void edit_data(uint32_t begin, uint32_t end) {
			//					fan_2d::opengl::rounded_rectangle::edit_data(begin, end);
			//					fan_2d::opengl::circle::edit_data(begin, end);
			//				}
			//
			//				void enable_draw() {
			//					fan_2d::opengl::rounded_rectangle::enable_draw();
			//					fan_2d::opengl::circle::enable_draw();
			//				}
			//				void disable_draw() {
			//					fan_2d::opengl::rounded_rectangle::disable_draw();
			//					fan_2d::opengl::circle::disable_draw();
			//				}
			//
			//			protected:
			//
			//				void on_drag(bool user, const std::function<void(uint32_t)>& function) {
			//					m_on_drag[user] = function;
			//				}
			//
			//				void on_click(bool user, const std::function<void(uint32_t)>& function) {
			//					m_on_click[user] = function;
			//				}
			//
			//				std::deque<circle_slider<T>::property_t> m_properties;
			//
			//				fan::vec2 m_click_begin;
			//
			//				uint32_t m_moving_id;
			//
			//				std::function<void(uint32_t i)> m_on_drag[2]; // 0 lib, 1 user
			//				std::function<void(uint32_t)> m_on_click[2]; // 0 lib, 1 user
			//
			//			};
			//
			//			template <typename T>
			//			class circle_text_slider : public circle_slider<T>, protected fan_2d::opengl::gui::text_renderer_t {
			//			public:
			//
			//				using value_type_t = T;
			//
			//				static constexpr f32_t text_gap_multiplier = 1.5;
			//
			//				circle_text_slider(fan::camera* camera) : circle_slider<T>(camera), fan_2d::opengl::gui::text_renderer_t(camera) {
			//					circle_text_slider::circle_slider::on_drag(false, [&](uint32_t i) {
			//						auto new_string = fan::to_wstring(this->get_current_value(i));
			//
			//						fan_2d::opengl::gui::text_renderer_t::set_text(i * 3 + 2, new_string);
			//					});
			//
			//					circle_text_slider::circle_slider::on_click(false, [&] (uint32_t i) {
			//
			//						auto new_string = fan::to_wstring(this->get_current_value(i));
			//
			//						bool resize = text_renderer_t::get_text(i * 3 + 2).size() != new_string.size();
			//
			//						fan_2d::opengl::gui::text_renderer_t::set_text(i * 3 + 2, new_string);
			//
			//					});
			//				}
			//
			//				void push_back(const typename circle_slider<T>::property_t& property) {
			//					circle_text_slider::circle_slider::push_back(property);
			//
			//					const fan::vec2 left_text_size = fan_2d::opengl::gui::text_renderer_t::get_text_size(fan::to_wstring(property.min), property.font_size);
			//
			//					fan::vec2 left_or_up;
			//
			//					if (property.box_size.x > property.box_size.y) {
			//						left_or_up = property.position - fan::vec2(property.box_size.x, left_text_size.y / 2 + property.button_radius * text_gap_multiplier);
			//					}
			//					else {
			//						left_or_up = property.position + fan::vec2(left_text_size.x + property.button_radius * text_gap_multiplier, -property.box_size.y + left_text_size.y / 2 - property.button_radius);
			//					}
			//
			//					fan_2d::opengl::gui::text_renderer_t::properties_t p3;
			//					p3.text = fan::to_wstring(property.min);
			//					p3.font_size = property.font_size;
			//					p3.position = left_or_up;
			//					p3.text_color = fan_2d::opengl::gui::defaults::text_color;
			//
			//					fan_2d::opengl::gui::text_renderer_t::push_back(
			//						p3
			//					);
			//
			//					const fan::vec2 right_text_size = fan_2d::opengl::gui::text_renderer_t::get_text_size(fan::to_wstring(property.max), property.font_size);
			//
			//					fan::vec2 right_or_down;
			//
			//					if (property.box_size.x > property.box_size.y) {
			//						right_or_down = property.position + fan::vec2(property.box_size.x + right_text_size.x / 4, -right_text_size.y / 2 - property.button_radius * text_gap_multiplier);
			//					}
			//					else {
			//						right_or_down = property.position + fan::vec2(left_text_size.x + property.button_radius * text_gap_multiplier, property.box_size.y - left_text_size.y / 2 + property.button_radius);
			//					}
			//
			//					fan_2d::opengl::gui::text_renderer_t::properties_t p;
			//					p.text = fan::to_wstring(property.max);
			//					p.font_size = property.font_size;
			//					p.position = right_or_down;
			//					p.text_color = fan_2d::opengl::gui::defaults::text_color;
			//
			//					fan_2d::opengl::gui::text_renderer_t::push_back(
			//						p
			//					);
			//
			//					const fan::vec2 middle_text_size = fan_2d::opengl::gui::text_renderer_t::get_text_size(fan::to_wstring(property.current), property.font_size);
			//
			//					fan::vec2 middle;
			//
			//					if (property.box_size.x > property.box_size.y) {
			//						middle = property.position + fan::vec2(0, -middle_text_size.y / 2 - property.button_radius * text_gap_multiplier);
			//					}
			//					else {
			//						middle = property.position + fan::vec2(middle_text_size.x + property.button_radius * text_gap_multiplier, 0);
			//					}
			//
			//					fan_2d::opengl::gui::text_renderer_t::properties_t p2;
			//					p2.text = fan::to_wstring(property.current);
			//					p2.font_size = property.font_size;
			//					p2.position = middle;
			//					p2.text_color = fan_2d::opengl::gui::defaults::text_color;
			//
			//					fan_2d::opengl::gui::text_renderer_t::push_back(
			//						p2
			//					);
			//				}
			//
			//				void draw() {
			//					// depth test
			//					//fan_2d::opengl::draw([&] {
			//						fan_2d::opengl::gui::circle_slider<value_type_t>::draw();
			//						fan_2d::opengl::gui::text_renderer_t::draw();
			//					//});
			//				}
			//
			//				void write_data() {
			//					circle_text_slider::circle_slider::write_data();
			//					circle_text_slider::text_renderer_t::write_data();
			//				}
			//				void edit_data(uint32_t i) {
			//					circle_text_slider::circle_slider::edit_data(i);
			//					circle_text_slider::text_renderer_t::edit_data(i * 3, i * 3 + 2);
			//				}
			//				void edit_data(uint32_t begin, uint32_t end) {
			//					circle_text_slider::circle_slider::edit_data(begin, end);
			//					circle_text_slider::text_renderer_t::edit_data(begin * 3, end * 3 + 3); // ?
			//				}
			//				
			//				void enable_draw() {
			//					circle_text_slider::circle_slider::enable_draw();
			//					circle_text_slider::text_renderer_t::enable_draw();
			//				}
			//				void disable_draw() {
			//					circle_text_slider::circle_slider::disable_draw();
			//					circle_text_slider::text_renderer_t::disable_draw();
			//				}
			//
			//
			//				using circle_slider<T>::enable_draw;
			//				using circle_slider<T>::disable_draw;
			//
			//			};
			//
			//			class checkbox : 
			//				protected fan::class_duplicator<fan_2d::opengl::line, 99>,
			//				protected fan::class_duplicator<fan_2d::opengl::rectangle, 99>, 
			//				protected fan::class_duplicator<fan_2d::opengl::gui::text_renderer_t, 99>,
			//				public base::mouse<checkbox> {
			//
			//			protected:
			//
			//				using line_t = fan::class_duplicator<fan_2d::opengl::line, 99>;
			//				using rectangle_t = fan::class_duplicator<fan_2d::opengl::rectangle, 99>;
			//				using text_renderer_t = fan::class_duplicator<fan_2d::opengl::gui::text_renderer_t, 99>;
			//
			//			public:
			//
			//				struct properties_t {
			//					fan::vec2 position;
			//
			//					f32_t font_size;
			//
			//					fan::utf16_string text;
			//
			//					uint8_t line_thickness = 2;
			//
			//					f32_t box_size_multiplier = 1;
			//
			//					bool checked = false;
			//				};
			//
			//				checkbox(fan::camera* camera, fan_2d::opengl::gui::theme theme);
			//
			//				void push_back(const checkbox::properties_t& property);
			//
			//				void draw();
			//
			//				void on_check(std::function<void(uint32_t i)> function);
			//				void on_uncheck(std::function<void(uint32_t i)> function);
			//
			//				uint32_t size() const;
			//				bool inside(uint32_t i, const fan::vec2& position = fan::math::inf) const;
			//
			//				fan::camera* get_camera();
			//
			//				void write_data();
			//				void edit_data(uint32_t i);
			//				void edit_data(uint32_t begin, uint32_t end);
			//
			//				void lib_add_on_input(fan::window_t *window, uint32_t i, uint16_t key, fan::key_state state, fan_2d::opengl::gui::mouse_stage stage);
			//
			//				void lib_add_on_mouse_move(fan::window_t *window, uint32_t i, fan_2d::opengl::gui::mouse_stage stage);
			//
			//				bool locked(uint32_t i) const { return false; }
			//
			//				void enable_draw();
			//				void disable_draw();
			//
			//			protected:
			//
			//				std::function<void(uint32_t)> m_on_check;
			//				std::function<void(uint32_t)> m_on_uncheck;
			//
			//				fan_2d::opengl::gui::theme m_theme;
			//
			//				std::deque<bool> m_visible;
			//				std::deque<checkbox::properties_t> m_properties;
			//
			//			};
			//
			//			struct dropdown_menu : protected sprite {
			//
			//				struct properties_t {
			//					fan::utf16_string text = empty_string;
			//
			//					fan::vec2 position;
			//
			//					f32_t font_size = fan_2d::opengl::gui::defaults::font_size;
			//
			//					text_position_e text_position = text_position_e::left;
			//
			//					fan::vec2 size;
			//
			//					f32_t advance = 0;
			//
			//					std::vector<fan::utf16_string> dropdown_texts;
			//				};
			//
			//				dropdown_menu(fan::camera* camera, const fan_2d::opengl::gui::theme& theme);
			//
			//				void push_back(const properties_t& property);
			//
			//				void draw();
			//
			//				using sprite::get_camera;
			//
			//			protected:
			//
			//				std::vector<src_dst_t> m_hitboxes;
			//
			//				uint32_t m_hovered = -1;
			//
			//				std::deque<uint32_t> m_amount_per_menu;
			//
			//			};
			//
			//			struct progress_bar : protected fan_2d::opengl::rectangle {
			//
			//			protected:
			//
			//				using rectangle_t = fan_2d::opengl::rectangle;
			//
			//				struct progress_bar_t {
			//					f32_t inner_size_multipliers;
			//					f32_t progress;
			//					bool progress_x = false;
			//				};
			//
			//				std::vector<progress_bar_t> m_progress_bar_properties;
			//
			//			public:
			//
			//				progress_bar(fan::camera* camera);
			//
			//				struct properties_t {
			//
			//					fan::vec2 position;
			//					fan::vec2 size;
			//
			//					fan::color back_color;
			//					fan::color front_color;
			//
			//					f32_t inner_size_multiplier = 0.8;
			//
			//					f32_t progress = 0;
			//
			//					bool progress_x = true;
			//
			//				};
			//
			//				void push_back(const properties_t& properties);
			//
			//				void clear();
			//
			//				f32_t get_progress(uint32_t i) const;
			//				void set_progress(uint32_t i, f32_t progress);
			//
			//				fan::vec2 get_position(uint32_t i) const;
			//				void set_position(uint32_t i, const fan::vec2& position);
			//
			//				using fan_2d::opengl::rectangle::enable_draw;
			//				using fan_2d::opengl::rectangle::disable_draw;
			//
			//			};
			//

		}
	}
}