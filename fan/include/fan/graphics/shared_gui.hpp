#pragma once

#define FED_set_debug_InvalidLineAccess 1
#include <fan/fed/FED.h>

#include <fan/graphics/shared_graphics.hpp>

#include <fan/graphics/graphics.hpp>

#include <fan/font.hpp>

#include <fan/types/utf_string.hpp>

#include <fan/physics/collision/rectangle.hpp>
#include <fan/physics/collision/circle.hpp>

namespace fan_2d {

	namespace graphics {

		namespace gui {

			static fan::utf16_string get_empty_string() {
				fan::utf16_string str;

				str.resize(1);

				return str;
			}

			inline fan::utf16_string empty_string = get_empty_string();

			static fan::vec2 get_resize_movement_offset(fan::window* window)
			{
				return fan::cast<f32_t>(window->get_size() - window->get_previous_size());
			}

			static void add_resize_callback(fan::window* window, fan::vec2& position) {
				window->add_resize_callback([&](fan::window* window, const fan::vec2i&) {
					position += fan_2d::graphics::gui::get_resize_movement_offset(window);
				});
			}

			struct rectangle : public fan_2d::graphics::rectangle {

				rectangle(fan::camera* camera)
					: fan_2d::graphics::rectangle(camera)
				{
					this->m_camera->m_window->add_resize_callback([&](const fan::window* window, const fan::vec2i&) {

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
				protected fan::buffer_object<f32_t, 98>,
				protected fan::buffer_object<fan::vec2, 100>,
				protected fan::buffer_object<fan::color, 102>,
				protected fan::buffer_object<f32_t, 103>

#elif fan_renderer == fan_renderer_vulkan

#endif
			
			{
			public:


				struct properties_t {
					fan::utf16_string text; 
					f32_t font_size; 
					fan::vec2 position;
					fan::color text_color;
					fan::color outline_color = fan::color(0, 0, 0, 0);
					f32_t outline_size = 0;
				};

				text_renderer(fan::camera* camera);

				~text_renderer();

				void push_back(properties_t properties);

				void insert(uint32_t i, properties_t properties);

				fan::vec2 get_position(uint32_t i) const {
					return m_position[i];
				}
				void set_position(uint32_t i, const fan::vec2& position);

				uint32_t size() const;

				static fan::font::single_info_t get_letter_info(wchar_t c, f32_t font_size) {
					auto found = font.font.find(c);

					if (found == font.font.end()) {
						throw std::runtime_error("failed to find character: " + std::to_string((int)c));
					}

					f32_t converted_size = text_renderer::convert_font_size(font_size);

					fan::font::single_info_t font_info;
					font_info.metrics.size = found->second.metrics.size * converted_size;
					font_info.metrics.offset = found->second.metrics.offset * converted_size;
					font_info.metrics.advance = (found->second.metrics.advance * converted_size);

					font_info.glyph = found->second.glyph;
					font_info.mapping = found->second.mapping;

					return font_info;
				}

				static fan::font::single_info_t get_letter_info(uint8_t* c, f32_t font_size) {

					auto found = font.font.find(fan::utf16_string(c).data()[0]);

					if (found == font.font.end()) {
						throw std::runtime_error("failed to find character: " + std::to_string(fan::utf16_string(c).data()[0]));
					}

					f32_t converted_size = text_renderer::convert_font_size(font_size);

					fan::font::single_info_t font_info;
					font_info.metrics.size = found->second.metrics.size * converted_size;
					font_info.metrics.offset = found->second.metrics.offset * converted_size;
					font_info.metrics.advance = (found->second.metrics.advance * converted_size);

					font_info.glyph = found->second.glyph;
					font_info.mapping = found->second.mapping;

					return font_info;
				}

				fan::vec2 get_character_position(uint32_t i, uint32_t j, f32_t font_size) const {
					fan::vec2 position = text_renderer::get_position(i);

					auto converted_size = convert_font_size(font_size);

					for (int k = 0; k < j; k++) {
						position.x += font.font[m_text[i][k]].metrics.advance * converted_size;
					}

					position.y = i * (font.line_height * converted_size);

					return position;
				}

				f32_t get_font_size(uintptr_t i) const;
				void set_font_size(uint32_t i, f32_t font_size);

				void set_angle(uint32_t i, f32_t angle);

				void set_rotation_point(uint32_t i, const fan::vec2& rotation_point);

				fan::color get_outline_color(uint32_t i) const;
				void set_outline_color(uint32_t i, const fan::color& outline_color);

				f32_t get_outline_size(uint32_t i) const;
				void set_outline_size(uint32_t i, f32_t outline_size);

				static f32_t convert_font_size(f32_t font_size) {
					return font_size / font.size;
				}

				void erase(uintptr_t i);

				void erase(uintptr_t begin, uintptr_t end);

				void clear();

				static f32_t get_line_height(f32_t font_size) {
					return font.line_height * convert_font_size(font_size);
				}

				fan::utf16_string get_text(uint32_t i) const {
					return m_text[i];
				}
				void set_text(uint32_t i, const fan::utf16_string& text);

				fan::color get_text_color(uint32_t i, uint32_t j = 0) const;
				void set_text_color(uint32_t i, const fan::color& color);
				void set_text_color(uint32_t i, uint32_t j, const fan::color& color);

				fan::vec2 get_text_size(uint32_t i) {

					uint32_t begin = 0;
					uint32_t end = 0;

					for (int j = 0; j < i; j++) {
						begin += m_text[j].size();
					}

					end = begin + m_text[i].size() - 1;

					auto p_first = sprite::get_position(begin);
					auto p_last = sprite::get_position(end);

					auto s_first = sprite::get_size(begin);
					auto s_last = sprite::get_size(end);

					return fan::vec2((p_last.x + s_last.x) - (p_first.x - s_first.x), font.line_height);
				}

				static fan::vec2 get_text_size(const fan::utf16_string& text, f32_t font_size) {
					fan::vec2 text_size;

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

						auto letter = font.font[text[i]];

						if (i == text.size() - 1) {
							width += letter.glyph.size.x;
						}
						else {
							width += letter.metrics.advance;
						}
					}

					text_size.x = std::max(width, text_size.x);

					return text_size * convert_font_size(font_size);
				}

				static f32_t get_original_font_size() {
					return font.size;
				}

				inline static fan::font::font_t font;
				inline static fan_2d::graphics::image_t font_image = nullptr;

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

				void enable_draw();
				void disable_draw();

			protected:

				void draw(uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized);

				void write_data();

				void edit_data(uint32_t i);

				void edit_data(uint32_t begin, uint32_t end);

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

				std::vector<uint32_t> m_new_lines;

				struct letter_t {
					fan::vec2 texture_position;
					fan::vec2 texture_size;
					std::array<fan::vec2, 4> texture_coordinates;

					fan::vec2 size;
					fan::vec2 offset;
				};

				letter_t get_letter(wchar_t character, f32_t font_size) {

					auto letter_info = get_letter_info(character, font_size);

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

				void push_letter(wchar_t character, properties_t proeprties) {

					bool write_ = m_queue_helper.m_write;

					letter_t letter = get_letter(character, proeprties.font_size);

					fan_2d::graphics::sprite::properties_t p;
					p.position = proeprties.position + fan::vec2(letter.size.x, -letter.size.y) + fan::vec2(letter.offset.x, -letter.offset.y);

					p.size = letter.size;
					p.image = font_image;
					p.texture_coordinates = letter.texture_coordinates;

					fan_2d::graphics::sprite::push_back(p);

					fan_2d::graphics::sprite::set_color(fan_2d::graphics::sprite::size() - 1, proeprties.text_color);

					font_size_t::m_buffer_object.insert(font_size_t::m_buffer_object.end(), 6, proeprties.font_size);
					outline_color_t::m_buffer_object.insert(outline_color_t::m_buffer_object.end(), 6, proeprties.outline_color);
					outline_size_t::m_buffer_object.insert(outline_size_t::m_buffer_object.end(), 6, proeprties.outline_size);

					if (!write_) {
						m_camera->m_window->edit_write_call(m_queue_helper.m_write_index, this, [&] {
							this->write_data();
						});
					}
				}

				void insert_letter(uint32_t i, wchar_t character, properties_t proeprties) {

					bool write_ = m_queue_helper.m_write;

					letter_t letter = get_letter(character, proeprties.font_size);

					fan_2d::graphics::sprite::properties_t p;
					p.position = proeprties.position + fan::vec2(letter.size.x, -letter.size.y) + fan::vec2(letter.offset.x, -letter.offset.y);

					p.size = letter.size;
					p.image = font_image;
					p.texture_coordinates = letter.texture_coordinates;

					fan_2d::graphics::sprite::insert(i, i * 6, p);

					fan_2d::graphics::sprite::set_color(i, proeprties.text_color);

					font_size_t::m_buffer_object.insert(font_size_t::m_buffer_object.begin() + i * 6, 6, proeprties.font_size);
					outline_color_t::m_buffer_object.insert(outline_color_t::m_buffer_object.begin() + i * 6, 6, proeprties.outline_color);
					outline_size_t::m_buffer_object.insert(outline_size_t::m_buffer_object.begin() + i * 6, 6, proeprties.outline_size);

					if (!write_) {
						m_camera->m_window->edit_write_call(m_queue_helper.m_write_index, this, [&] {
							this->write_data();
						});
					}
				}

				constexpr uint32_t get_index(uint32_t i) const {
					return i == 0 ? 0 : m_indices[i - 1];
				}

#if fan_renderer == fan_renderer_opengl

				using font_size_t = fan::buffer_object<f32_t, 98>;
				using rotation_point_t = fan::buffer_object<fan::vec2, 100>;
				using outline_color_t = fan::buffer_object<fan::color, 102>;
				using outline_size_t = fan::buffer_object<f32_t, 103>;

				static constexpr auto location_font_size = "layout_font_size";
				static constexpr auto location_rotation_point = "layout_rotation_point";
				static constexpr auto location_outline_color = "layout_outline_color";
				static constexpr auto location_outline_size = "layout_outline_size";

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

				queue_helper_t m_queue_helper;

				uint32_t m_draw_index = -1;

				view_projection_t view_projection{};

				using instance_buffer_t = fan::gpu_memory::buffer_object<instance_t, fan::gpu_memory::buffer_type::buffer>;

				instance_buffer_t* instance_buffer = nullptr;

				uint32_t m_begin = 0;
				uint32_t m_end = 0;

				fan::gpu_memory::uniform_handler* uniform_handler = nullptr;

				std::vector<VkDeviceSize> descriptor_offsets;
			#ifdef fan_renderer == fan_renderer_vulkan
				uint64_t pipeline_offset = 0;
			#endif


#endif

			};

			struct text_renderer0 : public text_renderer {
				
				text_renderer0(fan::camera* camera, void* gp, std::function<void(void*, uint64_t)> index_change_cb) 
					:	text_renderer(camera),
						m_index_change_cb(index_change_cb),
						m_gp(gp)
				{}

				struct properties_t : public text_renderer::properties_t {
					uint32_t entity_id;
				};

				void push_back(properties_t properties) {
					m_entity_ids.emplace_back(properties.entity_id);
					text_renderer::push_back(properties);
				}

				void erase(uintptr_t i) {
					for(uint32_t start = i + 1; start < text_renderer::size(); start++) {
            m_index_change_cb(m_gp, m_entity_ids[start]);
          }
					text_renderer::erase(i);
					m_entity_ids.erase(m_entity_ids.begin() + i);
				}

				void erase(uintptr_t begin, uintptr_t end) = delete;

			protected:

				std::function<void(void*, uint64_t)> m_index_change_cb;

				void* m_gp = nullptr;
				std::vector<uint64_t> m_entity_ids;

			};

			enum class mouse_stage {
				outside,
				inside,
				outside_drag
			};

			namespace base {

				// requires to have functions: 
				// get_camera() which returns camera, 
				// size() returns amount of objects, 
				// inside() if mouse inside
				// requires push_back to be called in every shape push_back
				// requires to have either add_on_input or add_on_input and add_on_mouse_event
				template <typename T>
				struct mouse {

				protected:

					mouse(T* object_) : object(object_) {

						add_mouse_move_callback_id = object->get_camera()->m_window->add_mouse_move_callback([&] (fan::window*, const fan::vec2&) {
							if (m_do_we_hold_button == 1) {
								return;
							}
							if (m_focused_button_id != fan::uninitialized) {

								if (m_focused_button_id >= object->size()) {
									m_focused_button_id = fan::uninitialized;
							 	}
								else if (object->inside(m_focused_button_id) || object->locked(m_focused_button_id)) {
									return;
								}
							}

							for (int i = object->size(); i--; ) {
								if (object->inside(i) && !object->locked(i)) {
									if (m_focused_button_id != fan::uninitialized) {
										object->lib_add_on_mouse_event(m_focused_button_id, mouse_stage::outside);
										if (on_mouse_event_function) {
											on_mouse_event_function(m_focused_button_id, mouse_stage::outside);
										}
									}
									m_focused_button_id = i;
									object->lib_add_on_mouse_event(m_focused_button_id, mouse_stage::inside);
									if (on_mouse_event_function) {
										on_mouse_event_function(m_focused_button_id, mouse_stage::inside);
									}
									return;
								}
							}
							if (m_focused_button_id != fan::uninitialized) {
								object->lib_add_on_mouse_event(m_focused_button_id, mouse_stage::outside);
								if (on_mouse_event_function) {
									on_mouse_event_function(m_focused_button_id, mouse_stage::outside);
								}
								m_focused_button_id = fan::uninitialized;
							}
							});

						add_keys_callback_id = object->get_camera()->m_window->add_keys_callback([&](fan::window* window, uint16_t key, fan::key_state state) {

							if (m_focused_button_id >= object->size()) {
								m_focused_button_id = fan::uninitialized;
							}

							if (m_do_we_hold_button == 0) {
								if (state == fan::key_state::press) {
									if (m_focused_button_id != fan::uninitialized) {
										m_do_we_hold_button = 1;
										object->lib_add_on_input(m_focused_button_id, key, fan::key_state::press, mouse_stage::inside);
										if (on_input_function) {
											on_input_function(m_focused_button_id, key, fan::key_state::press, mouse_stage::inside);
										}
									}
									else {
										return; // clicked at space
									}
								}
								else {
									return;
								}
							}
							else {
								if (state == fan::key_state::press) {
									return; // double press
								}
								else {
									if (m_focused_button_id >= object->size()) {
										m_focused_button_id = fan::uninitialized;
									}
									else if (object->inside(m_focused_button_id) && !object->locked(m_focused_button_id)) {
										object->lib_add_on_input(m_focused_button_id, key, fan::key_state::release, mouse_stage::inside);
										if (on_input_function) {
											pointer_remove_flag = 1;
											on_input_function(m_focused_button_id, key, fan::key_state::release, mouse_stage::inside);
											if (pointer_remove_flag == 0) {
												return;
												//rtb is deleted
											}
										}
									}
									else {
										object->lib_add_on_input(m_focused_button_id, key, fan::key_state::release, mouse_stage::outside);
										if (on_input_function) {
											pointer_remove_flag = 1;
											on_input_function(m_focused_button_id, key, fan::key_state::release, mouse_stage::outside);
											if (pointer_remove_flag == 0) {
												return;
												//rtb is deleted
											}
											pointer_remove_flag = 0;
										}
									}
									m_do_we_hold_button = 0;
								}
							}

						});
					}

					~mouse() {
						if (add_keys_callback_id != -1) {
							object->get_camera()->m_window->remove_keys_callback(add_keys_callback_id);
							add_keys_callback_id = -1;
						}
						if (add_mouse_move_callback_id != -1) {
							object->get_camera()->m_window->remove_mouse_move_callback(add_mouse_move_callback_id);
							add_mouse_move_callback_id = -1;
						}
					}

				public:

					/*void erase(uint32_t i) {
						m_on_input.erase(m_on_input.begin() + i);

						m_on_mouse_event.erase(m_on_mouse_event.begin() + i);

						input_instance.erase(input_instance.begin() + i);
						mouse_instance.erase(mouse_instance.begin() + i);
					}

					void erase(uint32_t begin, uint32_t end) {
						m_on_input.erase(m_on_input.begin() + begin, m_on_input.begin() + end);

						m_on_mouse_event.erase(m_on_mouse_event.begin() + begin, m_on_mouse_event.begin() + end);

						input_instance.erase(input_instance.begin() + begin, input_instance.begin() + end);
						mouse_instance.erase(mouse_instance.begin() + begin, mouse_instance.begin() + end);
					}*/

					//void clear() {
					//	m_on_input.clear();
					//	// uint32_t index, bool inside
					//	m_on_mouse_event.clear();

					//	input_instance.clear();
					//	mouse_instance.clear();
					//}

				public:

					std::function<void(uint32_t index, uint16_t key, fan::key_state key_state, mouse_stage mouse_stage)> on_input_function;

					std::function<void(uint32_t index, mouse_stage mouse_stage)> on_mouse_event_function;

					// uint32_t index, fan::key_state state, mouse_stage mouse_stage
					void set_on_input(std::function<void(uint32_t index, uint16_t key, fan::key_state key_state, mouse_stage mouse_stage)> function) {
						on_input_function = function;
					}

					// uint32_t index, bool inside
					void set_on_mouse_event(std::function<void(uint32_t index, mouse_stage mouse_stage)> function) {
						on_mouse_event_function = function;
					}

					uint32_t holding_button() const {
						return m_focused_button_id;
					}

				protected:

					static thread_local bool pointer_remove_flag;

					uint8_t m_old_mouse_stage = fan::uninitialized;
					bool m_do_we_hold_button = 0;
					uint32_t m_focused_button_id = fan::uninitialized;
					uint32_t add_keys_callback_id = fan::uninitialized;
					uint32_t add_mouse_move_callback_id = fan::uninitialized;

					T* object;

				};
				
				template <typename T>
				thread_local bool mouse<T>::pointer_remove_flag = 0;

#define define_get_property_size \
				fan::vec2 get_size(properties_t properties) \
				{ \
					f32_t h = fan_2d::graphics::gui::text_renderer::font.line_height * fan_2d::graphics::gui::text_renderer::convert_font_size(properties.font_size); \
																																															\
					int64_t new_lines = fan_2d::graphics::gui::text_renderer::get_new_lines(properties.text); \
						\
					if (new_lines) { \
						h += text_renderer::font.line_height * fan_2d::graphics::gui::text_renderer::convert_font_size(properties.font_size) * new_lines; \
					} \
						\
					return (properties.text.empty() ? fan::vec2(0, h) : fan::vec2(fan_2d::graphics::gui::text_renderer::get_text_size(properties.text, properties.font_size).x, h)) + properties.padding; \
				}
			}

			namespace focus {

				constexpr uint32_t no_focus = -1;

				struct properties_t {

					properties_t() {}
					properties_t(uint32_t x) : window_id((fan::window_t)x), shape(0), i(x) {}
					properties_t(fan::window_t window_id, void* shape, uint32_t i) : 
						window_id(window_id), shape(shape), i(i) {}

					// focus window
					fan::window_t window_id;
					// shape class, initialized with shape = this
					void* shape;
					// index of focus
					uint32_t i;
				};

				inline properties_t current_focus = { 0, 0, focus::no_focus };

				static bool operator==(const properties_t& focus, uint32_t focused) {
					return focus.i == focused;
				}

				static bool operator==(const properties_t& focus, const properties_t& focused) {
					return focus.window_id == focused.window_id && focus.shape == focused.shape;
				}

				static bool operator!=(const properties_t& focus, uint32_t focused) {
					return focus.i != focused;
				}

				static properties_t get_focus() {
					return current_focus;
				}

				static bool get_focus(const properties_t& focus) {
					return 
						current_focus.window_id == focus.window_id && 
						current_focus.shape == focus.shape && 
						current_focus.i == focus.i;
				}

				static void set_focus(const properties_t focus) {
					current_focus = focus;
				}

			}

			/*#define create_getset_focus(window_handle, shape, i)  \
				bool get_focus(uint32_t i) { \
					return 
						focus::current_focus.window_id == window_handle &&
						focus::current_focus.shape == shape &&
						focus::current_focus.i == 

				}*/

			namespace cursor_properties {
				inline fan::color color = fan::colors::white;
				// nanoseconds
				inline fan::time::nanoseconds blink_speed = 500000000;
				// i dont suggest changing for now, need to do srcdst - size / 2
				inline auto line_thickness = 1;
			}


			struct src_dst_t {
				fan::vec2 src;
				fan::vec2 dst;
			};

			template <typename T>
			struct text_input {

				/* used for hold precise floats on FED */
				/* bigger number means better precision */
				static constexpr uint32_t line_multiplier = 100;

				virtual void backspace_callback(uint32_t i) = 0;
				virtual void text_callback(uint32_t i) = 0;
				//virtual void on_input_callback() = 0;

				uint32_t text_callback_id = -1;
				uint32_t keys_callback_id = -1;

				text_input(T* base) :
					render_cursor(false), m_cursor(base->get_camera()), m_box(base), cursor_timer(cursor_properties::blink_speed)
				{
					m_cursor.disable_draw();

					fan_2d::graphics::rectangle::properties_t properties;
					properties.color = cursor_properties::color;
					m_cursor.push_back(properties);
					cursor_timer.start();

					keys_callback_id = base->get_camera()->m_window->add_keys_callback([&](fan::window*, uint16 key, fan::key_state state) {

						auto current_focus = focus::get_focus();

						if (current_focus.i != fan::uninitialized && current_focus.i >= m_input_allowed.size()) {
							assert(0);
						}

						if (
							current_focus.shape == nullptr ||
							state == fan::key_state::release || 
							current_focus.window_id != m_box->get_camera()->m_window->get_handle() ||  
							current_focus.i == focus::no_focus ||
							current_focus.shape != (void*)this || 
							!m_input_allowed[current_focus.i]
						) {
							return;
						}

						render_cursor = true;
						cursor_timer.restart();

						switch (key) {
							case fan::key_backspace: {
								if (m_box->get_text(current_focus.i).size() == 1) {
									backspace_callback(current_focus.i);
									FED_DeleteCharacterFromCursor(&m_wed[current_focus.i], cursor_reference[current_focus.i]);
									m_box->set_text(current_focus.i, " ");
									break;
								}

								FED_DeleteCharacterFromCursor(&m_wed[current_focus.i], cursor_reference[current_focus.i]);

								update_text(current_focus.i);

								backspace_callback(current_focus.i);

								update_cursor(current_focus.i);

								break;
							}
							case fan::key_delete: {
								FED_DeleteCharacterFromCursorRight(&m_wed[current_focus.i], cursor_reference[current_focus.i]);

								update_text(current_focus.i);

								backspace_callback(current_focus.i);

								break;
							}
							case fan::key_left: {
								FED_MoveCursorFreeStyleToLeft(&m_wed[current_focus.i], cursor_reference[current_focus.i]);

								update_cursor(current_focus.i);

								break;
							}
							case fan::key_right: {

								FED_MoveCursorFreeStyleToRight(&m_wed[current_focus.i], cursor_reference[current_focus.i]);

								update_cursor(current_focus.i);

								break;
							}
							case fan::key_up: {
								FED_MoveCursorFreeStyleToUp(&m_wed[current_focus.i], cursor_reference[current_focus.i]);

								update_cursor(current_focus.i);

								break;
							}
							case fan::key_down: {
								FED_MoveCursorFreeStyleToDown(&m_wed[current_focus.i], cursor_reference[current_focus.i]);

								update_cursor(current_focus.i);

								break;
							}
							case fan::key_home: {
								FED_MoveCursorFreeStyleToBeginOfLine(&m_wed[current_focus.i], cursor_reference[current_focus.i]);

								update_cursor(current_focus.i);

								break;
							}
							case fan::key_end: {
								FED_MoveCursorFreeStyleToEndOfLine(&m_wed[current_focus.i], cursor_reference[current_focus.i]);

								update_cursor(current_focus.i);

								break;
							}
							case fan::key_enter: {

								focus::set_focus(focus::no_focus);

								break;
							}
							case fan::key_tab: {

								auto current_focus = focus::get_focus();

								if (m_box->get_camera()->m_window->key_press(fan::key_shift)) {
									if (current_focus.i - 1 == ~0) {
										current_focus.i = m_box->size() - 1;
									}
									else {
										current_focus.i = (current_focus.i - 1) % m_box->size();
									}
								}
								else {
									current_focus.i = (current_focus.i + 1) % m_box->size();
								}

								focus::set_focus(current_focus);

								update_cursor(current_focus.i);

								break;
							}
							case fan::mouse_left: {

								/*fan::vec2 src = m_box->get_camera()->m_window->get_mouse_position() - m_box->get_text_starting_point(current_focus.i);
								fan::vec2 dst = src + fan::vec2(cursor_properties::line_thickness, fan_2d::graphics::gui::text_renderer::font_info.font['\n'].size.y *
									fan_2d::graphics::gui::text_renderer::convert_font_size(m_box->get_font_size(current_focus.i)));

								FED_LineReference_t FirstLineReference = _FED_LineList_GetNodeFirst(&m_wed[current_focus.i].LineList);
								FED_LineReference_t LineReference0, LineReference1;
								FED_CharacterReference_t CharacterReference0, CharacterReference1;
								FED_GetLineAndCharacter(&m_wed[current_focus.i], FirstLineReference, src.y, src.x * line_multiplier, &LineReference0, &CharacterReference0);
								FED_GetLineAndCharacter(&m_wed[current_focus.i], FirstLineReference, dst.y, dst.x * line_multiplier, &LineReference1, &CharacterReference1);
								FED_ConvertCursorToSelection(&m_wed[current_focus.i], cursor_reference[current_focus.i], LineReference0, CharacterReference0, LineReference1, CharacterReference1);

								update_cursor(current_focus.i);*/

								break;
							}
							case fan::key_v: {

								if (m_box->get_camera()->m_window->key_press(fan::key_control)) {
									
									//get_clipboard_text()
								}

								break;
							}
						}

					});

					text_callback_id = base->get_camera()->m_window->add_text_callback([&](fan::window*, uint32_t character) {

						auto current_focus = focus::get_focus();

						if (
							current_focus.window_id != m_box->get_camera()->m_window->get_handle() ||
							current_focus.shape != this ||
							current_focus.i == focus::no_focus || 
							!m_input_allowed[current_focus.i]
						) {
							return;
						}

						render_cursor = true;
						cursor_timer.restart();

						fan::utf8_string utf8;
						utf8.push_back(character);

						auto wc = utf8.to_utf16()[0];

						FED_AddCharacterToCursor(&m_wed[current_focus.i], cursor_reference[current_focus.i], character, fan_2d::graphics::gui::text_renderer::font.font[wc].metrics.size.x * fan_2d::graphics::gui::text_renderer::convert_font_size(m_box->get_font_size(current_focus.i)) * line_multiplier);

						fan::VEC_t text_vector;
						VEC_init(&text_vector, sizeof(uint8_t));

						fan::VEC_t cursor_vector;
						VEC_init(&cursor_vector, sizeof(FED_ExportedCursor_t));

						uint32_t line_index = 0;
						FED_LineReference_t line_reference = FED_GetLineReferenceByLineIndex(&m_wed[current_focus.i], 0);

						std::vector<FED_ExportedCursor_t*> exported_cursors;

						while (1) {
							bool is_endline;
							FED_ExportLine(&m_wed[current_focus.i], line_reference, &text_vector, &cursor_vector, &is_endline, utf8_data_callback);

							line_reference = _FED_LineList_GetNodeByReference(&m_wed[current_focus.i].LineList, line_reference)->NextNodeReference;

							for (uint_t i = 0; i < cursor_vector.Current; i++) {
								FED_ExportedCursor_t* exported_cursor = &((FED_ExportedCursor_t*)cursor_vector.ptr.data())[i];
								exported_cursor->y = line_index;
								exported_cursors.emplace_back(exported_cursor);
							}

							cursor_vector.Current = 0;

							if (line_reference == m_wed[current_focus.i].LineList.dst) {
								break;
							}

							{
								fan::VEC_handle(&text_vector);
								text_vector.ptr[text_vector.Current] = '\n';
								text_vector.Current++;
							}


							line_index++;
						}

						{
							fan::VEC_handle(&text_vector);
							text_vector.ptr[text_vector.Current] = 0;
							text_vector.Current++;
						}

						m_box->set_text(current_focus.i, text_vector.ptr.data());

						text_callback(current_focus.i);

						for (int i = 0; i < exported_cursors.size(); i++) {

							FED_ExportedCursor_t* exported_cursor = exported_cursors[i];

							auto src_dst = get_cursor_src_dst(current_focus.i, exported_cursor->x, exported_cursor->y);

							fan::vec2 size = { 0.5,
								fan_2d::graphics::gui::text_renderer::font.line_height *
								fan_2d::graphics::gui::text_renderer::convert_font_size(m_box->get_font_size(i)) / 2
							};

							m_cursor.set_position(0, src_dst.src - size / 2);
							m_cursor.set_size(0, size);

						}
					});

				}
				~text_input() {

					if (keys_callback_id != -1) {
						m_box->get_camera()->m_window->remove_keys_callback(keys_callback_id);
						keys_callback_id = -1;
					}

					if (text_callback_id != -1) {
						m_box->get_camera()->m_window->remove_text_callback(text_callback_id);
						text_callback_id = -1;
					}

					auto focus = focus::get_focus();

					if (focus.shape == this) {
						focus::properties_t fp;
						fp.i = fan::uninitialized;
						fp.shape = nullptr;
						fp.window_id = 0;
						focus::set_focus(fp);
					}

				}

				// must be called after T::push_back
				void push_back(uint32_t character_limit, f32_t line_width_limit, uint32_t line_limit) {
					m_wed.resize(m_wed.size() + 1);
					uint64_t offset = m_wed.size() - 1;
					FED_open(&m_wed[offset], fan_2d::graphics::gui::text_renderer::font.line_height, line_width_limit * line_multiplier, line_limit, character_limit);
					cursor_reference.emplace_back(FED_cursor_open(&m_wed[offset]));

					m_input_allowed.emplace_back(false);

					update_cursor(m_box->size() - 1);
				}

				// box i
				void set_line_width(uint32_t i, f32_t line_width) {
					FED_SetLineWidth(&m_wed[i], line_width * line_multiplier);
				}

				bool input_allowed(uint32_t i) const {
					return m_input_allowed[i];
				}

				void allow_input(uint32_t i, bool state) {
					m_input_allowed[i] = state;

					auto current_focus = focus::get_focus();

					if (current_focus.i == i && current_focus.window_id == m_box->get_camera()->m_window->get_handle() && current_focus.shape == this) {
						focus::set_focus(focus::no_focus);
					}
				}

				void enable_draw() {
					if (m_cursor.m_draw_index == -1 || m_cursor.m_camera->m_window->m_draw_queue[m_cursor.m_draw_index].first != this) {
						m_cursor.m_draw_index = m_cursor.m_camera->m_window->push_draw_call(this, [&] {
							this->draw();
						});
					}
					else {
						m_cursor.m_camera->m_window->edit_draw_call(m_cursor.m_draw_index, this, [&] {
							this->draw();
						});
					}
				}
				void disable_draw() {
					if (m_cursor.m_draw_index == -1) {
						return;
					}

					m_cursor.m_camera->m_window->erase_draw_call(m_cursor.m_draw_index);
					m_cursor.m_draw_index = -1;
				}

				void draw() {

					if (cursor_timer.finished()) {
						render_cursor = !render_cursor;
						cursor_timer.restart();
					}

					auto focus = focus::get_focus();

					if (render_cursor && focus == get_focus_info() && m_input_allowed[focus.i]) {
						m_cursor.enable_draw();
					}
					else {
						//m_cursor.disable_draw();
					}
				}

				focus::properties_t get_focus_info() const {
					return { m_box->get_camera()->m_window->get_handle(), (void*)this, 0 };
				}

				uint32_t get_focus() const {
					auto current_focus = focus::get_focus();

					if (current_focus == get_focus_info()) {
						return current_focus.i;
					}

					return focus::no_focus;
				}
				void set_focus(uintptr_t focus_id) {

					auto current_focus = get_focus_info();

					current_focus.i = focus_id;

					focus::set_focus(current_focus);

					update_cursor(focus_id);
				}

			/*	void erase(uint32_t i) {
					cursor_reference.erase(cursor_reference.begin() + i);

					m_wed.erase(m_wed.begin() + i);

					m_input_allowed.erase(m_input_allowed.begin() + i);
					m_cursor.erase(i);
				}

				void erase(uint32_t begin, uint32_t end) {
					cursor_reference.erase(cursor_reference.begin() + begin, cursor_reference.begin() + end);

					m_wed.erase(m_wed.begin() + begin, m_wed.begin() + end);

					m_input_allowed.erase(m_input_allowed.begin() + begin, m_input_allowed.begin() + end);
					m_cursor.erase(begin, end);
				}

				void clear() {
					cursor_reference.clear();

					m_wed.clear();

					m_input_allowed.clear();
					m_cursor.clear();
				}*/

			protected:

				bool render_cursor = false;

				fan::time::clock cursor_timer;

				std::vector<FED_CursorReference_t> cursor_reference;

				std::vector<FED_t> m_wed;

				std::vector<uint32_t> m_input_allowed;

				T* m_box;
				fan_2d::graphics::rectangle m_cursor;

				static void utf8_data_callback(fan::VEC_t* string, FED_Data_t data) {
					uint8_t size = fan::utf8_get_sizeof_character(data);
					for(uint8_t i = 0; i < size; i++){
						{
							fan::VEC_handle(string);
							string->ptr[string->Current] = data;
							string->Current++;
						}
						data >>= 8;
					}
				}

				// check focus before calling
				src_dst_t get_cursor_src_dst(uint32_t rtb_index, uint32_t x, uint32_t line_index) {
					return m_box->get_cursor(rtb_index, x, line_index);
				}

				void update_text(uint32_t i) {

					fan::VEC_t text_vector;
					VEC_init(&text_vector, sizeof(uint8_t));

					fan::VEC_t cursor_vector;
					VEC_init(&cursor_vector, sizeof(FED_ExportedCursor_t));

					bool is_endline;

					FED_LineReference_t line_reference = FED_GetLineReferenceByLineIndex(&m_wed[i], 0);

					while(1){
						bool is_endline;
						FED_ExportLine(&m_wed[i], line_reference, &text_vector, &cursor_vector, &is_endline, utf8_data_callback);

						line_reference = _FED_LineList_GetNodeByReference(&m_wed[i].LineList, line_reference)->NextNodeReference;

						cursor_vector.Current = 0;

						if(line_reference == m_wed[i].LineList.dst){
							break;
						}

						{
							fan::VEC_handle(&text_vector);
							text_vector.ptr[text_vector.Current] = '\n';
							text_vector.Current++;
						}

					}

					{
						fan::VEC_handle(&text_vector);
						text_vector.ptr[text_vector.Current] = 0;
						text_vector.Current++;
					}

					m_box->set_text(i, text_vector.ptr.data());

					text_callback(i);
				}

				void update_cursor(uint32_t i) {

					if (!(focus::get_focus() == get_focus_info())) {
						return;
					}

					fan::VEC_t text_vector;
					VEC_init(&text_vector, sizeof(uint8_t));

					fan::VEC_t cursor_vector;
					VEC_init(&cursor_vector, sizeof(FED_ExportedCursor_t));

					uint32_t line_index = 0; /* we dont know which line we are at so */
					FED_LineReference_t line_reference = FED_GetLineReferenceByLineIndex(&m_wed[i], 0);

					while(1){
						bool is_endline;
						FED_ExportLine(&m_wed[i], line_reference, &text_vector, &cursor_vector, &is_endline, utf8_data_callback);

						line_reference = _FED_LineList_GetNodeByReference(&m_wed[i].LineList, line_reference)->NextNodeReference;

						for(uint_t j = 0; j < cursor_vector.Current; j++){
							FED_ExportedCursor_t* exported_cursor = &((FED_ExportedCursor_t *)cursor_vector.ptr.data())[j];

							auto src_dst = get_cursor_src_dst(i, exported_cursor->x, line_index);

							fan::vec2 size = { 0.5, 
								fan_2d::graphics::gui::text_renderer::font.line_height *
								fan_2d::graphics::gui::text_renderer::convert_font_size(m_box->get_font_size(i)) / 2
							};

							m_cursor.set_position(0, src_dst.src - size / 2);
							m_cursor.set_size(0, size);
						}

						cursor_vector.Current = 0;

						if(line_reference == m_wed[i].LineList.dst){
							break;
						}

						line_index++;
					}
				}

				std::vector<src_dst_t> get_cursor_src_dsts(uint32_t i) {

					if (!(focus::get_focus() == get_focus_info())) {
						return {};
					}

					std::vector<src_dst_t> cursor_src_dsts;

					fan::VEC_t text_vector;
					VEC_init(&text_vector, sizeof(uint8_t));

					fan::VEC_t cursor_vector;
					VEC_init(&cursor_vector, sizeof(FED_ExportedCursor_t));

					uint32_t line_index = 0;
					FED_LineReference_t line_reference = FED_GetLineReferenceByLineIndex(&m_wed[i], 0);

					while(1){
						bool is_endline;
						FED_ExportLine(&m_wed[i], line_reference, &text_vector, &cursor_vector, &is_endline, utf8_data_callback);

						line_reference = _FED_LineList_GetNodeByReference(&m_wed[i].LineList, line_reference)->NextNodeReference;

						for(uint_t i = 0; i < cursor_vector.Current; i++){
							FED_ExportedCursor_t* exported_cursor = &((FED_ExportedCursor_t *)cursor_vector.ptr.data())[i];
							exported_cursor->y = line_index;
							cursor_src_dsts.emplace_back(get_cursor_src_dst(0, i, exported_cursor->x, exported_cursor->y));
						}

						cursor_vector.Current = 0;

						if(line_reference == m_wed[i].LineList.dst){
							break;
						}

						{
							fan::VEC_handle(&text_vector);
							text_vector.ptr[text_vector.Current] = '\n';
							text_vector.Current++;
						}

						line_index++;
					}

					return cursor_src_dsts;
				}

			};

			//struct editable_text_renderer : 
			//	public text_renderer,
			//	public base::mouse<editable_text_renderer>,
			//	public fan_2d::graphics::gui::text_input<editable_text_renderer>
			//{

			//	editable_text_renderer(fan::camera* camera) :
			//		editable_text_renderer::text_renderer(camera), 
			//		editable_text_renderer::text_input(this),
			//		editable_text_renderer::mouse(this)
			//	{

			//	}

			//	void push_back(const fan::utf16_string& text, f32_t font_size, fan::vec2 position, const fan::color& text_color) {
			//		editable_text_renderer::text_renderer::push_back(text, font_size, position, text_color);
			//		editable_text_renderer::text_input::push_back(-1, -1, -1);

			//		uint32_t index = this->size() - 1;

			//		for (int i = 0; i < text.size(); i++) {

			//			fan::utf8_string utf8;
			//			utf8.push_back(text[i]);

			//			auto wc = utf8.to_utf16()[0];

			//			FED_AddCharacterToCursor(&m_wed[index], cursor_reference[index], text[i], fan_2d::graphics::gui::text_renderer::font.font[wc].metrics.size.x * fan_2d::graphics::gui::text_renderer::convert_font_size(m_box->get_font_size(index)) * line_multiplier);
			//		}

			//		editable_text_renderer::text_input::update_text(index);
			//	}

			//	bool inside(uint32_t i) const {

			//		auto box_size = editable_text_renderer::text_renderer::get_text_size(
			//			editable_text_renderer::text_renderer::get_text(i),
			//			editable_text_renderer::text_renderer::get_font_size(i)
			//		);

			//		auto mouse_position = editable_text_renderer::text_renderer::m_camera->m_window->get_mouse_position();

			//	
			//		f32_t converted = fan_2d::graphics::gui::text_renderer::convert_font_size(this->get_font_size(i));
			//		auto line_height = fan_2d::graphics::gui::text_renderer::font.font['\n'].metrics.size.y * converted;

			//		return fan_2d::collision::rectangle::point_inside_no_rotation(
			//			mouse_position,
			//			editable_text_renderer::text_renderer::get_position(i),
			//			fan::vec2(box_size.x, line_height)
			//		);
			//	}

			//	uint32_t size() const {
			//		return editable_text_renderer::text_renderer::size();
			//	}

			//	// receives uint32_t box_i, uint32_t character number x, uint32_t character number y
			//	src_dst_t get_cursor(uint32_t i, uint32_t x, uint32_t y) {
			//		f32_t converted = fan_2d::graphics::gui::text_renderer::convert_font_size(this->get_font_size(i));
			//		auto line_height = fan_2d::graphics::gui::text_renderer::font.font['\n'].metrics.size.y * converted;

			//		fan::vec2 src, dst;

			//		src += text_renderer::get_position(i);

			//		//src.y -= line_height / 2;

			//		uint32_t offset = 0;

			//		auto str = this->get_text(i);

			//		for (int j = 0; j < y; j++) {
			//			while (str[offset++] != '\n') {
			//				if (offset >= str.size() - 1) {
			//					throw std::runtime_error("string didnt have endline");
			//				}
			//			}
			//		}

			//		for (int j = 0; j < x; j++) {
			//			wchar_t letter = str[j + offset];
			//			if (letter == '\n') {
			//				continue;
			//			}

			//			std::wstring wstr;

			//			wstr.push_back(letter);

			//			auto letter_info = fan_2d::graphics::gui::text_renderer::get_letter_info(fan::utf16_string(wstr).to_utf8().data(), this->get_font_size(i));

			//			if (j == x - 1) {
			//				src.x += letter_info.metrics.size.x + (letter_info.metrics.advance - letter_info.metrics.size.x) / 2 - 1;
			//			}
			//			else {
			//				src.x += letter_info.metrics.advance;
			//			}

			//		}

			//		src.y += line_height * y;


			//		dst = src + fan::vec2(0, line_height);

			//		dst = dst - src + fan::vec2(cursor_properties::line_thickness, 0);


			//		return { src, dst };
			//	}

			//	fan::camera* get_camera() {
			//		return text_renderer::m_camera;
			//	}

			//	void draw() {
			//		text_renderer::draw();
			//		text_input::draw();
			//	}

			//	void backspace_callback(uint32_t i) override {}
			//	void text_callback(uint32_t i) override {}

			//	void lib_add_on_input(uint32_t i, uint16_t key, fan::key_state state, fan_2d::graphics::gui::mouse_stage stage) {}

			//	void lib_add_on_mouse_event(uint32_t i, fan_2d::graphics::gui::mouse_stage stage) {}

			//	bool locked(uint32_t i) const { return false; }

			//};

			enum class text_position_e {
				left,
				middle
			};

			struct button_properties_t {

				button_properties_t() {}

				button_properties_t(
					const fan::utf16_string& text,
					const fan::vec2& position
				) : text(text), position(position) {}

				fan::utf16_string text = empty_string;

				fan::utf16_string place_holder;

				fan::vec2 position;
				fan::vec2 padding;

				f32_t font_size = fan_2d::graphics::gui::defaults::font_size;

			};

			struct rectangle_button_sized_properties {

				rectangle_button_sized_properties(fan::window* window) : theme(gui::themes::deep_blue(window)) {}

				fan::utf16_string text = empty_string;

				fan::utf16_string place_holder;

				fan::vec2 position;

				f32_t font_size = fan_2d::graphics::gui::defaults::font_size;

				text_position_e text_position = text_position_e::middle;

				fan_2d::graphics::gui::theme theme;

				fan::vec2 size;

				f32_t advance = 0;

			};

			// returns half size
			static fan::vec2 get_button_size(const fan::utf16_string text, f32_t font_size, uint32_t new_lines, const fan::vec2& padding)
			{

				f32_t h = text_renderer::font.line_height * fan_2d::graphics::gui::text_renderer::convert_font_size(font_size) * (new_lines + 1);

				return ((text.empty() ? fan::vec2(0, h) : fan::vec2(fan_2d::graphics::gui::text_renderer::get_text_size(text, font_size).x, h)) + padding) / 2; 
			}

			struct rectangle_text_box_sized :
				protected fan::class_duplicator<fan_2d::graphics::rectangle, 0>,
				protected fan::class_duplicator<fan_2d::graphics::rectangle, 1>,
				public graphics::gui::text_renderer
				{

				using properties_t = rectangle_button_sized_properties;

				using inner_rect_t = fan::class_duplicator<fan_2d::graphics::rectangle, 0>;
				using outer_rect_t = fan::class_duplicator<fan_2d::graphics::rectangle, 1>;

				rectangle_text_box_sized(fan::camera* camera);

				void push_back(const properties_t& properties);

				void set_position(uint32_t i, const fan::vec2& position);

				using fan_2d::graphics::rectangle::get_size;

				bool inside(uintptr_t i, const fan::vec2& position = fan::math::inf) const;

				void set_text(uint32_t i, const fan::utf16_string& text);

				fan::color get_text_color(uint32_t i) const;
				void set_text_color(uint32_t i, const fan::color& color);

				fan::vec2 get_position(uint32_t i) const;
				fan::vec2 get_size(uint32_t i) const;

				f32_t get_font_size(uint32_t i) const;

				fan::color get_color(uint32_t i) const;

				// receives uint32_t box_i, uint32_t character number x, uint32_t character number y
				src_dst_t get_cursor(uint32_t i, uint32_t x, uint32_t y);

				fan::vec2 get_text_starting_point(uint32_t i) const;

				properties_t get_property(uint32_t i) const;

				fan::camera* get_camera();

				uintptr_t size() const;

				void erase(uint32_t i);
				void erase(uint32_t begin, uint32_t end);

				void enable_draw();
				void disable_draw();

				// sets shape's draw order in window
				//void set_draw_order(uint32_t i);

				void clear();

				void update_theme(uint32_t i);

				using inner_rect_t::get_color;

				std::vector<fan_2d::graphics::gui::theme> theme;

			protected:

				void draw(uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized);

				void write_data();

				void edit_data(uint32_t i);

				void edit_data(uint32_t begin, uint32_t end);


				std::vector<properties_t> m_properties;

			};

			struct rectangle_text_box : 
				protected fan::class_duplicator<fan_2d::graphics::rectangle, 0>,
				protected fan::class_duplicator<fan_2d::graphics::rectangle, 1>,
				protected graphics::gui::text_renderer
			{

				using properties_t = button_properties_t;

				using inner_rect_t = fan::class_duplicator<fan_2d::graphics::rectangle, 0>;
				using outer_rect_t = fan::class_duplicator<fan_2d::graphics::rectangle, 1>;

				rectangle_text_box(fan::camera* camera, fan_2d::graphics::gui::theme theme);

				void push_back(const properties_t& properties);

				void draw(uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized);

				void set_position(uint32_t i, const fan::vec2& position);

				using fan_2d::graphics::rectangle::get_size;

				define_get_property_size;

				bool inside(uintptr_t i, const fan::vec2& position = fan::math::inf) const;

				fan::utf16_string get_text(uint32_t i) const;
				void set_text(uint32_t i, const fan::utf16_string& text);

				fan::color get_text_color(uint32_t i) const;
				void set_text_color(uint32_t i, const fan::color& color);

				fan::vec2 get_position(uint32_t i) const;
				fan::vec2 get_size(uint32_t i) const;

				fan::vec2 get_padding(uint32_t i) const;

				f32_t get_font_size(uint32_t i) const;

				properties_t get_property(uint32_t i) const;

				fan::color get_color(uint32_t i) const;

				// receives uint32_t box_i, uint32_t character number x, uint32_t character number y
				src_dst_t get_cursor(uint32_t i, uint32_t x, uint32_t y);

				fan::vec2 get_text_starting_point(uint32_t i) const;

				fan::camera* get_camera();

				uintptr_t size() const;

				void write_data();

				void edit_data(uint32_t i);

				void edit_data(uint32_t begin, uint32_t end);

				void erase(uint32_t i);
				void erase(uint32_t begin, uint32_t end);

				void clear();

				void set_theme(fan_2d::graphics::gui::theme theme_);

				void set_theme(uint32_t i, fan_2d::graphics::gui::theme theme_);

				void enable_draw();
				void disable_draw();

				using inner_rect_t::get_color;
				using inner_rect_t::set_color;

				using graphics::gui::text_renderer::font;

				fan_2d::graphics::gui::theme theme;

			protected:

				std::vector<button_properties_t> m_properties;

			};


			struct rectangle_text_button :
				public fan_2d::graphics::gui::rectangle_text_box,
				public fan_2d::graphics::gui::base::mouse<rectangle_text_button>,
				public fan_2d::graphics::gui::text_input<rectangle_text_button>
			{

				struct properties_t : public rectangle_text_box::properties_t{
					f32_t character_width = (f32_t)0xdfffffff / rectangle_text_button::text_input::line_multiplier;
					uint32_t character_limit = 99;
					uint32_t line_limit = 99;
				};

				using input_instance_t = fan_2d::graphics::gui::text_input<rectangle_text_button>;

				rectangle_text_button(fan::camera* camera, fan_2d::graphics::gui::theme theme);

				void push_back(const properties_t& properties);

				void set_place_holder(uint32_t i, const fan::utf16_string& place_holder);

				void draw();

				void backspace_callback(uint32_t i) override;
				void text_callback(uint32_t i) override;

				void erase(uint32_t i);
				void erase(uint32_t begin, uint32_t end);

				void clear();

				void set_locked(uint32_t i);

				bool locked(uint32_t i) const;

				void lib_add_on_input(uint32_t i, uint16_t key, fan::key_state state, fan_2d::graphics::gui::mouse_stage stage);

				void lib_add_on_mouse_event(uint32_t i, fan_2d::graphics::gui::mouse_stage stage);

			};

			enum class button_states_e {
				clickable = 1,
				locked = 2
			};

			struct circle_button : 
				protected fan_2d::graphics::circle,
				public fan_2d::graphics::gui::base::mouse<circle_button>
			{

				struct properties_t {

					properties_t(fan::window* window) : theme(window) {}

					fan::vec2 position;

					f32_t radius;

					fan_2d::graphics::gui::theme theme;

					button_states_e button_state = button_states_e::clickable;
				};

				circle_button(fan::camera* camera);
				~circle_button();

				using fan_2d::graphics::circle::inside;
				using fan_2d::graphics::circle::size;

				void push_back(properties_t properties);

				void erase(uint32_t i);
				void erase(uint32_t begin, uint32_t end);
				void clear();

				void set_locked(uint32_t i, bool flag);

				bool locked(uint32_t i) const;

				void enable_draw();
				void disable_draw();

				void update_theme(uint32_t i);

				virtual void lib_add_on_input(uint32_t i, uint16_t key, fan::key_state state, fan_2d::graphics::gui::mouse_stage stage);

				virtual void lib_add_on_mouse_event(uint32_t i, fan_2d::graphics::gui::mouse_stage stage);

				fan::camera* get_camera();

			protected:

				std::vector<fan_2d::graphics::gui::theme> m_theme;
				std::vector<uint32_t> m_reserved;

			};

			struct rectangle_text_button_sized :
				public fan_2d::graphics::gui::rectangle_text_box_sized,
				public fan_2d::graphics::gui::base::mouse<rectangle_text_button_sized>,
				public fan_2d::graphics::gui::text_input<rectangle_text_button_sized>
			{

				struct properties_t : public rectangle_text_box_sized::properties_t {

					using rectangle_text_box_sized::properties_t::rectangle_button_sized_properties;

					f32_t character_width = (f32_t)0xdfffffff / rectangle_text_button::text_input::line_multiplier;
					uint32_t character_limit = -1;
					uint32_t line_limit = -1;
					button_states_e button_state = button_states_e::clickable;
				};

				using input_instance_t = fan_2d::graphics::gui::text_input<rectangle_text_button_sized>;

				rectangle_text_button_sized(fan::camera* camera);
				~rectangle_text_button_sized();

				void push_back(properties_t properties);

				void set_place_holder(uint32_t i, const fan::utf16_string& place_holder);

				void backspace_callback(uint32_t i) override;
				void text_callback(uint32_t i) override;

				void erase(uint32_t i);
				void erase(uint32_t begin, uint32_t end);

				void clear();

				void set_locked(uint32_t i, bool flag);

				bool locked(uint32_t i) const;

				void enable_draw();
				void disable_draw();

				virtual void lib_add_on_input(uint32_t i, uint16_t key, fan::key_state state, fan_2d::graphics::gui::mouse_stage stage);

				virtual void lib_add_on_mouse_event(uint32_t i, fan_2d::graphics::gui::mouse_stage stage);

			protected:

				void draw(uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized);

				std::vector<uint32_t> m_reserved;

				rectangle_text_button_sized(bool custom, fan::camera* camera);

			};

			struct rectangle_selectable_button_sized : public rectangle_text_button_sized{

				rectangle_selectable_button_sized(fan::camera* camera);

				uint32_t get_selected(uint32_t i) const;
				void set_selected(uint32_t i);

				void add_on_select(std::function<void(uint32_t i)> function);

				void lib_add_on_input(uint32_t i, uint16_t key, fan::key_state state, fan_2d::graphics::gui::mouse_stage stage) override;

				void lib_add_on_mouse_event(uint32_t i, fan_2d::graphics::gui::mouse_stage stage) override;

			protected:

				std::vector<std::function<void(uint32_t)>> m_on_select;

				uint32_t m_selected = (uint32_t)fan::uninitialized;

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
				protected fan_2d::graphics::gui::text_renderer {

			public:

				using properties_t = button_properties_t;

				using sprite_t = fan::class_duplicator<fan_2d::graphics::sprite, 0>;

				sprite_text_box(fan::camera* camera, const std::string& path);

				void push_back(const properties_t& properties);

				void draw(uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized);

				define_get_property_size

					fan::camera* get_camera();

				uint64_t size() const;
				bool inside(uint32_t i, const fan::vec2& position = fan::math::inf) const;

				fan_2d::graphics::image_t image;

			protected:

				std::vector<properties_t> m_properties;
			};

			struct sprite_text_button :
				public fan_2d::graphics::gui::sprite_text_box,
				public base::mouse<sprite_text_button> {

				sprite_text_button(fan::camera* camera, const std::string& path);

				void lib_add_on_input(uint32_t i, uint16_t key, fan::key_state state, fan_2d::graphics::gui::mouse_stage stage);

				void lib_add_on_mouse_event(uint32_t i, fan_2d::graphics::gui::mouse_stage stage);

				bool locked(uint32_t i) const { return false; }

			};

			struct scrollbar : public fan_2d::graphics::rectangle, public fan_2d::graphics::gui::base::mouse<scrollbar> {

				enum class scroll_direction_e {
					horizontal,
					vertical,
				};

				struct properties_t {
					fan::vec2 position;
					fan::vec2 size;
					fan::color color;
					scroll_direction_e scroll_direction;
					f32_t current;
					f32_t length;
					uint32_t outline_thickness;
				};

				using on_scroll_t = std::function<void(uint32_t i, f32_t current)>;

				scrollbar(fan::camera* camera);

				void push_back(const properties_t& instance);

				void draw();

				void write_data();

				fan::camera* get_camera();

				void add_on_scroll(on_scroll_t function);

			protected:

				struct scroll_properties_t {
					scroll_direction_e scroll_direction;
					f32_t length;
					f32_t current;
					uint32_t outline_thickness;
				};

				std::vector<scroll_properties_t> m_properties;

				std::vector<on_scroll_t> m_on_scroll;

			};

			// takes slider value type as parameter
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
					camera->m_window->add_key_callback(fan::mouse_left, fan::key_state::press, [&] (fan::window*) {

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

							if (fan_2d::collision::rectangle::point_inside_no_rotation(mouse_position, box_position - box_size - fan::vec2(horizontal ? 0 : circle_diameter / 2 - 2, horizontal ? circle_diameter / 2 - 2 : 0), box_position + fan::vec2(horizontal ? box_size.x : circle_diameter, horizontal ? circle_diameter : box_size.y))) {

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

								f32_t length = box_size[!horizontal] * 2;

								T new_value = min + (((circle_position[!horizontal] - (box_position[!horizontal] - box_size[!horizontal])) / length) * (max - min));

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
							}
						}
					});

					camera->m_window->add_key_callback(fan::mouse_left, fan::key_state::release, [&] (fan::window*) {

						m_click_begin = fan::uninitialized;
						m_moving_id = fan::uninitialized;

					});

					camera->m_window->add_mouse_move_callback([&](fan::window*, const fan::vec2& position) {

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

						f32_t length = box_size[!horizontal] * 2;

						f32_t min = get_min_value(m_moving_id);
						f32_t max = get_max_value(m_moving_id);

						circle_position[!horizontal] = m_click_begin[!horizontal] + (position[!horizontal] - m_click_begin[!horizontal]);

						circle_position = circle_position.clamp(
							fan::vec2(box_position.x - box_size.x - circle::get_radius(m_moving_id), box_position.y - box_size.y - circle::get_radius(m_moving_id)),
							fan::vec2(box_position.x + box_size.x, box_position.y + box_size.y)
						);

						T new_value = min + (((circle_position[!horizontal] - (box_position[!horizontal] - box_size[!horizontal] - circle::get_radius(m_moving_id) )) / length) * (max - min));

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
						f32_t max = property.position.x + property.box_size.x * 2;

						f32_t new_x = (f32_t(property.current - property.min) / (property.max - property.min)) * (max - min);

						fan_2d::graphics::circle::properties_t cp;
						cp.position = property.position + fan::vec2(new_x - property.box_size.x, 0);
						cp.radius = property.button_radius;
						cp.color = property.button_color;

						fan_2d::graphics::circle::push_back(cp);
					}
					else {

						f32_t min = property.position.y;
						f32_t max = property.position.y + property.box_size.y * 2;

						f32_t new_y = (f32_t(property.current - property.min) / (property.max - property.min)) * (max - min);

						fan_2d::graphics::circle::properties_t cp;
						cp.position = property.position + fan::vec2(0, new_y - property.box_size.y);
						cp.radius = property.button_radius;
						cp.color = property.button_color;

						fan_2d::graphics::circle::push_back(cp);
					}
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

				void enable_draw() {
					fan_2d::graphics::rounded_rectangle::enable_draw();
					fan_2d::graphics::circle::enable_draw();
				}
				void disable_draw() {
					fan_2d::graphics::rounded_rectangle::disable_draw();
					fan_2d::graphics::circle::disable_draw();
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

						fan_2d::graphics::gui::text_renderer::set_text(i * 3 + 2, new_string);
					});

					circle_text_slider::circle_slider::on_click(false, [&] (uint32_t i) {

						auto new_string = fan::to_wstring(this->get_current_value(i));

						bool resize = text_renderer::get_text(i * 3 + 2).size() != new_string.size();

						fan_2d::graphics::gui::text_renderer::set_text(i * 3 + 2, new_string);

					});
				}

				void push_back(const typename circle_slider<T>::property_t& property) {
					circle_text_slider::circle_slider::push_back(property);

					const fan::vec2 left_text_size = fan_2d::graphics::gui::text_renderer::get_text_size(fan::to_wstring(property.min), property.font_size);

					fan::vec2 left_or_up;

					if (property.box_size.x > property.box_size.y) {
						left_or_up = property.position - fan::vec2(property.box_size.x, left_text_size.y / 2 + property.button_radius * text_gap_multiplier);
					}
					else {
						left_or_up = property.position + fan::vec2(left_text_size.x + property.button_radius * text_gap_multiplier, -property.box_size.y + left_text_size.y / 2 - property.button_radius);
					}

					fan_2d::graphics::gui::text_renderer::push_back(
						fan::to_wstring(property.min),
						property.font_size,
						left_or_up, 
						fan_2d::graphics::gui::defaults::text_color
					);

					const fan::vec2 right_text_size = fan_2d::graphics::gui::text_renderer::get_text_size(fan::to_wstring(property.max), property.font_size);

					fan::vec2 right_or_down;

					if (property.box_size.x > property.box_size.y) {
						right_or_down = property.position + fan::vec2(property.box_size.x - property.button_radius * text_gap_multiplier, -right_text_size.y / 2 - property.button_radius * text_gap_multiplier);
					}
					else {
						right_or_down = property.position + fan::vec2(left_text_size.x + property.button_radius * text_gap_multiplier, property.box_size.y - left_text_size.y / 2 + property.button_radius);
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
						middle = property.position + fan::vec2(0, -middle_text_size.y / 2 - property.button_radius * text_gap_multiplier);
					}
					else {
						middle = property.position + fan::vec2(middle_text_size.x + property.button_radius * text_gap_multiplier, 0);
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
				
				void enable_draw() {
					circle_text_slider::circle_slider::enable_draw();
					circle_text_slider::text_renderer::enable_draw();
				}
				void disable_draw() {
					circle_text_slider::circle_slider::disable_draw();
					circle_text_slider::text_renderer::disable_draw();
				}


				using circle_slider<T>::enable_draw;
				using circle_slider<T>::disable_draw;

			};

			class checkbox : 
				protected fan::class_duplicator<fan_2d::graphics::line, 99>,
				protected fan::class_duplicator<fan_2d::graphics::rectangle, 99>, 
				protected fan::class_duplicator<fan_2d::graphics::gui::text_renderer, 99>,
				public base::mouse<checkbox> {

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

					f32_t box_size_multiplier = 1;

					bool checked = false;
				};

				checkbox(fan::camera* camera, fan_2d::graphics::gui::theme theme);

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

				void lib_add_on_input(uint32_t i, uint16_t key, fan::key_state state, fan_2d::graphics::gui::mouse_stage stage);

				void lib_add_on_mouse_event(uint32_t i, fan_2d::graphics::gui::mouse_stage stage);

				bool locked(uint32_t i) const { return false; }

				void enable_draw();
				void disable_draw();

			protected:

				std::function<void(uint32_t)> m_on_check;
				std::function<void(uint32_t)> m_on_uncheck;

				fan_2d::graphics::gui::theme m_theme;

				std::deque<bool> m_visible;
				std::deque<checkbox::properties_t> m_properties;

			};

			struct dropdown_menu : protected rectangle_text_button_sized {

				struct properties_t {
					fan::utf16_string text = empty_string;

					fan::vec2 position;

					f32_t font_size = fan_2d::graphics::gui::defaults::font_size;

					text_position_e text_position = text_position_e::left;

					fan::vec2 size;

					f32_t advance = 0;

					std::vector<fan::utf16_string> dropdown_texts;
				};

				dropdown_menu(fan::camera* camera, const fan_2d::graphics::gui::theme& theme);

				void push_back(const properties_t& property);

				void draw();

				using rectangle_text_button_sized::get_camera;

			protected:

				std::vector<src_dst_t> m_hitboxes;

				uint32_t m_hovered = -1;

				std::deque<uint32_t> m_amount_per_menu;

			};

			struct progress_bar : protected fan_2d::graphics::rectangle {

			protected:

				using rectangle_t = fan_2d::graphics::rectangle;

				struct progress_bar_t {
					f32_t inner_size_multipliers;
					f32_t progress;
					bool progress_x = false;
				};

				std::vector<progress_bar_t> m_progress_bar_properties;

			public:

				progress_bar(fan::camera* camera);

				struct properties_t {

					fan::vec2 position;
					fan::vec2 size;

					fan::color back_color;
					fan::color front_color;

					f32_t inner_size_multiplier = 0.8;

					f32_t progress = 0;

					bool progress_x = true;

				};

				void push_back(const properties_t& properties);

				void clear();

				f32_t get_progress(uint32_t i) const;
				void set_progress(uint32_t i, f32_t progress);

				fan::vec2 get_position(uint32_t i) const;
				void set_position(uint32_t i, const fan::vec2& position);

				using fan_2d::graphics::rectangle::enable_draw;
				using fan_2d::graphics::rectangle::disable_draw;

			};

		}

	}

}