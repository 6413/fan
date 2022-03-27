#pragma once

#include <fan/types/types.h>

#define FED_set_debug_InvalidLineAccess 1
#include <fan/fed/FED.h>

#include <fan/graphics/shared_graphics.h>

#include <fan/graphics/graphics.h>

#include <fan/font.h>

#include <fan/types/utf_string.h>

#include <fan/physics/collision/rectangle.h>
#include <fan/physics/collision/circle.h>

#include <fan/graphics/opengl/2D/objects/text_renderer.h>

namespace fan_2d {

	namespace opengl {

		namespace gui {

			static fan::utf16_string get_empty_string() {
				fan::utf16_string str;

				str.resize(1);

				return str;
			}

			inline fan::utf16_string empty_string = get_empty_string();

			static fan::vec2 get_resize_movement_offset(fan::window_t* window)
			{
				return fan::cast<f32_t>(window->get_size() - window->get_previous_size());
			}

			static void add_resize_callback(fan::window_t* window, fan::vec2& position) {
				window->add_resize_callback([&](fan::window_t* window, const fan::vec2i&) {
					position += fan_2d::opengl::gui::get_resize_movement_offset(window);
				});
			}

			/*	struct rectangle : public fan_2d::opengl::rectangle {

					rectangle(fan::camera* camera)
						: fan_2d::opengl::rectangle(camera)
					{
						this->m_camera->m_window->add_resize_callback([&](const fan::window_t* window, const fan::vec2i&) {

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

				};*/

			struct text_renderer0 : public text_renderer_t {

				typedef void(*index_change_cb_t)(void*, uint64_t);

				text_renderer0() = default;

				void open(fan::opengl::context_t* context, void* gp, index_change_cb_t index_change_cb) {
					m_index_change_cb = index_change_cb;
					m_entity_ids.open();
					m_gp = gp;
					text_renderer_t::open(context);
				}

				void close(fan::opengl::context_t* context) {
					m_entity_ids.close();
				}

				struct properties_t : public text_renderer_t::properties_t {
					uint32_t entity_id;
				};

				void push_back(fan::opengl::context_t* context, properties_t properties) {
					m_entity_ids.emplace_back(properties.entity_id);
					text_renderer_t::push_back(context, properties);
				}

				void erase(fan::opengl::context_t* context, uintptr_t i) {
					for (uint32_t start = i + 1; start < text_renderer_t::size(context); start++) {
						m_index_change_cb(m_gp, m_entity_ids[start]);
					}
					text_renderer_t::erase(context, i);
					m_entity_ids.erase(i);
				}

				void erase(uintptr_t begin, uintptr_t end) = delete;

			protected:

				index_change_cb_t m_index_change_cb;

				void* m_gp;
				fan::hector_t<uint64_t> m_entity_ids;

			};

			namespace base {

				// requires to have functions: 
				// get_camera() which returns camera, 
				// size() returns amount of objects, 
				// inside() if mouse inside
				// requires push_back to be called in every shape push_back
				// requires to have either add_on_input or add_on_input and add_on_mouse_event
				template <typename T>
				struct button_event_t {

					button_event_t() = default;

					void open(fan::window_t* window, fan::opengl::context_t* context) {
						m_old_mouse_stage = fan::uninitialized;
						m_do_we_hold_button = 0;
						m_focused_button_id = fan::uninitialized;
						add_keys_callback_id = fan::uninitialized;
						add_mouse_move_callback_id = fan::uninitialized;
						on_input_function = new std::remove_pointer_t<decltype(on_input_function)>;
						on_mouse_event_function = new std::remove_pointer_t<decltype(on_mouse_event_function)>;

						add_mouse_move_callback_id = window->add_mouse_move_callback([this, context, object = OFFSETLESS(this, T, m_button_event)](fan::window_t* w, const fan::vec2&) {
							if (m_do_we_hold_button == 1) {
								return;
							}
							if (m_focused_button_id != fan::uninitialized) {

								if (m_focused_button_id >= object->size(context)) {
									m_focused_button_id = fan::uninitialized;
								}
								else if (object->inside(context, m_focused_button_id, fan::vec2(context->camera.get_position()) + w->get_mouse_position()) || object->locked(w, context, m_focused_button_id)) {
									return;
								}
							}

							for (int i = object->size(context); i--; ) {
								if (object->inside(context, i, fan::vec2(context->camera.get_position()) + w->get_mouse_position()) && !object->locked(w, context, i)) {
									if (m_focused_button_id != fan::uninitialized) {
										object->lib_add_on_mouse_move(w, context, m_focused_button_id, mouse_stage::outside);
										if ((*on_mouse_event_function)) {
											(*on_mouse_event_function)(w, context, m_focused_button_id, mouse_stage::outside);
										}
									}
									m_focused_button_id = i;
									object->lib_add_on_mouse_move(w, context, m_focused_button_id, mouse_stage::inside);
									if ((*on_mouse_event_function)) {
										(*on_mouse_event_function)(w, context, m_focused_button_id, mouse_stage::inside);
									}
									return;
								}
							}
							if (m_focused_button_id != fan::uninitialized) {
								object->lib_add_on_mouse_move(w, context, m_focused_button_id, mouse_stage::outside);
								if (*on_mouse_event_function) {
									(*on_mouse_event_function)(w, context, m_focused_button_id, mouse_stage::outside);
								}
								m_focused_button_id = fan::uninitialized;
							}
						});

						add_keys_callback_id = window->add_keys_callback([this, context, object = OFFSETLESS(this, T, m_button_event)](fan::window_t* w, uint16_t key, fan::key_state state) {

							if (m_focused_button_id >= object->size(context)) {
								m_focused_button_id = fan::uninitialized;
							}

							if (m_do_we_hold_button == 0) {
								if (state == fan::key_state::press) {
									if (m_focused_button_id != fan::uninitialized) {
										m_do_we_hold_button = 1;
										object->lib_add_on_input(w, context, m_focused_button_id, key, fan::key_state::press, mouse_stage::inside);
										if ((*on_input_function)) {
											(*on_input_function)(w, context, m_focused_button_id, key, fan::key_state::press, mouse_stage::inside);
										}
									}
									else {
										for (int i = object->size(context); i--; ) {
											object->lib_add_on_input(w, context, i, key, state, mouse_stage::outside);
											if ((*on_input_function)) {
												(*on_input_function)(w, context, i, key, state, mouse_stage::outside);
											}
										}
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
									if (m_focused_button_id >= object->size(context)) {
										m_focused_button_id = fan::uninitialized;
									}
									else if (object->inside(context, m_focused_button_id, fan::vec2(context->camera.get_position()) + w->get_mouse_position()) && !object->locked(w, context, m_focused_button_id)) {
										object->lib_add_on_input(w, context, m_focused_button_id, key, fan::key_state::release, mouse_stage::inside);
										if ((*on_input_function)) {
											pointer_remove_flag = 1;
											(*on_input_function)(w, context, m_focused_button_id, key, fan::key_state::release, mouse_stage::inside);
											if (pointer_remove_flag == 0) {
												return;
												//rtb is deleted
											}
										}
									}
									else {
										object->lib_add_on_input(w, context, m_focused_button_id, key, fan::key_state::release, mouse_stage::outside);

										for (int i = object->size(context); i--; ) {
											if (object->inside(context, i, fan::vec2(context->camera.get_position()) + w->get_mouse_position()) && !object->locked(w, context, i)) {
												object->lib_add_on_input(w, context, i, key, fan::key_state::release, mouse_stage::inside_drag);
												m_focused_button_id = i;
												break;
											}
										}

										if ((*on_input_function)) {
											pointer_remove_flag = 1;
											(*on_input_function)(w, context, m_focused_button_id, key, fan::key_state::release, mouse_stage::outside);
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

					void close(fan::window_t* window, fan::opengl::context_t* context) {
						delete on_input_function;
						delete on_mouse_event_function;
						if (add_keys_callback_id != -1) {
							window->remove_keys_callback(add_keys_callback_id);
							add_keys_callback_id = -1;
						}
						if (add_mouse_move_callback_id != -1) {
							window->remove_mouse_move_callback(add_mouse_move_callback_id);
							add_mouse_move_callback_id = -1;
						}
					}

				public:

					// uint32_t index, fan::key_state state, mouse_stage mouse_stage
					void set_on_input(std::function<void(fan::window_t* window, fan::opengl::context_t* context, uint32_t index, uint16_t key, fan::key_state key_state, mouse_stage mouse_stage)> function) {
						*on_input_function = function;
					}

					// uint32_t index, bool inside
					void set_on_mouse_event(std::function<void(fan::window_t* window, fan::opengl::context_t* context, uint32_t index, mouse_stage mouse_stage)> function) {
						*on_mouse_event_function = function;
					}

					uint32_t holding_button() const {
						return m_focused_button_id;
					}

					std::function<void(fan::window_t* window, fan::opengl::context_t* context, uint32_t index, uint16_t key, fan::key_state key_state, mouse_stage mouse_stage)>* on_input_function;

					std::function<void(fan::window_t* window, fan::opengl::context_t* context, uint32_t index, mouse_stage mouse_stage)>* on_mouse_event_function;

					inline static thread_local bool pointer_remove_flag;

					uint8_t m_old_mouse_stage;
					bool m_do_we_hold_button;
					uint32_t m_focused_button_id;
					uint32_t add_keys_callback_id;
					uint32_t add_mouse_move_callback_id;

				};

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

			namespace focus {

				constexpr uint32_t no_focus = -1;

				struct properties_t {

					properties_t() {}
					properties_t(uint32_t x) : window_id((fan::window_handle_t)x), shape(0), i(x) {}
					properties_t(fan::window_handle_t window_id, void* shape, uint32_t i) :
						window_id(window_id), shape(shape), i(i) {}

					// focus window
					fan::window_handle_t window_id;
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
				inline fan::time::nanoseconds blink_speed = 100000000;
				inline f32_t line_thickness = 1;
			}
			//
			//
			struct src_dst_t {
				fan::vec2 src = 0;
				fan::vec2 dst = 0;
			};

			template <typename T>
			struct key_event_t {

				/* used for hold precise floats on FED */
				/* bigger number means better precision */
				static constexpr uint32_t line_multiplier = 100;

				//virtual void on_input_callback() = 0;

				uint32_t text_callback_id;
				uint32_t keys_callback_id;
				uint32_t mouse_move_callback_id;

				fan::vec2 click_begin;

				void open(fan::window_t* window, fan::opengl::context_t* context) {
					m_store.open();
					m_draw_index2 = fan::uninitialized;
					render_cursor = false;
					m_cursor.open(context);
					cursor_timer = fan::time::clock(cursor_properties::blink_speed);
					mouse_move_callback_id = fan::uninitialized;
					text_callback_id = fan::uninitialized;
					keys_callback_id = fan::uninitialized;
					mouse_move_callback_id = fan::uninitialized;

					cursor_timer.start();

					keys_callback_id = window->add_keys_callback([this, context, object = OFFSETLESS(this, T, m_key_event)](fan::window_t* w, uint16 key, fan::key_state state) {

						auto current_focus = focus::get_focus();

						if (
							current_focus.shape == nullptr ||
							current_focus.window_id != w->get_handle() ||
							current_focus.i == focus::no_focus ||
							current_focus.shape != (void*)this ||
							!m_store[current_focus.i].m_input_allowed
							) {
							return;
						}

						switch (state) {
						case fan::key_state::press: {

							if (mouse_move_callback_id != -1) {
								break;
							}

							click_begin = fan::cast<f32_t>(w->get_mouse_position()) - object->get_text_starting_point(context, current_focus.i);

							mouse_move_callback_id = w->add_mouse_move_callback([this, context, object](fan::window_t* w, const fan::vec2& p) {
								if (!w->key_press(fan::mouse_left)) {
									return;
								}

								auto current_focus = focus::get_focus();

								if (current_focus.shape == nullptr ||
									current_focus.window_id != w->get_handle() ||
									current_focus.i == focus::no_focus ||
									current_focus.shape != (void*)this ||
									!m_store[current_focus.i].m_input_allowed)
								{
									return;
								}

								fan::vec2 src = click_begin;
								// dst release
								fan::vec2 dst = fan::cast<f32_t>(w->get_mouse_position()) - object->get_text_starting_point(context, current_focus.i);
								dst.x = fan::clamp(dst.x, (f32_t)0, dst.x);

								FED_LineReference_t FirstLineReference = _FED_LineList_GetNodeFirst(&m_store[current_focus.i].m_wed.LineList);
								FED_LineReference_t LineReference0, LineReference1;
								FED_CharacterReference_t CharacterReference0, CharacterReference1;
								FED_GetLineAndCharacter(&m_store[current_focus.i].m_wed, FirstLineReference, src.y, src.x * line_multiplier, &LineReference0, &CharacterReference0);
								FED_GetLineAndCharacter(&m_store[current_focus.i].m_wed, FirstLineReference, dst.y, dst.x * line_multiplier, &LineReference1, &CharacterReference1);
								FED_ConvertCursorToSelection(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference, LineReference0, CharacterReference0, LineReference1, CharacterReference1);

								update_cursor(w, context, current_focus.i);

								render_cursor = true;
								cursor_timer.restart();
								m_cursor.enable_draw(context);
							});

							break;
						}
						case fan::key_state::release: {

							if (mouse_move_callback_id != -1) {
								w->remove_mouse_move_callback(mouse_move_callback_id);
								mouse_move_callback_id = -1;
							}

							return;
							break;
						}
						}

						render_cursor = true;
						cursor_timer.restart();
						m_cursor.enable_draw(context);

						switch (key) {
						case fan::key_backspace: {
							if (object->get_text(context, current_focus.i).size() == 1) {
								object->backspace_callback(w, context, current_focus.i);
								FED_DeleteCharacterFromCursor(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference);
								object->set_text(context, current_focus.i, L" ");

								break;
							}

							FED_DeleteCharacterFromCursor(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference);

							update_text(w, context, current_focus.i);

							object->backspace_callback(w, context, current_focus.i);

							update_cursor(w, context, current_focus.i);

							break;
						}
						case fan::key_delete: {
							FED_DeleteCharacterFromCursorRight(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference);

							update_text(w, context, current_focus.i);

							object->backspace_callback(w, context, current_focus.i);

							break;
						}
						case fan::key_left: {
							FED_MoveCursorFreeStyleToLeft(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference);

							update_cursor(w, context, current_focus.i);

							break;
						}
						case fan::key_right: {

							FED_MoveCursorFreeStyleToRight(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference);

							update_cursor(w, context, current_focus.i);

							break;
						}
						case fan::key_up: {
							FED_MoveCursorFreeStyleToUp(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference);

							update_cursor(w, context, current_focus.i);

							break;
						}
						case fan::key_down: {
							FED_MoveCursorFreeStyleToDown(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference);

							update_cursor(w, context, current_focus.i);

							break;
						}
						case fan::key_home: {
							FED_MoveCursorFreeStyleToBeginOfLine(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference);

							update_cursor(w, context, current_focus.i);

							break;
						}
						case fan::key_end: {
							FED_MoveCursorFreeStyleToEndOfLine(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference);

							update_cursor(w, context, current_focus.i);

							break;
						}
						case fan::key_enter: {

							focus::set_focus(focus::no_focus);

							break;
						}
						case fan::key_tab: {

							auto current_focus = focus::get_focus();

							if (w->key_press(fan::key_shift)) {
								if (current_focus.i - 1 == ~0) {
									current_focus.i = object->size(context) - 1;
								}
								else {
									current_focus.i = (current_focus.i - 1) % object->size(context);
								}
							}
							else {
								current_focus.i = (current_focus.i + 1) % object->size(context);
							}

							focus::set_focus(current_focus);

							update_cursor(w, context, current_focus.i);

							break;
						}
						case fan::mouse_left: {

							// src press
							fan::vec2 src = fan::cast<f32_t>(w->get_mouse_position()) - object->get_text_starting_point(context, current_focus.i);
							// dst release
							src.x = fan::clamp(src.x, (f32_t)0, src.x);
							fan::vec2 dst = src;

							FED_LineReference_t FirstLineReference = _FED_LineList_GetNodeFirst(&m_store[current_focus.i].m_wed.LineList);
							FED_LineReference_t LineReference0, LineReference1;
							FED_CharacterReference_t CharacterReference0, CharacterReference1;
							FED_GetLineAndCharacter(&m_store[current_focus.i].m_wed, FirstLineReference, src.y, src.x * line_multiplier, &LineReference0, &CharacterReference0);
							FED_GetLineAndCharacter(&m_store[current_focus.i].m_wed, FirstLineReference, dst.y, dst.x * line_multiplier, &LineReference1, &CharacterReference1);
							FED_ConvertCursorToSelection(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference, LineReference0, CharacterReference0, LineReference1, CharacterReference1);

							update_cursor(w, context, current_focus.i);

							break;
						}
						case fan::key_v: {

							if (w->key_press(fan::key_control)) {

								auto pasted_text = fan::io::get_clipboard_text(w->get_handle());

								for (int i = 0; i < pasted_text.size(); i++) {
									add_character(context, &m_store[current_focus.i].m_wed, &m_store[current_focus.i].cursor_reference, pasted_text[i], object->get_font_size(context, current_focus.i));
								}

								update_text(w, context, current_focus.i);
							}

							break;
						}
						}

					});

					text_callback_id = window->add_text_callback([this, context, object = OFFSETLESS(this, T, m_key_event)](fan::window_t* w, uint32_t character) {

						auto current_focus = focus::get_focus();

						if (
							current_focus.window_id != w->get_handle() ||
							current_focus.shape != this ||
							current_focus.i == focus::no_focus ||
							!m_store[current_focus.i].m_input_allowed
							) {
							return;
						}

						render_cursor = true;
						cursor_timer.restart();

						m_cursor.enable_draw(context);

						fan::utf8_string utf8;
						utf8.push_back(character);

						auto wc = utf8.to_utf16()[0];

						f32_t font_size = object->get_font_size(context, current_focus.i);
						add_character(context, &m_store[current_focus.i].m_wed, &m_store[current_focus.i].cursor_reference, character, font_size);

						fan::vector_t text_vector;
						vector_init(&text_vector, sizeof(uint8_t));

						fan::vector_t cursor_vector;
						vector_init(&cursor_vector, sizeof(FED_ExportedCursor_t));

						uint32_t line_index = 0;
						FED_LineReference_t line_reference = FED_GetLineReferenceByLineIndex(&m_store[current_focus.i].m_wed, 0);

						std::vector<FED_ExportedCursor_t*> exported_cursors;

						while (1) {
							bool is_endline;
							FED_ExportLine(&m_store[current_focus.i].m_wed, line_reference, &text_vector, &cursor_vector, &is_endline, utf8_data_callback);

							line_reference = _FED_LineList_GetNodeByReference(&m_store[current_focus.i].m_wed.LineList, line_reference)->NextNodeReference;

							for (uint_t i = 0; i < cursor_vector.Current; i++) {
								FED_ExportedCursor_t* exported_cursor = &((FED_ExportedCursor_t*)cursor_vector.ptr.data())[i];
								exported_cursor->y = line_index;
								exported_cursors.emplace_back(exported_cursor);
							}

							cursor_vector.Current = 0;

							if (line_reference == m_store[current_focus.i].m_wed.LineList.dst) {
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

						object->set_text(context, current_focus.i, text_vector.ptr.data());

						object->text_callback(w, context, current_focus.i);

						for (int i = 0; i < exported_cursors.size(); i++) {

							FED_ExportedCursor_t* exported_cursor = exported_cursors[i];

							auto src_dst = get_cursor_src_dst(w, context, current_focus.i, exported_cursor->x, exported_cursor->y);

							m_cursor.set_position(context, 0, src_dst.src);
							m_cursor.set_size(context, 0, src_dst.dst);
							update_cursor(w, context, current_focus.i);

						}
					});
				}

				void close(fan::window_t* window, fan::opengl::context_t* context) {
					m_store.close();

					if (keys_callback_id != -1) {
						window->remove_keys_callback(keys_callback_id);
						keys_callback_id = -1;
					}

					if (text_callback_id != -1) {
						window->remove_text_callback(text_callback_id);
						text_callback_id = -1;
					}

					if (m_draw_index2 != -1) {
						context->m_draw_queue.erase(m_draw_index2);
						m_draw_index2 = -1;
					}

					if (mouse_move_callback_id != -1) {
						window->remove_mouse_move_callback(mouse_move_callback_id);
						mouse_move_callback_id = -1;
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

				void add_character(fan::opengl::context_t* context, FED_t* wed, FED_CursorReference_t* cursor_reference, FED_Data_t character, f32_t font_size) {

					auto found = fan_2d::opengl::gui::text_renderer_t::font.characters.find(character);

					if (found == fan_2d::opengl::gui::text_renderer_t::font.characters.end()) {
						return;
					}

					auto letter = fan_2d::opengl::gui::text_renderer_t::get_letter_info(context, character, font_size);

					FED_AddCharacterToCursor(
						wed,
						*cursor_reference,
						character,
						letter.metrics.advance * line_multiplier
					);
				}

				// must be called after T::push_back
				void push_back(fan::window_t* window, fan::opengl::context_t* context, uint32_t character_limit, f32_t line_width_limit, uint32_t line_limit) {
					auto object = (T*)OFFSETLESS(this, T, m_key_event);

					m_store.resize(m_store.size() + 1);
					uint64_t offset = m_store.size() - 1;
					FED_open(&m_store[offset].m_wed, fan_2d::opengl::gui::text_renderer_t::font.line_height, line_width_limit * line_multiplier, line_limit, character_limit);
					m_store[offset].cursor_reference = FED_cursor_open(&m_store[offset].m_wed);
					m_store[offset].m_input_allowed = false;

					auto str = object->get_text(context, offset);

					for (int i = 0; i < str.size(); i++) {
						add_character(context, &m_store[offset].m_wed, &m_store[offset].cursor_reference, str[i], object->get_font_size(context, offset));
					}

					update_cursor(window, context, object->size(context) - 1);
				}

				// box i
				void set_line_width(fan::window_t* window, uint32_t i, f32_t line_width) {
					FED_SetLineWidth(&m_store[i].m_wed, line_width * line_multiplier);
				}

				bool input_allowed(fan::window_t* window, uint32_t i) const {
					return m_store[i].m_input_allowed;
				}

				void allow_input(fan::window_t* window, uint32_t i, bool state) {
					m_store[i].m_input_allowed = state;

					auto current_focus = focus::get_focus();

					if (current_focus.i == i && current_focus.window_id == window->get_handle() && current_focus.shape == this) {
						focus::set_focus(focus::no_focus);
					}
				}

				uint32_t m_draw_index2;

				void enable_draw(fan::window_t* window, fan::opengl::context_t* context) {
					/*if (m_draw_index2 == -1 || context->m_draw_queue[m_draw_index2].first != this) {
						m_draw_index2 = window->push_draw_call(this, [&] {
							this->draw(window, context);
						});
					}
					else {
						window->edit_draw_call(m_draw_index2, this, [&] {
							this->draw(window, context);
						});
					}*/
				}
				void disable_draw(fan::window_t* window, fan::opengl::context_t* context) {
					if (m_draw_index2 == -1) {
						return;
					}

					context->m_draw_queue.erase(m_draw_index2);
					m_draw_index2 = -1;
				}

				void draw(fan::window_t* window, fan::opengl::context_t* context) {

					if (cursor_timer.finished()) {
						render_cursor = !render_cursor;
						cursor_timer.restart();

						auto focus_ = focus::get_focus();

						if (render_cursor && focus_ == get_focus_info(window) && m_store[focus_.i].m_input_allowed) {
							m_cursor.enable_draw(context);
						}
						else {
							m_cursor.disable_draw(context);
						}
					}

				}

				focus::properties_t get_focus_info(fan::window_t* window) const {
					return { window->get_handle(), (void*)this, 0 };
				}

				uint32_t get_focus() const {
					auto current_focus = focus::get_focus();

					if (current_focus == get_focus_info()) {
						return current_focus.i;
					}

					return focus::no_focus;
				}
				void set_focus(fan::window_t* window, fan::opengl::context_t* context, uintptr_t focus_id) {

					auto current_focus = get_focus_info(window);

					current_focus.i = focus_id;

					focus::set_focus(current_focus);

					update_cursor(window, context, focus_id);
				}

				void erase(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) {

					FED_cursor_close(&m_store[i].m_wed, m_store[i].cursor_reference);

					FED_close(&m_store[i].m_wed);

					m_store.erase(i);

					m_cursor.clear(context);
				}

				void erase(fan::window_t* window, fan::opengl::context_t* context, uint32_t begin, uint32_t end) {

					for (int j = begin; j < end - begin; j++) {
						FED_cursor_close(&m_store[j].m_wed, m_store[j].cursor_reference);
					}

					for (int j = begin; j < end - begin; j++) {
						FED_close(&m_store[j].m_wed);
					}

					m_store.erase(begin, end);

					m_cursor.clear(context);
				}

				void clear(fan::window_t* window, fan::opengl::context_t* context) {
					m_store.clear();
					m_cursor.clear(context);
				}

				static void utf8_data_callback(fan::vector_t* string, FED_Data_t data) {
					uint8_t size = fan::utf8_get_sizeof_character(data);
					for (uint8_t i = 0; i < size; i++) {
						{
							fan::VEC_handle(string);
							string->ptr[string->Current] = data;
							string->Current++;
						}
						data >>= 8;
					}
				}

				// check focus before calling
				src_dst_t get_cursor_src_dst(fan::window_t* window, fan::opengl::context_t* context, uint32_t rtb_index, uint32_t x, uint32_t line_index) {
					auto object = OFFSETLESS(this, T, m_key_event);
					return object->get_cursor(context, rtb_index, x, line_index);
				}

				void update_text(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) {

					auto object = (T*)OFFSETLESS(this, T, m_key_event);

					fan::vector_t text_vector;
					vector_init(&text_vector, sizeof(uint8_t));

					fan::vector_t cursor_vector;
					vector_init(&cursor_vector, sizeof(FED_ExportedCursor_t));

					bool is_endline;

					FED_LineReference_t line_reference = FED_GetLineReferenceByLineIndex(&m_store[i].m_wed, 0);

					while (1) {
						bool is_endline;
						FED_ExportLine(&m_store[i].m_wed, line_reference, &text_vector, &cursor_vector, &is_endline, utf8_data_callback);

						line_reference = _FED_LineList_GetNodeByReference(&m_store[i].m_wed.LineList, line_reference)->NextNodeReference;

						cursor_vector.Current = 0;

						if (line_reference == m_store[i].m_wed.LineList.dst) {
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

					object->set_text(context, i, text_vector.ptr.data());

					object->text_callback(window, context, i);
				}

				void update_cursor(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) {

					if (!(focus::get_focus() == get_focus_info(window)) || i == fan::uninitialized) {
						return;
					}

					fan::vector_t text_vector;
					vector_init(&text_vector, sizeof(uint8_t));

					fan::vector_t cursor_vector;
					vector_init(&cursor_vector, sizeof(FED_ExportedCursor_t));

					uint32_t line_index = 0; /* we dont know which line we are at so */
					FED_LineReference_t line_reference = FED_GetLineReferenceByLineIndex(&m_store[i].m_wed, 0);
					//auto t = fan::time::clock::now();

					m_cursor.clear(context);
					//fan::print(fan::time::clock::elapsed(t));
					while (1) {
						bool is_endline;
						FED_ExportLine(&m_store[i].m_wed, line_reference, &text_vector, &cursor_vector, &is_endline, utf8_data_callback);

						line_reference = _FED_LineList_GetNodeByReference(&m_store[i].m_wed.LineList, line_reference)->NextNodeReference;

						for (uint_t j = 0; j < cursor_vector.Current; j++) {
							FED_ExportedCursor_t* exported_cursor = &((FED_ExportedCursor_t*)cursor_vector.ptr.data())[j];

							auto src_dst = get_cursor_src_dst(window, context, i, exported_cursor->x, line_index);

							fan_2d::opengl::rectangle_t::properties_t cursor_properties;
							cursor_properties.color = cursor_properties::color;
							cursor_properties.position = src_dst.src;
							cursor_properties.size = src_dst.dst;

							m_cursor.push_back(context, cursor_properties);
						}

						cursor_vector.Current = 0;

						if (line_reference == m_store[i].m_wed.LineList.dst) {
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

					fan::vector_t text_vector;
					vector_init(&text_vector, sizeof(uint8_t));

					fan::vector_t cursor_vector;
					vector_init(&cursor_vector, sizeof(FED_ExportedCursor_t));

					uint32_t line_index = 0;
					FED_LineReference_t line_reference = FED_GetLineReferenceByLineIndex(&m_store[i].m_wed, 0);

					while (1) {
						bool is_endline;
						FED_ExportLine(&m_store[i].m_wed, line_reference, &text_vector, &cursor_vector, &is_endline, utf8_data_callback);

						line_reference = _FED_LineList_GetNodeByReference(&m_store[i].m_wed.LineList, line_reference)->NextNodeReference;

						for (uint_t i = 0; i < cursor_vector.Current; i++) {
							FED_ExportedCursor_t* exported_cursor = &((FED_ExportedCursor_t*)cursor_vector.ptr.data())[i];
							exported_cursor->y = line_index;
							cursor_src_dsts.emplace_back(get_cursor_src_dst(0, i, exported_cursor->x, exported_cursor->y));
						}

						cursor_vector.Current = 0;

						if (line_reference == m_store[i].m_wed.LineList.dst) {
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

				bool render_cursor;

				fan::time::clock cursor_timer;

				struct store_t {
					FED_CursorReference_t cursor_reference;
					FED_t m_wed;
					uint32_t m_input_allowed;
				};

				fan::hector_t<store_t> m_store;

				fan_2d::opengl::rectangle_t m_cursor;

			};
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
			enum class text_position_e {
				left,
				middle
			};
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
			struct rectangle_button_sized_properties {

				fan::utf16_string text;

				fan::utf16_string place_holder;

				fan::vec2 position = 0;

				f32_t font_size = fan_2d::opengl::gui::defaults::font_size;

				text_position_e text_position = text_position_e::middle;

				fan_2d::opengl::gui::theme theme = fan_2d::opengl::gui::themes::deep_blue();

				fan::vec2 size = 0;

				fan::vec2 offset = 0;

			};
			//
			//			// returns half size
			//			static fan::vec2 get_button_size(const fan::utf16_string text, f32_t font_size, uint32_t new_lines, const fan::vec2& padding)
			//			{
			//
			//				f32_t h = text_renderer_t::font.line_height * fan_2d::opengl::gui::text_renderer_t::convert_font_size(font_size) * (new_lines + 1);
			//
			//				return ((text.empty() ? fan::vec2(0, h) : fan::vec2(fan_2d::opengl::gui::text_renderer_t::get_text_size(text, font_size).x, h)) + padding) / 2; 
			//			}
			//
			struct rectangle_text_box_sized :
				protected fan::class_duplicator<fan_2d::opengl::rectangle_t, 0>,
				protected fan::class_duplicator<fan_2d::opengl::rectangle_t, 1>,
				public fan_2d::opengl::gui::text_renderer_t
			{

				using properties_t = rectangle_button_sized_properties;

				using inner_rect_t = fan::class_duplicator<fan_2d::opengl::rectangle_t, 0>;
				using outer_rect_t = fan::class_duplicator<fan_2d::opengl::rectangle_t, 1>;

				rectangle_text_box_sized() = default;

				void open(fan::opengl::context_t* context);
				void close(fan::opengl::context_t* context);

				void push_back(fan::opengl::context_t* context, const properties_t& properties);

				void set_position(fan::opengl::context_t* context, uint32_t i, const fan::vec2& position);


				bool inside(fan::opengl::context_t* context, uintptr_t i, const fan::vec2& position) const;

				void set_text(fan::opengl::context_t* context, uint32_t i, const fan::utf16_string& text);
				void set_size(fan::opengl::context_t* context, uint32_t i, const fan::vec2& size);

				fan::color get_text_color(fan::opengl::context_t* context, uint32_t i) const;
				void set_text_color(fan::opengl::context_t* context, uint32_t i, const fan::color& color);

				fan::vec2 get_position(fan::opengl::context_t* context, uint32_t i) const;
				fan::vec2 get_size(fan::opengl::context_t* context, uint32_t i) const;

				f32_t get_font_size(fan::opengl::context_t* context, uint32_t i) const;

				fan::color get_color(fan::opengl::context_t* context, uint32_t i) const;

				// receives uint32_t box_i, uint32_t character number x, uint32_t character number y
				src_dst_t get_cursor(fan::opengl::context_t* context, uint32_t i, uint32_t x, uint32_t y);

				fan::vec2 get_text_starting_point(fan::opengl::context_t* context, uint32_t i) const;

				properties_t get_property(fan::opengl::context_t* context, uint32_t i) const;

				void set_offset(fan::opengl::context_t* context, uint32_t i, const fan::vec2& offset);

				uintptr_t size(fan::opengl::context_t* context) const;

				void erase(fan::opengl::context_t* context, uint32_t i);
				void erase(fan::opengl::context_t* context, uint32_t begin, uint32_t end);

				void enable_draw(fan::opengl::context_t* context);
				void disable_draw(fan::opengl::context_t* context);

				// sets shape's draw order in window
				//void set_draw_order(uint32_t i);

				void clear(fan::opengl::context_t* context);

				void update_theme(fan::opengl::context_t* context, uint32_t i);
				void set_theme(fan::opengl::context_t* context, uint32_t i, const fan_2d::opengl::gui::theme& theme_);

				using inner_rect_t::get_color;

				using fan_2d::opengl::rectangle_t::get_size;

			protected:

				struct p_t {
					fan::utf16_string_ptr_t text;

					fan::utf16_string_ptr_t place_holder;

					fan::vec2 position;

					f32_t font_size = fan_2d::opengl::gui::defaults::font_size;

					text_position_e text_position = text_position_e::middle;

					theme_ptr_t theme;

					fan::vec2 size;

					fan::vec2 offset;
				};

				struct store_t {
					p_t m_properties;
				};

				void draw(fan::opengl::context_t* context);

				void write_data();

				void edit_data(uint32_t i);

				void edit_data(uint32_t begin, uint32_t end);

				fan::hector_t<store_t> m_store;

			};
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
			enum class button_states_e {
				clickable = 1,
				locked = 2
			};

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

				// text color needs to be less than 1.0 or 255 in rgb to see color change
			struct text_renderer_clickable :
				public text_renderer_t
			{

				struct properties_t : public text_renderer_t::properties_t {
					fan::vec2 hitbox_position; // middle
					fan::vec2 hitbox_size; // half
				};

				void open(fan::window_t* window, fan::opengl::context_t* context);
				void close(fan::window_t* window, fan::opengl::context_t* context);

				void lib_add_on_input(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, uint16_t key, fan::key_state state, fan_2d::opengl::gui::mouse_stage stage);

				void lib_add_on_mouse_move(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, fan_2d::opengl::gui::mouse_stage stage);

				void push_back(fan::window_t* window, fan::opengl::context_t* context, const text_renderer_clickable::properties_t& properties);

				// hitbox is half size
				void set_hitbox(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, const fan::vec2& hitbox_position, const fan::vec2& hitbox_size);

				fan::vec2 get_hitbox_position(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) const;
				void set_hitbox_position(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, const fan::vec2& hitbox_position);
				// hitbox is half size
				fan::vec2 get_hitbox_size(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) const;
				void set_hitbox_size(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, const fan::vec2& hitbox_size);

				bool inside(fan::opengl::context_t* context, uint32_t i, const fan::vec2& p) const;
				bool locked(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) const { return 0; }

				static constexpr f32_t hover_strength = 0.2;
				static constexpr f32_t click_strength = 0.3;

				base::button_event_t<text_renderer_clickable> m_button_event;

			protected:

				using text_renderer_t::erase;
				using text_renderer_t::insert;

				struct hitbox_t {
					fan::vec2 hitbox_position; // middle
					fan::vec2 hitbox_size; // half
				};

				struct store_t {
					hitbox_t m_hitbox;
					uint8_t previous_states;
				};

				fan::hector_t<store_t> m_store;
			};

			struct rectangle_text_button_sized :
				public fan_2d::opengl::gui::rectangle_text_box_sized
			{

				rectangle_text_button_sized() = default;

				struct properties_t : public rectangle_text_box_sized::properties_t {

					using rectangle_text_box_sized::properties_t::rectangle_button_sized_properties;

					f32_t character_width = (f32_t)0xdfffffff / key_event_t<rectangle_text_button_sized>::line_multiplier;
					uint32_t character_limit = -1;
					uint32_t line_limit = -1;
					button_states_e button_state = button_states_e::clickable;
				};

				void open(fan::window_t* window, fan::opengl::context_t* context);
				void close(fan::window_t* window, fan::opengl::context_t* context);

				void push_back(fan::window_t* window, fan::opengl::context_t* context, properties_t properties);

				void set_place_holder(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, const fan::utf16_string& place_holder);

				void backspace_callback(fan::window_t* window, fan::opengl::context_t* context, uint32_t i);
				void text_callback(fan::window_t* window, fan::opengl::context_t* context, uint32_t i);

				void erase(fan::window_t* window, fan::opengl::context_t* context, uint32_t i);
				void erase(fan::window_t* window, fan::opengl::context_t* context, uint32_t begin, uint32_t end);

				void clear(fan::window_t* window, fan::opengl::context_t* context);

				void set_locked(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, bool flag, bool change_theme = true);

				bool locked(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) const;

				void enable_draw(fan::window_t* window, fan::opengl::context_t* context);
				void disable_draw(fan::window_t* window, fan::opengl::context_t* context);

				void lib_add_on_input(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, uint16_t key, fan::key_state state, fan_2d::opengl::gui::mouse_stage stage);

				void lib_add_on_mouse_move(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, fan_2d::opengl::gui::mouse_stage stage);

				fan_2d::opengl::gui::key_event_t<rectangle_text_button_sized> m_key_event;
				fan_2d::opengl::gui::base::button_event_t<fan_2d::opengl::gui::rectangle_text_button_sized> m_button_event;

			protected:

				void draw(fan::window_t* window, fan::opengl::context_t* context);

				fan::hector_t<uint32_t> m_reserved;
			};
			//
			//			struct rectangle_selectable_button_sized : public rectangle_text_button_sized{
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
			//			struct dropdown_menu : protected rectangle_text_button_sized {
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
			//				using rectangle_text_button_sized::get_camera;
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