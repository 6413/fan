#pragma once

#include <fan/graphics/opengl/gl_core.h>
#include <fan/graphics/gui/types.h>
#include <fan/graphics/gui/focus.h>
#include <fan/graphics/opengl/2D/gui/text_renderer.h>

#define FED_set_debug_InvalidLineAccess 1
#include <fan/fed/FED.h>

namespace fan_2d {
  namespace graphics {
    namespace gui {
			using namespace fan_2d::graphics::gui::focus;

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

					//keys_callback_id = window->add_keys_callback(context, [](fan::window_t* w, uint16 key, fan::key_state state, void* user_ptr) {

					//	auto object = OFFSETLESS(this, T, m_key_event);

					//	// not pushed
					//	if (current_focus.i >= m_store.size()) {
					//		return;
					//	}

					//	fan_2d::graphics::gui::focus::properties_t p;
					//	p.window_handle = (void*)w->get_handle();
					//	p.shape = this;

					//	if (
					//		!fan_2d::graphics::gui::focus::has_no_focus(p) ||
					//		!m_store[current_focus.i].m_input_allowed
					//		) {
					//		return;
					//	}

					//	switch (state) {
					//	case fan::key_state::press: {

					//		if (mouse_move_callback_id != -1) {
					//			break;
					//		}

					//		click_begin = fan::cast<f32_t>(w->get_mouse_position()) - object->get_text_starting_point(w, context, current_focus.i);

					//		mouse_move_callback_id = w->add_mouse_move_callback([this, context, object](fan::window_t* w, const fan::vec2& p) {
					//			if (!w->key_press(fan::mouse_left)) {
					//				return;
					//			}

					//			fan_2d::graphics::gui::focus::properties_t fp;
					//			fp.window_handle = (void*)w->get_handle();
					//			fp.shape = this;

					//			if (
					//				!fan_2d::graphics::gui::focus::has_no_focus(fp) ||
					//				!m_store[current_focus.i].m_input_allowed)
					//			{
					//				return;
					//			}

					//			fan::vec2 src = click_begin;
					//			// dst release
					//			fan::vec2 dst = fan::cast<f32_t>(w->get_mouse_position()) - object->get_text_starting_point(w, context, current_focus.i);
					//			dst.x = fan::clamp(dst.x, (f32_t)0, dst.x);

					//			FED_LineReference_t FirstLineReference = _FED_LineList_GetNodeFirst(&m_store[current_focus.i].m_wed.LineList);
					//			FED_LineReference_t LineReference0, LineReference1;
					//			FED_CharacterReference_t CharacterReference0, CharacterReference1;
					//			FED_GetLineAndCharacter(&m_store[current_focus.i].m_wed, FirstLineReference, src.y, src.x * line_multiplier, &LineReference0, &CharacterReference0);
					//			FED_GetLineAndCharacter(&m_store[current_focus.i].m_wed, FirstLineReference, dst.y, dst.x * line_multiplier, &LineReference1, &CharacterReference1);
					//			FED_ConvertCursorToSelection(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference, LineReference0, CharacterReference0, LineReference1, CharacterReference1);

					//			update_cursor(w, context, current_focus.i);

					//			render_cursor = true;
					//			cursor_timer.restart();
					//			m_cursor.enable_draw(context);
					//		});

					//		break;
					//	}
					//	case fan::key_state::release: {

					//		if (mouse_move_callback_id != -1) {
					//			w->remove_mouse_move_callback(mouse_move_callback_id);
					//			mouse_move_callback_id = -1;
					//		}

					//		return;
					//		break;
					//	}
					//	}

					//	render_cursor = true;
					//	cursor_timer.restart();
					//	m_cursor.enable_draw(context);

					//	switch (key) {
					//	case fan::key_backspace: {
					//		if (object->get_text(w, context, current_focus.i).size() == 1) {
					//			object->backspace_callback(w, context, current_focus.i);
					//			FED_DeleteCharacterFromCursor(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference);
					//			object->set_text(w, context, current_focus.i, L" ");

					//			break;
					//		}

					//		FED_DeleteCharacterFromCursor(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference);

					//		update_text(w, context, current_focus.i);

					//		object->backspace_callback(w, context, current_focus.i);

					//		update_cursor(w, context, current_focus.i);

					//		break;
					//	}
					//	case fan::key_delete: {
					//		FED_DeleteCharacterFromCursorRight(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference);

					//		update_text(w, context, current_focus.i);

					//		object->backspace_callback(w, context, current_focus.i);

					//		break;
					//	}
					//	case fan::key_left: {
					//		FED_MoveCursorFreeStyleToLeft(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference);

					//		update_cursor(w, context, current_focus.i);

					//		break;
					//	}
					//	case fan::key_right: {

					//		FED_MoveCursorFreeStyleToRight(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference);

					//		update_cursor(w, context, current_focus.i);

					//		break;
					//	}
					//	case fan::key_up: {
					//		FED_MoveCursorFreeStyleToUp(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference);

					//		update_cursor(w, context, current_focus.i);

					//		break;
					//	}
					//	case fan::key_down: {
					//		FED_MoveCursorFreeStyleToDown(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference);

					//		update_cursor(w, context, current_focus.i);

					//		break;
					//	}
					//	case fan::key_home: {
					//		FED_MoveCursorFreeStyleToBeginOfLine(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference);

					//		update_cursor(w, context, current_focus.i);

					//		break;
					//	}
					//	case fan::key_end: {
					//		FED_MoveCursorFreeStyleToEndOfLine(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference);

					//		update_cursor(w, context, current_focus.i);

					//		break;
					//	}
					//	case fan::key_enter: {

					//		focus::set_focus(focus::no_focus);

					//		break;
					//	}
					//	case fan::key_tab: {

					//		auto current_focus = focus::get_focus();

					//		if (w->key_press(fan::key_shift)) {
					//			if (current_focus.i - 1 == ~0) {
					//				current_focus.i = object->size(w, context) - 1;
					//			}
					//			else {
					//				current_focus.i = (current_focus.i - 1) % object->size(w, context);
					//			}
					//		}
					//		else {
					//			current_focus.i = (current_focus.i + 1) % object->size(w, context);
					//		}

					//		focus::set_focus(current_focus);

					//		update_cursor(w, context, current_focus.i);

					//		break;
					//	}
					//	case fan::mouse_left: {

					//		// src press
					//		fan::vec2 src = fan::cast<f32_t>(w->get_mouse_position()) - object->get_text_starting_point(w, context, current_focus.i);
					//		// dst release
					//		src.x = fan::clamp(src.x, (f32_t)0, src.x);
					//		fan::vec2 dst = src;

					//		FED_LineReference_t FirstLineReference = _FED_LineList_GetNodeFirst(&m_store[current_focus.i].m_wed.LineList);
					//		FED_LineReference_t LineReference0, LineReference1;
					//		FED_CharacterReference_t CharacterReference0, CharacterReference1;
					//		FED_GetLineAndCharacter(&m_store[current_focus.i].m_wed, FirstLineReference, src.y, src.x * line_multiplier, &LineReference0, &CharacterReference0);
					//		FED_GetLineAndCharacter(&m_store[current_focus.i].m_wed, FirstLineReference, dst.y, dst.x * line_multiplier, &LineReference1, &CharacterReference1);
					//		FED_ConvertCursorToSelection(&m_store[current_focus.i].m_wed, m_store[current_focus.i].cursor_reference, LineReference0, CharacterReference0, LineReference1, CharacterReference1);

					//		update_cursor(w, context, current_focus.i);

					//		break;
					//	}
					//	case fan::key_v: {

					//		if (w->key_press(fan::key_control)) {

					//			auto pasted_text = fan::io::get_clipboard_text(w->get_handle());

					//			for (int i = 0; i < pasted_text.size(); i++) {
					//				add_character(context, &m_store[current_focus.i].m_wed, &m_store[current_focus.i].cursor_reference, pasted_text[i], object->get_font_size(w, context, current_focus.i));
					//			}

					//			update_text(w, context, current_focus.i);
					//		}

					//		break;
					//	}
					//	}

					//});

					//text_callback_id = window->add_text_callback([this, context, object = OFFSETLESS(this, T, m_key_event)](fan::window_t* w, uint32_t character) {

					//	fan_2d::graphics::gui::focus::properties_t p;
					//	p.window_handle = (void*)w->get_handle();
					//	p.shape = this;

					//	if (current_focus.i >= m_store.size()) {
					//		return;
					//	}

					//	if (
					//		!fan_2d::graphics::gui::focus::has_no_focus(p) ||
					//		!m_store[current_focus.i].m_input_allowed
					//		) {
					//		return;
					//	}

					//	render_cursor = true;
					//	cursor_timer.restart();

					//	m_cursor.enable_draw(context);

					//	fan::utf8_string utf8;
					//	utf8.push_back(character);

					//	auto wc = utf8.to_utf16()[0];

					//	f32_t font_size = object->get_font_size(w, context, current_focus.i);
					//	add_character(context, &m_store[current_focus.i].m_wed, &m_store[current_focus.i].cursor_reference, character, font_size);

					//	fan::vector_t text_vector;
					//	vector_init(&text_vector, sizeof(uint8_t));

					//	fan::vector_t cursor_vector;
					//	vector_init(&cursor_vector, sizeof(FED_ExportedCursor_t));

					//	uint32_t line_index = 0;
					//	FED_LineReference_t line_reference = FED_GetLineReferenceByLineIndex(&m_store[current_focus.i].m_wed, 0);

					//	std::vector<FED_ExportedCursor_t*> exported_cursors;

					//	while (1) {
					//		bool is_endline;
					//		FED_ExportLine(&m_store[current_focus.i].m_wed, line_reference, &text_vector, &cursor_vector, &is_endline, utf8_data_callback);

					//		line_reference = _FED_LineList_GetNodeByReference(&m_store[current_focus.i].m_wed.LineList, line_reference)->NextNodeReference;

					//		for (uint_t i = 0; i < cursor_vector.Current; i++) {
					//			FED_ExportedCursor_t* exported_cursor = &((FED_ExportedCursor_t*)cursor_vector.ptr.data())[i];
					//			exported_cursor->y = line_index;
					//			exported_cursors.emplace_back(exported_cursor);
					//		}

					//		cursor_vector.Current = 0;

					//		if (line_reference == m_store[current_focus.i].m_wed.LineList.dst) {
					//			break;
					//		}

					//		{
					//			fan::VEC_handle(&text_vector);
					//			text_vector.ptr[text_vector.Current] = '\n';
					//			text_vector.Current++;
					//		}


					//		line_index++;
					//	}

					//	{
					//		fan::VEC_handle(&text_vector);
					//		text_vector.ptr[text_vector.Current] = 0;
					//		text_vector.Current++;
					//	}

					//	object->set_text(w, context, current_focus.i, text_vector.ptr.data());

					//	object->text_callback(w, context, current_focus.i);

					//	for (int i = 0; i < exported_cursors.size(); i++) {

					//		FED_ExportedCursor_t* exported_cursor = exported_cursors[i];

					//		auto src_dst = get_cursor_src_dst(w, context, current_focus.i, exported_cursor->x, exported_cursor->y);

					//		m_cursor.set_position(context, 0, src_dst.src);
					//		m_cursor.set_size(context, 0, src_dst.dst);
					//		update_cursor(w, context, current_focus.i);

					//	}
					//});
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
						fp.window_handle = 0;
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

					auto str = object->get_text(window, context, offset);

					for (int i = 0; i < str.size(); i++) {
						add_character(context, &m_store[offset].m_wed, &m_store[offset].cursor_reference, str[i], object->get_font_size(window, context, offset));
					}

					update_cursor(window, context, object->size(window, context) - 1);
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
					fan_2d::graphics::gui::focus::properties_t p;
					p.window_handle = (void*)window->get_handle();
					p.shape = this;
					p.i = i;

					if (fan_2d::graphics::gui::focus::has_focus(p)) {
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
					return { (void*)window->get_handle(), (void*)this, 0 };
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
					return object->get_cursor(window, context, rtb_index, x, line_index);
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

					object->set_text(window, context, i, text_vector.ptr.data());

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
    }
  }
}