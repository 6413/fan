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

        typedef void(*focus_loss_cb_t)(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, void* userptr);

        /* used for hold precise floats on FED */
        /* bigger number means better precision */
        static constexpr uint32_t line_multiplier = 100;

        void open(fan::window_t* w, fan::opengl::context_t* c) {

          context = c;
          window = w;
          object = OFFSETLESS(this, T, m_key_event);

          m_store.open();
          m_cursor.open(context);
          cursor_timer = fan::time::clock(cursor_properties::blink_speed);
          cursor_timer.start();

          m_draw_node_reference = fan::uninitialized;
          mouse_move_callback_id = fan::uninitialized;

          focus_loss_cb = [](fan::window_t* window, fan::opengl::context_t* context, uint32_t i, void* userptr) {};

          keys_callback_id = window->add_keys_callback(this, [](fan::window_t* window, uint16 key, fan::key_state state, void* userptr) {
            key_event_t<T>* instance = (key_event_t<T>*)userptr;

            if (state == fan::key_state::release) {
              return;
            }

            if (current_focus.i == no_focus) {
              return;
            }
            if (current_focus.i >= instance->m_store.size()) {
              return;
            }
            if (!instance->input_allowed(window, current_focus.i)) {
              return;
            }
            if (!instance->is_focused()) {
              return;
            }

            FED_CursorReference_t cursor_reference = instance->m_store[current_focus.i].cursor_reference;
            FED_t* wed = &instance->m_store[current_focus.i].m_wed;

            switch (key) {
              case fan::key_backspace: {
                if (instance->object->get_text(window, instance->context, current_focus.i).size() == 1) {
                  instance->object->backspace_callback(window, instance->context, current_focus.i);
                  FED_DeleteCharacterFromCursor(
                    wed,
                    cursor_reference
                  );
                  instance->object->set_text(window, instance->context, current_focus.i, L" ");

                  break;
                }

                FED_DeleteCharacterFromCursor(
                  wed,
                  cursor_reference
                );

                instance->update_text(window, instance->context, current_focus.i);

                instance->object->backspace_callback(window, instance->context, current_focus.i);

                instance->update_cursor(window, instance->context, current_focus.i);

                break;
              }
              case fan::key_delete: {

                if (instance->object->get_text(window, instance->context, current_focus.i).size() == 1) {
                  instance->object->backspace_callback(window, instance->context, current_focus.i);
                  FED_DeleteCharacterFromCursorRight(wed, cursor_reference);
                  instance->object->set_text(window, instance->context, current_focus.i, L" ");
                  break;
                }

                FED_DeleteCharacterFromCursorRight(wed, cursor_reference);

                instance->update_text(window, instance->context, current_focus.i);

                instance->object->backspace_callback(window, instance->context, current_focus.i);

                instance->update_cursor(window, instance->context, current_focus.i);

                break;
              }
              case fan::key_left: {
                FED_MoveCursorFreeStyleToLeft(wed, cursor_reference);

                instance->update_cursor(window, instance->context, current_focus.i);

                break;
              }
              case fan::key_right: {

                FED_MoveCursorFreeStyleToRight(wed, cursor_reference);

                instance->update_cursor(window, instance->context, current_focus.i);

                break;
              }
              case fan::key_up: {
                FED_MoveCursorFreeStyleToUp(wed, cursor_reference);

                instance->update_cursor(window, instance->context, current_focus.i);

                break;
              }
              case fan::key_down: {
                FED_MoveCursorFreeStyleToDown(wed, cursor_reference);

                instance->update_cursor(window, instance->context, current_focus.i);

                break;
              }
              case fan::key_home: {
                FED_MoveCursorFreeStyleToBeginOfLine(wed, cursor_reference);

                instance->update_cursor(window, instance->context, current_focus.i);

                break;
              }
              case fan::key_end: {
                FED_MoveCursorFreeStyleToEndOfLine(wed, cursor_reference);

                instance->update_cursor(window, instance->context, current_focus.i);

                break;
              }
              case fan::key_enter: {

                instance->set_focus(focus::no_focus);

                break;
              }
              case fan::key_tab: {

                auto current_focus = focus::get_focus();

                // if keypress(shift)
                if (0) {
                  if (current_focus.i - 1 == ~0) {
                    current_focus.i = instance->object->size(window, instance->context) - 1;
                  }
                  else {
                    current_focus.i = (current_focus.i - 1) % instance->object->size(window, instance->context);
                  }
                }
                else {
                  current_focus.i = (current_focus.i + 1) % instance->object->size(window, instance->context);
                }

                instance->set_focus(current_focus.i);

                instance->update_cursor(window, instance->context, current_focus.i);

                break;
              }
            }

          });

          text_callback_id = window->add_text_callback(this, [](fan::window_t* window, uint32_t character, void* user_ptr) {

            key_event_t<T>* instance = (key_event_t<T>*)user_ptr;

            if (current_focus.i == no_focus) {
              return;
            }
            if (current_focus.i >= instance->m_store.size()) {
              return;
            }
            if (!instance->input_allowed(window, current_focus.i)) {
              return;
            }
            if (!instance->is_focused()) {
              return;
            }

            instance->render_cursor = true;
            instance->cursor_timer.restart();

            FED_CursorReference_t cursor_reference = instance->m_store[current_focus.i].cursor_reference;
            FED_t* wed = &instance->m_store[current_focus.i].m_wed;

            fan::utf8_string utf8;
            utf8.push_back(character);

            auto wc = utf8.to_utf16()[0];

            f32_t font_size = instance->object->get_font_size(window, instance->context, current_focus.i);
            instance->add_character(instance->context, wed, &cursor_reference, character, font_size);

            fan::vector_t text_vector;
            vector_init(&text_vector, sizeof(uint8_t));

            fan::vector_t cursor_vector;
            vector_init(&cursor_vector, sizeof(FED_ExportedCursor_t));

            uint32_t line_index = 0;
            FED_LineReference_t line_reference = FED_GetLineReferenceByLineIndex(wed, 0);

            std::vector<FED_ExportedCursor_t*> exported_cursors;

            while (1) {
              bool is_endline;
              FED_ExportLine(wed, line_reference, &text_vector, &cursor_vector, &is_endline, utf8_data_callback);

              line_reference = _FED_LineList_GetNodeByReference(&wed->LineList, line_reference)->NextNodeReference;

              for (uint_t i = 0; i < cursor_vector.Current; i++) {
                FED_ExportedCursor_t* exported_cursor = &((FED_ExportedCursor_t*)cursor_vector.ptr.data())[i];
                exported_cursor->y = line_index;
                exported_cursors.emplace_back(exported_cursor);
              }

              cursor_vector.Current = 0;

              if (line_reference == wed->LineList.dst) {
                break;
              }

              fan::VEC_handle(&text_vector);
              text_vector.ptr[text_vector.Current] = '\n';
              text_vector.Current++;

              line_index++;
            }

            fan::VEC_handle(&text_vector);
            text_vector.ptr[text_vector.Current] = 0;
            text_vector.Current++;

            instance->object->set_text(window, instance->context, current_focus.i, text_vector.ptr.data());

            instance->object->text_callback(window, instance->context, current_focus.i);

            for (int i = 0; i < exported_cursors.size(); i++) {

              FED_ExportedCursor_t* exported_cursor = exported_cursors[i];

              auto src_dst = instance->get_cursor_src_dst(window, instance->context, current_focus.i, exported_cursor->x, exported_cursor->y);

              instance->update_cursor(window, instance->context, current_focus.i);
              instance->m_cursor.set_position(instance->context, 0, src_dst.src);
              instance->m_cursor.set_size(instance->context, 0, src_dst.dst);
            }
          });

        }
        void close(fan::window_t* window, fan::opengl::context_t* context) {
          m_store.close();

					if (keys_callback_id != fan::uninitialized) {
						window->remove_keys_callback(keys_callback_id);
						keys_callback_id = fan::uninitialized;
					}

					if (text_callback_id != fan::uninitialized) {
						window->remove_text_callback(text_callback_id);
						text_callback_id = fan::uninitialized;
					}

					if (m_draw_node_reference != fan::uninitialized) {
						context->m_draw_queue.erase(m_draw_node_reference);
						m_draw_node_reference = fan::uninitialized;
					}

					if (mouse_move_callback_id != fan::uninitialized) {
						window->remove_mouse_move_callback(mouse_move_callback_id);
						mouse_move_callback_id = fan::uninitialized;
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

        void enable_draw(fan::window_t* window, fan::opengl::context_t* context) {
          m_cursor.enable_draw(context);
          m_draw_node_reference = context->enable_draw(this, [](fan::opengl::context_t* c, void* d) {
            key_event_t<T>* thiS = ((decltype(this))d);
            thiS->draw(thiS->window, thiS->context);
          });
        }
        void disable_draw(fan::window_t* window, fan::opengl::context_t* context) {
          #if fan_debug >= fan_debug_low
          if (m_draw_node_reference == fan::uninitialized) {
            fan::throw_error("trying to disable unenabled draw call");
          }
          #endif
          context->disable_draw(m_draw_node_reference);
        }

        void draw(fan::window_t* window, fan::opengl::context_t* context) {

          if (cursor_timer.finished()) {
            cursor_timer.restart();

            auto focus_ = focus::get_focus();

            if (
              has_focus(focus_) && render_cursor &&
              focus_ == get_focus_info(window) &&
              m_store[focus_.i].m_input_allowed
              ) {
              update_cursor(window, context, focus_.i);
              m_cursor.draw(context);
            }
            else {
              m_cursor.clear(context);
            }
            render_cursor = !render_cursor;
          }

        }

        focus::properties_t get_focus_info(fan::window_t* window) const {
          return { (void*)window->get_handle(), (void*)this->object, 0 };
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

          this->set_focus(current_focus.i);

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

        bool is_focused() const {
          fan_2d::graphics::gui::focus::properties_t p;
          p.window_handle = (void*)window->get_handle();
          p.shape = object;

          return !fan_2d::graphics::gui::focus::has_no_focus(p);
        }

        void set_on_focus_loss_callback(void* userptr_, focus_loss_cb_t focus_loss_cb_) {
          focus_loss_cb = focus_loss_cb_;
          focus_loss_userptr = userptr_;
        }

        void set_focus(uint32_t focus_id) {
          focus_loss_cb(window, context, current_focus.i, focus_loss_userptr);
          focus::set_focus(focus_id);
        }

        uint32_t m_draw_node_reference;

        bool render_cursor;

        fan::time::clock cursor_timer;

        struct store_t {
          FED_CursorReference_t cursor_reference;
          FED_t m_wed;
          uint32_t m_input_allowed;
        };

        fan::hector_t<store_t> m_store;

        fan_2d::opengl::rectangle_t m_cursor;
        fan::graphics::context_t* context;
        fan::window_t* window;
        T* object;

        uint32_t text_callback_id;
        uint32_t keys_callback_id;
        uint32_t mouse_move_callback_id;

        fan::vec2 click_begin;

        focus_loss_cb_t focus_loss_cb;
        void* focus_loss_userptr;

      };
    }
  }
}