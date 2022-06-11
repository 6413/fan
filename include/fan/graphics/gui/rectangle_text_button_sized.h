#pragma once

#include _FAN_PATH(graphics/gui/rectangle_text_box_sized.h)
#include _FAN_PATH(graphics/gui/button_event.h)
#include _FAN_PATH(graphics/gui/key_event.h)

namespace fan_2d {
  namespace graphics {
    namespace gui {
      struct rectangle_text_button_sized_t {

        rectangle_text_button_sized_t() = default;

        struct properties_t : public rectangle_text_box_sized_t::properties_t {

          using rectangle_text_box_sized_t::properties_t::properties_t;

          f32_t character_width = (f32_t)0xdfffffff / key_event_t<rectangle_text_button_sized_t>::line_multiplier;
          uint32_t character_limit = -1;
          uint32_t line_limit = -1;
          button_states_e button_state = button_states_e::clickable;
          bool allow_input = false;
        };

        void open(fan::window_t* window, fan::opengl::context_t* context) {
          rtbs.open(context);
          m_key_event.open(window, context);
          m_button_event.open(window, context);
          m_reserved.open();

          viewport_collision_offset = 0;
        }

        void close(fan::window_t* window, fan::opengl::context_t* context) {
          rtbs.close(context);
          m_key_event.close(window, context);
          m_button_event.close(window, context);
          m_reserved.close();
          if (decltype(m_button_event)::pointer_remove_flag == 1) {
            decltype(m_button_event)::pointer_remove_flag = 0;
          }
        }

        void push_back(fan::window_t* window, fan::opengl::context_t* context, properties_t properties)
        {
          m_reserved.emplace_back((uint32_t)properties.button_state);
          if (properties.button_state == button_states_e::locked) {

            properties.theme = fan_2d::graphics::gui::themes::locked();
          }

          rtbs.push_back(context, properties);

          m_key_event.push_back(window, context, properties.character_limit, properties.character_width, properties.line_limit);

          if (properties.allow_input) {
            m_key_event.allow_input(window, rtbs.size(context) - 1, true);
          }

          if (inside(window, context, size(window, context) - 1, window->get_mouse_position()) && properties.button_state != button_states_e::locked) {
            m_button_event.m_focused_button_id = size(window, context) - 1;
            lib_add_on_mouse_move(window, context, m_button_event.m_focused_button_id, fan_2d::graphics::gui::mouse_stage::inside, m_button_event.mouse_user_ptr);
          }
        }

        uint32_t size(fan::window_t* window, fan::opengl::context_t* context) {
          return rtbs.size(context);
        }
        bool inside(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, const fan::vec2& position) {
          return rtbs.inside(context, i, position);
        }

        void draw(fan::window_t* window, fan::opengl::context_t* context)
        {
          rtbs.draw(context);
          m_key_event.draw(window, context);
        }

        void backspace_callback(fan::window_t* window, fan::opengl::context_t* context, uint32_t i)
        {
          /*auto current_string = rtbs.get_text(context, i);
          auto current_property = rtbs.get_property(context, i);

          if (current_string.size() && current_string[0] == '\0' && current_property.place_holder.size()) {
            rtbs.set_text(context, i, current_property.place_holder);
            rtbs.set_text_color(context, i, defaults::text_color_place_holder);
          }*/
        }

        void text_callback(fan::window_t* window, fan::opengl::context_t* context, uint32_t i)
        {
          if (rtbs.get_text_color(context, i) != rtbs.rbs.m_store[i].m_properties.theme->button.text_color) {
            rtbs.set_text_color(context, i, rtbs.rbs.m_store[i].m_properties.theme->button.text_color);
          }
        }

        void erase(fan::window_t* window, fan::opengl::context_t* context, uint32_t i)
        {
          rtbs.erase(context, i);
          m_reserved.erase(i);
          m_key_event.set_focus(window, context, -1);
          m_key_event.erase(window, context, i);
        }

        void erase(fan::window_t* window, fan::opengl::context_t* context, uint32_t begin, uint32_t end)
        {
          rtbs.erase(context, begin, end);
          m_reserved.erase(begin, end);
          m_key_event.set_focus(window, context, -1);
          m_key_event.erase(window, context, begin, end);
        }

        void clear(fan::window_t* window, fan::opengl::context_t* context)
        {
          rtbs.clear(context);
          m_reserved.clear();

          m_key_event.set_focus(window, context, fan::uninitialized);
          m_key_event.clear(window, context);

          // otherwise default add_inputs in constructor will be erased as well
          //sprite::mouse::clear();
          //sprite::text_input::clear();
        }

        fan::utf16_string get_text(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) const {
          return rtbs.get_text(context, i);
        }
        void set_text(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, const fan::utf16_string& text) {
          rtbs.set_text(context, i, text);
        }
        f32_t get_font_size(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) const {
          return rtbs.get_font_size(context, i);
        }
        void set_font_size(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, f32_t font_size) {
          rtbs.set_font_size(context, i, font_size);
        }
        f32_t get_outline_size(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) const {
          return rtbs.get_outline_size(context, i);
        }
        void set_outline_size(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, f32_t outline_size) {
          rtbs.set_outline_size(context, i, outline_size);
        }
        fan::color get_text_color(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, uint32_t j = 0) const {
          return rtbs.get_text_color(context, i, j);
        }
        void set_text_color(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, const fan::color& color) {
          rtbs.set_text_color(context, i, color);
        }
        fan::color get_outline_color(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) const {
          return rtbs.get_outline_color(context, i);
        }
        void set_outline_color(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, const fan::color& outline_color) {
          rtbs.set_outline_color(context, i, outline_color);
        }
        fan::vec2 get_text_starting_point(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) const {
          return rtbs.get_text_starting_point(context, i);
        }

        fan::vec2 get_position(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) const {
          return rtbs.get_position(context, i);
        }
        void set_position(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, const fan::vec2& position) {
          rtbs.set_position(context, i, position);
        }
        fan::vec2 get_size(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) const {
          return rtbs.get_size(context, i);
        }
        void set_size(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, const fan::vec2& size) {
          rtbs.set_size(context, i, size);
        }
        void set_theme(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, const fan_2d::graphics::gui::theme& theme) {
          rtbs.set_theme(context, i, theme);
        }

        void set_locked(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, bool flag, bool change_theme) {
          if (flag) {
            if (m_button_event.m_focused_button_id == i) {
              m_button_event.m_focused_button_id = fan::uninitialized;
            }
            m_reserved[i] |= (uint32_t)button_states_e::locked;
            if (change_theme) {
              *rtbs.rbs.m_store[i].m_properties.theme = fan_2d::graphics::gui::themes::locked();
              rtbs.update_theme(context, i);
            }
          }
          else {
            m_reserved[i] &= ~(uint32_t)button_states_e::locked;
            if (inside(window, context, i, window->get_mouse_position())) {
              m_button_event.m_focused_button_id = i;
            }
          }
        }

        void* get_userptr(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) {
          return rtbs.get_userptr(context, i);
        }
        void set_userptr(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, void* userptr) {
          rtbs.set_userptr(context, i, userptr);
        }

        void lib_add_on_input(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, uint16_t key, fan::key_state state, fan_2d::graphics::gui::mouse_stage stage, void* user_ptr)
        {
          if (this->locked(window, context, i)) {
            return;
          }

          if (rtbs.rbs.m_store[i].m_properties.theme->button.m_click_callback) {
            rtbs.rbs.m_store[i].m_properties.theme->button.m_click_callback(window, i, key, state, stage, user_ptr);
          }

          if (key != fan::mouse_left) {
            return;
          }

          switch (stage) {
            case fan_2d::graphics::gui::mouse_stage::inside: {

              switch (state) {
                case fan::key_state::press: {
                  rtbs.rbs.set_background_color(context, i, rtbs.rbs.m_store[i].m_properties.theme->button.click_outline_color);
                  rtbs.rbs.set_foreground_color(context, i, rtbs.rbs.m_store[i].m_properties.theme->button.click_color);
                  break;
                }
                case fan::key_state::release: {
                  focus::set_focus(focus::properties_t((void*)window->get_handle(), this, i));
                  if (m_key_event.m_store[i].m_input_allowed) {
                    m_key_event.render_cursor = true;
                    m_key_event.update_cursor(window, context, i);
                    m_key_event.cursor_timer.restart();
                    if (m_key_event.m_cursor.m_draw_node_reference == fan::uninitialized) {
                      m_key_event.m_cursor.enable_draw(context);
                    }
                  }
                  else {
                    m_key_event.render_cursor = false;
                  }
                  rtbs.rbs.set_background_color(context, i, rtbs.rbs.m_store[i].m_properties.theme->button.hover_outline_color);
                  rtbs.rbs.set_foreground_color(context, i, rtbs.rbs.m_store[i].m_properties.theme->button.hover_color);
                  break;
                }
              }
              break;
            }
            case fan_2d::graphics::gui::mouse_stage::outside: {
              switch (state) {
                case fan::key_state::press: {
                  focus::set_focus(focus::no_focus);
                  break;
                }
                case fan::key_state::release: {
                  rtbs.rbs.set_background_color(context, i, rtbs.rbs.m_store[i].m_properties.theme->button.outline_color);
                  rtbs.rbs.set_foreground_color(context, i, rtbs.rbs.m_store[i].m_properties.theme->button.color);
                  for (int j = 0; j < this->size(window, context); j++) {
                    if (this->inside(window, context, j, window->get_mouse_position()) && !this->locked(window, context, j)) {
                      rtbs.rbs.set_background_color(context, i, rtbs.rbs.m_store[i].m_properties.theme->button.hover_outline_color);
                      rtbs.rbs.set_foreground_color(context, i, rtbs.rbs.m_store[i].m_properties.theme->button.hover_color);
                      break;
                    }
                  }
                  break;
                }
              }
              break;
            }
          }
        }

        void lib_add_on_mouse_move(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, fan_2d::graphics::gui::mouse_stage stage, void* user_ptr)
        {
          if (this->locked(window, context, i)) {
            return;
          }

          if (rtbs.rbs.m_store[i].m_properties.theme->button.m_hover_callback) {
            rtbs.rbs.m_store[i].m_properties.theme->button.m_hover_callback(window, i, stage, user_ptr);
          }

          switch (stage) {
            case mouse_stage::inside: {

              rtbs.rbs.set_background_color(context, i, rtbs.rbs.m_store[i].m_properties.theme->button.hover_outline_color);
              rtbs.rbs.set_foreground_color(context, i, rtbs.rbs.m_store[i].m_properties.theme->button.hover_color);
              break;
            }
            default: { // outside, outside drag
              rtbs.rbs.set_background_color(context, i, rtbs.rbs.m_store[i].m_properties.theme->button.outline_color);
              rtbs.rbs.set_foreground_color(context, i, rtbs.rbs.m_store[i].m_properties.theme->button.color);
              break;
            }
          }
        }

        fan_2d::graphics::gui::src_dst_t get_cursor(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, uint32_t x, uint32_t y) {
          return rtbs.get_cursor(context, i, x, y);
        }

        bool locked(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) const
        {
          return m_reserved[i] & (uint32_t)button_states_e::locked;
        }

        void enable_draw(fan::window_t* window, fan::opengl::context_t* context)
        {
          rtbs.enable_draw(context);
          m_key_event.enable_draw(window, context);
        }

        void disable_draw(fan::window_t* window, fan::opengl::context_t* context)
        {
          rtbs.disable_draw(context);
          m_key_event.disable_draw(window, context);
        }

        void bind_matrices(fan::opengl::context_t* context, fan::opengl::matrices_t* matrices) {
          rtbs.bind_matrices(context, matrices);
          m_key_event.m_cursor.m_shader.bind_matrices(context, matrices);
        }

        fan::vec2 get_viewport_collision_offset() const {
          return viewport_collision_offset;
        }
        void set_viewport_collision_offset(const fan::vec2& offset) {
          viewport_collision_offset = offset;
        }

         // IO +

        void write_out(fan::opengl::context_t* context, fan::io::file::file_t* f) {
				  rtbs.write_out(context, f);
          uint64_t count = m_reserved.size() * sizeof(decltype(m_reserved)::value_type);
          fan::io::file::write(f, &count, sizeof(count), 1);
					fan::io::file::write(f, m_reserved.data(), sizeof(decltype(m_reserved)::value_type) * m_reserved.size(), 1);
          m_key_event.write_out(context, f);
			  }
        void write_in(fan::opengl::context_t* context, fan::io::file::file_t* f) {
          rtbs.write_in(context, f);
          uint64_t count;
          fan::io::file::read(f, &count, sizeof(count), 1);
          m_reserved.resize(count / sizeof(decltype(m_reserved)::value_type));
          fan::io::file::read(f, m_reserved.data(), count, 1);
          m_key_event.write_in(context, f);
			  }

        // IO -

        fan_2d::graphics::gui::rectangle_text_box_sized_t rtbs;
        fan_2d::graphics::gui::key_event_t<rectangle_text_button_sized_t> m_key_event;
        fan_2d::graphics::gui::button_event_t<rectangle_text_button_sized_t> m_button_event;

     // protected:

        fan::vec2 viewport_collision_offset;
        fan::hector_t<uint32_t> m_reserved;
      };
    }
  }
}