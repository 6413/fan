#pragma once

#include _FAN_PATH(graphics/gui/rectangle_box_sized.h)
#include _FAN_PATH(graphics/gui/focus.h)

namespace fan_2d {
  namespace graphics {
    namespace gui {
      struct rectangle_button_sized_t {

        rectangle_button_sized_t() = default;

        struct properties_t : public rectangle_box_sized_t::properties_t {

          using rectangle_box_sized_t::properties_t::properties_t;
          button_states_e button_state = button_states_e::clickable;
        };

        void open(fan::window_t* window, fan::opengl::context_t* context)
        {
          rbs.open(context);
          m_button_event.open(window, context);
          m_reserved.open();

          viewport_collision_offset = 0;
        }

        void close(fan::window_t* window, fan::opengl::context_t* context)
        {
          rbs.close(context);
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

          rbs.push_back(context, properties);

          if (inside(window, context, size(window, context) - 1, window->get_mouse_position()) && properties.button_state != button_states_e::locked) {
            m_button_event.m_focused_button_id = size(window, context) - 1;
            lib_add_on_mouse_move(window, context, m_button_event.m_focused_button_id, fan_2d::graphics::gui::mouse_stage::inside, m_button_event.mouse_user_ptr);
          }
        }

        uint32_t size(fan::window_t* window, fan::opengl::context_t* context) {
          return rbs.size(context);
        }

        void draw(fan::window_t* window, fan::opengl::context_t* context)
        {
          rbs.draw(context);
        }

        void erase(fan::window_t* window, fan::opengl::context_t* context, uint32_t i)
        {
          rbs.erase(context, i);
          m_reserved.erase(i);
        }

        void erase(fan::window_t* window, fan::opengl::context_t* context, uint32_t begin, uint32_t end)
        {
          rbs.erase(context, begin, end);
          m_reserved.erase(begin, end);
        }

        void clear(fan::window_t* window, fan::opengl::context_t* context)
        {
          rbs.clear(context);
          m_reserved.clear();

          // otherwise default add_inputs in constructor will be erased as well
          //rectangle_text_button_sized::mouse::clear();
          //rectangle_text_button_sized::text_input::clear();
        }

        void set_locked(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, bool flag, bool change_theme) {
          if (flag) {
            if (m_button_event.m_focused_button_id == i) {
              m_button_event.m_focused_button_id = fan::uninitialized;
            }
            m_reserved[i] |= (uint32_t)button_states_e::locked;
            if (change_theme) {
              *rbs.m_store[i].m_properties.theme = fan_2d::graphics::gui::themes::locked();
              rbs.update_theme(context, i);
            }
          }
          else {
            m_reserved[i] &= ~(uint32_t)button_states_e::locked;
            if (inside(window, context, i, window->get_mouse_position())) {
              m_button_event.m_focused_button_id = i;
            }
          }

        }

        void lib_add_on_input(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, uint16_t key, fan::key_state state, fan_2d::graphics::gui::mouse_stage stage, void* user_ptr)
        {
          if (this->locked(window, context, i)) {
            return;
          }

          if (rbs.m_store[i].m_properties.theme->button.m_click_callback) {
            rbs.m_store[i].m_properties.theme->button.m_click_callback(window, i, key, state, stage, user_ptr);
          }

          if (key != fan::mouse_left) {
            return;
          }

          switch (stage) {
            case fan_2d::graphics::gui::mouse_stage::inside: {

              switch (state) {
                case fan::key_state::press: {
                  rbs.set_background_color(context, i, rbs.m_store[i].m_properties.theme->button.click_outline_color);
                  rbs.set_foreground_color(context, i, rbs.m_store[i].m_properties.theme->button.click_color);
                  break;
                }
                case fan::key_state::release: {
                  focus::set_focus(focus::properties_t((void*)window->get_handle(), this, i));
                  rbs.set_background_color(context, i, rbs.m_store[i].m_properties.theme->button.hover_outline_color);
                  rbs.set_foreground_color(context, i, rbs.m_store[i].m_properties.theme->button.hover_color);
                  break;
                }
              }
              break;
            }
            case fan_2d::graphics::gui::mouse_stage::outside: {
              switch (state) {
                case fan::key_state::press: {
                  fan_2d::graphics::gui::focus::set_focus(fan_2d::graphics::gui::focus::no_focus);
                  break;
                }
                case fan::key_state::release: {
                  rbs.set_background_color(context, i, rbs.m_store[i].m_properties.theme->button.outline_color);
                  rbs.set_foreground_color(context, i, rbs.m_store[i].m_properties.theme->button.color);
                  for (int j = 0; j < rbs.size(context); j++) {
                    if (this->inside(window, context, j, window->get_mouse_position()) && !this->locked(window, context, j)) {
                      rbs.set_background_color(context, i, rbs.m_store[i].m_properties.theme->button.hover_outline_color);
                      rbs.set_foreground_color(context, i, rbs.m_store[i].m_properties.theme->button.hover_color);
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

          if (rbs.m_store[i].m_properties.theme->button.m_hover_callback) {
            rbs.m_store[i].m_properties.theme->button.m_hover_callback(window, i, stage, user_ptr);
          }

          switch (stage) {
            case mouse_stage::inside: {
              rbs.set_background_color(context, i, rbs.m_store[i].m_properties.theme->button.hover_outline_color);
              rbs.set_foreground_color(context, i, rbs.m_store[i].m_properties.theme->button.hover_color);
              break;
            }
            default: { // outside, outside drag
              rbs.set_background_color(context, i, rbs.m_store[i].m_properties.theme->button.outline_color);
              rbs.set_foreground_color(context, i, rbs.m_store[i].m_properties.theme->button.color);

              break;
            }
          }

        }

        bool locked(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) const
        {
          return m_reserved[i] & (uint32_t)button_states_e::locked;
        }

        void enable_draw(fan::window_t* window, fan::opengl::context_t* context)
        {
          rbs.enable_draw(context);
        }

        void disable_draw(fan::window_t* window, fan::opengl::context_t* context)
        {
          rbs.disable_draw(context);
        }

        bool inside(fan::window_t* window,  fan::opengl::context_t* context, uint32_t i, const fan::vec2& position) {
          return rbs.inside(context, i, position);
        }

        fan::vec2 get_viewport_collision_offset() const {
          return viewport_collision_offset;
        }
        void set_viewport_collision_offset(const fan::vec2& offset) {
          viewport_collision_offset = offset;
        }

        void bind_matrices(fan::opengl::context_t* context, fan::opengl::matrices_t* matrices) {
          rbs.bind_matrices(context, matrices);
        }

        fan::vec2 get_position(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) const {
          return rbs.get_position(context, i);
        }
        void set_position(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, const fan::vec2& position) {
          rbs.set_position(context, i, position);
        }

        fan_2d::graphics::gui::button_event_t<rectangle_button_sized_t> m_button_event;

        fan_2d::graphics::gui::rectangle_box_sized_t rbs;

      protected:

        fan::vec2 viewport_collision_offset;
        fan::hector_t<uint32_t> m_reserved;
      };
    }
  }
}