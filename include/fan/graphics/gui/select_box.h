#pragma once

#include _FAN_PATH(graphics/renderer.h)

#if fan_renderer == fan_renderer_opengl
  #include _FAN_PATH(graphics/gui/rectangle_text_button_sized.h)
#endif

#include _FAN_PATH(graphics/gui/themes.h)

namespace fan_2d {
  namespace graphics {
    namespace gui {

      struct select_box_t {

        typedef void(*checkbox_action_cb)(select_box_t*, fan::window_t*, uint32_t i, void* userptr);

        struct properties_t {
          fan::utf16_string text;
        };

        struct open_properties_t {
          fan::vec2 position;
          f32_t max_text_length;
          f32_t gui_size;
          fan_2d::graphics::gui::theme_t theme = fan_2d::graphics::gui::themes::deep_blue();
        };

        void open(fan::window_t* window, fan::opengl::context_t* context, const open_properties_t& p) {
          m_select_box.open(window, context);
          m_op = p;
          m_cb = [](select_box_t*, fan::window_t*, uint32_t i, void* userptr){};

          decltype(m_select_box)::properties_t sp;
          sp.position = p.position;
          sp.theme = p.theme;
          sp.size = fan::vec2(p.max_text_length + p.gui_size, p.gui_size);
          sp.text;
          sp.text.resize(1);

          m_select_box.push_back(window, context, sp);

          m_menu_open = 0;

          m_select_box.m_button_event.set_on_input(this, [](fan::window_t* window, fan::opengl::context_t* context, uint32_t i, uint16_t key, fan::key_state key_state, fan_2d::graphics::gui::mouse_stage mouse_stage, void* user_ptr) {
            if (key != fan::mouse_left) {
              return;
            }
            if (key_state != fan::key_state::release) {
              return;
            }
            if (mouse_stage != fan_2d::graphics::gui::mouse_stage::inside) {
              return;
            }
            auto thiS = (decltype(this))user_ptr;
            
            for (int j = 1; j < thiS->m_select_box.size(window, context); j++) {
              thiS->m_select_box.set_locked(window, context, j, 0, thiS->m_menu_open);
            }

            thiS->m_menu_open = !thiS->m_menu_open;

            if (i == 0) {
              return;
            }

            thiS->m_selected = i;
            thiS->m_select_box.set_text(window, context, 0, thiS->m_select_box.get_text(window, context, i));
            thiS->m_cb(thiS, window, i - 1, thiS->m_userptr);
          });
        }
        void close(fan::window_t* window, fan::opengl::context_t* context) {
          m_select_box.close(window, context);
        }

        void push_back(fan::window_t* window, fan::opengl::context_t* context, const properties_t& p) {
          decltype(m_select_box)::properties_t sp;
          fan::vec2 position = m_select_box.get_position(window, context, m_select_box.size(window, context) - 1);
          fan::vec2 size = m_select_box.get_size(window, context, 0);
          sp.position = fan::vec2(position.x, position.y + size.y * 2);
          sp.theme = m_op.theme;
          sp.size = size;
          sp.text = p.text;
          m_select_box.push_back(window, context, sp);
          m_select_box.set_locked(window, context, m_select_box.size(window, context) - 1, true, false);
        }

        void set_on_select_action(fan::window_t* window, fan::opengl::context_t* context, void* userptr, checkbox_action_cb cb) {
          m_cb = cb;
          m_userptr = userptr;
        }

        void enable_draw(fan::window_t* window, fan::opengl::context_t* context) {
          m_select_box.rtbs.rbs.m_box.m_draw_node_reference = context->enable_draw(this, [](fan::opengl::context_t* c, void* d) { 
            auto thiS = ((decltype(this))d);
            thiS->m_select_box.rtbs.rbs.m_box.draw(c, 0, thiS->m_menu_open ? thiS->m_select_box.rtbs.size(c) * 2 : 2);
            thiS->m_select_box.rtbs.tr.draw(c, 0, thiS->m_menu_open ? thiS->m_select_box.rtbs.tr.size(c) : 1);
          });
        }

     // protected:

        fan_2d::graphics::gui::rectangle_text_button_sized_t m_select_box;

        checkbox_action_cb m_cb;

        open_properties_t m_op;
        uint8_t m_menu_open;
        uint32_t m_selected;
        void* m_userptr;
      };

    }
  }
}