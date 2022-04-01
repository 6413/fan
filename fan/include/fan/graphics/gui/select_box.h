#pragma once

#include <fan/graphics/renderer.h>

#if fan_renderer == fan_renderer_opengl
  #include <fan/graphics/gui/rectangle_text_button_sized.h>
#endif

#include <fan/graphics/gui/themes.h>

namespace fan_2d {
  namespace graphics {
    namespace gui {

      using rectangle_backend_t = fan_2d::opengl::rectangle_t;
      using text_renderer_backend_t = fan_2d::opengl::gui::text_renderer_t;

      struct select_box_t {

        typedef void(*checkbox_action_cb)(select_box_t*, fan::window_t*, uint32_t i);

        struct properties_t {
          fan::utf16_string text;
          fan::vec2 text_position = 0;
          fan::vec2 background_padding = 0;
          uint32_t m_selected = 0;
        };

        struct open_properties_t {
          fan::vec2 position;
          f32_t max_text_length;
          f32_t gui_size;
          fan_2d::graphics::gui::theme theme = fan_2d::graphics::gui::themes::deep_blue();
        };

        void open(fan::window_t* window, fan::opengl::context_t* context, const open_properties_t& p) {
          m_selection_box_store.open();
          m_select_box.open(window, context);
          m_op = p;

          decltype(m_select_box)::properties_t sp;
          sp.position = p.position;
          sp.theme = p.theme;
          sp.size = p.max_text_length + p.gui_size;
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
            thiS->m_menu_open = thiS->m_menu_open;
          });
        }
        void close(fan::window_t* window, fan::opengl::context_t* context) {
          m_select_box.close(window, context);
        }

        void push_back(fan::window_t* window, fan::opengl::context_t* context, const properties_t& p) {

        }

      /*  void on_check_action(fan::window_t* window, fan::opengl::context_t* context, checkbox_action_cb cb) {
          c_cb = cb;
          m_select_box.m_button_event.set_on_input(this, [](fan::window_t* window, fan::opengl::context_t* context, uint32_t i, uint16_t key, fan::key_state key_state, fan_2d::graphics::gui::mouse_stage mouse_stage, void* user_ptr) {
            if (key != fan::key_left) {
              return;
            }
            if (key_state != fan::key_state::release) {
              return;
            }
            if (mouse_stage != fan_2d::graphics::gui::mouse_stage::inside) {
              return;
            }
            thiS->m_selection_box_store[i].m_checked = !thiS->m_selection_box_store[i].m_checked;
            if (thiS->m_selection_box_store[i].m_checked == 0) {
              thiS->m_check.set_color(context, i * 2, fan::colors::transparent);
              thiS->m_check.set_color(context, i * 2 + 1, fan::colors::transparent);
            }
            else {
              thiS->m_check.set_color(context, i * 2, thiS->m_selection_box_store[i].m_theme.checkbox.check_color);
              thiS->m_check.set_color(context, i * 2 + 1, thiS->m_selection_box_store[i].m_theme.checkbox.check_color);
            }
            thiS->c_cb((decltype(this))user_ptr, window, i);
          });
        }*/

        void enable_draw(fan::window_t* window, fan::opengl::context_t* context) {
          m_select_box.rtbs.rbs.m_box.m_draw_node_reference = context->enable_draw(this, [](fan::opengl::context_t* c, void* d) { 
            auto thiS = ((decltype(this))d);
            thiS->m_select_box.rtbs.rbs.m_box.draw(c, 0, thiS->m_menu_open ? thiS->m_select_box.rtbs.size(c) : 1);
            thiS->m_select_box.rtbs.tr.draw(c, 0, thiS->m_menu_open ? thiS->m_select_box.rtbs.size(c) : 1);
          });
        }

     // protected:

        fan_2d::graphics::gui::rectangle_text_button_sized_t m_select_box;

        struct selection_box_store_t {
          uint32_t m_selected;
          fan_2d::graphics::gui::theme m_theme;
        };

        fan::hector_t<selection_box_store_t> m_selection_box_store;

        checkbox_action_cb c_cb;

        open_properties_t m_op;
        uint8_t m_menu_open;
      };

    }
  }
}