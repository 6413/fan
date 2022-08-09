#pragma once

#include _FAN_PATH(graphics/renderer.h)

#if fan_renderer == fan_renderer_opengl
  #include _FAN_PATH(graphics/opengl/2D/objects/line.h)
  #include _FAN_PATH(graphics/gui/rectangle_button_sized.h)
#endif

#include _FAN_PATH(graphics/gui/themes.h)

//namespace fan_2d {
//  namespace graphics {
//    namespace gui {
//
//      struct checkbox_t {
//
//        typedef void(*checkbox_action_cb)(checkbox_t*, fan::window_t*, uint32_t i, void* userptr);
//
//        struct properties_t {
//          fan::utf16_string text;
//          fan::vec2 text_position = 0;
//          fan::vec2 background_padding = 0;
//          fan_2d::graphics::gui::theme_t theme = fan_2d::graphics::gui::themes::deep_blue();
//          bool checked = false;
//        };
//
//        struct open_properties_t {
//          fan::vec2 position;
//          fan::color background_color;
//          f32_t max_text_length;
//          f32_t gui_size;
//        };
//
//        void open(fan::window_t* window, fan::opengl::context_t* context, const open_properties_t& p) {
//          m_checkbox_store.open();
//          m_check.open(context);
//          m_checkbox.open(window, context);
//          m_text.open(context);
//          m_background.open(context);
//          m_op = p;
//          fan_2d::graphics::rectangle_t::properties_t rp;
//          rp.position = p.position;
//          rp.size = 0;
//          rp.color = p.background_color;
//          m_background.push_back(context, rp);
//          m_cb = [](checkbox_t*, fan::window_t*, uint32_t i, void* userptr){};
//          m_checkbox.m_button_event.set_on_input(this, [](fan::window_t* window, fan::opengl::context_t* context, uint32_t i, uint16_t key, fan::key_state key_state, fan_2d::graphics::gui::mouse_stage mouse_stage, void* user_ptr) {
//            if (key != fan::mouse_left) {
//              return;
//            }
//            if (key_state != fan::key_state::release) {
//              return;
//            }
//            if (mouse_stage != fan_2d::graphics::gui::mouse_stage::inside) {
//              return;
//            }
//            auto thiS = (decltype(this))user_ptr;
//
//            thiS->m_checkbox_store[i].m_checked = !thiS->m_checkbox_store[i].m_checked;
//            if (thiS->m_checkbox_store[i].m_checked == 0) {
//              thiS->m_check.set_color(context, i * 2, fan::colors::transparent);
//              thiS->m_check.set_color(context, i * 2 + 1, fan::colors::transparent);
//            }
//            else {
//              thiS->m_check.set_color(context, i * 2, thiS->m_checkbox_store[i].m_theme.checkbox.check_color);
//              thiS->m_check.set_color(context, i * 2 + 1, thiS->m_checkbox_store[i].m_theme.checkbox.check_color);
//            }
//            thiS->m_cb((decltype(this))user_ptr, window, i, thiS->m_userptr);
//          });
//        }
//        void close(fan::window_t* window, fan::opengl::context_t* context) {
//          m_check.close(context);
//          m_checkbox.close(window, context);
//          m_text.close(context);
//        }
//
//        void push_back(fan::window_t* window, fan::opengl::context_t* context, const properties_t& p) {
//
//          f32_t y_offset = m_op.gui_size * (m_checkbox.size(window, context) + 1);
//
//          fan::vec2 text_size = fan_2d::opengl::gui::text_renderer_t::get_text_size(context, p.text, m_op.gui_size);
//          fan::vec2 box_size = m_op.gui_size / 2.5;
//          fan::vec2 bc_position = m_op.position + fan::vec2(0, m_op.gui_size * m_checkbox.size(window, context));
//          fan::vec2 bc_size = fan::vec2(m_op.gui_size * 2 + m_op.max_text_length / 2, y_offset);
//          fan::vec2 box_position = bc_position + fan::vec2(-m_op.max_text_length / 2 - m_op.gui_size / 2 , y_offset / 2);
//
//          m_background.set_position(
//            context,
//            0,
//            bc_position
//          );
//
//          m_background.set_size(
//            context,
//            0,
//            bc_size
//          );
//
//          fan_2d::graphics::line_t::properties_t lp;
//          lp.src = box_position + box_size;
//          lp.dst = box_position - box_size;
//          lp.color = p.checked ? p.theme.checkbox.check_color : fan::colors::transparent;
//          m_check.push_back(context, lp);
//          lp.src = box_position + fan::vec2(-box_size.x, box_size.y);
//          lp.dst = box_position + fan::vec2(box_size.x, -box_size.y);
//          m_check.push_back(context, lp);
//
//          fan_2d::graphics::gui::rectangle_button_sized_t::properties_t rp;
//          rp.position = box_position;
//          rp.size = box_size;
//          rp.theme = p.theme;
//          m_checkbox.push_back(window, context, rp);
//
//          fan_2d::opengl::gui::text_renderer_t::properties_t trp;
//          trp.position = box_position + fan::vec2(text_size.x / 2 + box_size.x * 2, 0);
//          trp.text_color = p.theme.checkbox.text_color;
//          trp.font_size = m_op.gui_size;
//          trp.text = p.text;
//          trp.outline_color = fan::colors::black;
//          trp.outline_size = 0.7;
//          m_text.push_back(context, trp);
//
//          selection_box_store_t sbs;
//          sbs.m_checked = p.checked;
//          sbs.m_theme = p.theme;
//          m_checkbox_store.push_back(sbs);
//        }
//
//        void set_on_check_action(fan::window_t* window, fan::opengl::context_t* context, void* userptr, checkbox_action_cb cb) {
//          m_userptr = userptr;
//          m_cb = cb;
//        }
//
//        void enable_draw(fan::window_t* window, fan::opengl::context_t* context) {
//          m_background.enable_draw(context);
//          m_checkbox.enable_draw(window, context);
//          m_check.enable_draw(context);
//          m_text.enable_draw(context);
//        }
//
//     // protected:
//
//        fan_2d::graphics::line_t m_check;
//        fan_2d::graphics::gui::rectangle_button_sized_t m_checkbox;
//        fan_2d::opengl::gui::text_renderer_t m_text;
//        fan_2d::opengl::rectangle_t m_background;
//
//        struct selection_box_store_t {
//          uint8_t m_checked;
//          fan_2d::graphics::gui::theme_t m_theme;
//        };
//
//        fan::hector_t<selection_box_store_t> m_checkbox_store;
//
//        checkbox_action_cb m_cb;
//
//        open_properties_t m_op;
//        void* m_userptr;
//      };
//
//    }
//  }
//}