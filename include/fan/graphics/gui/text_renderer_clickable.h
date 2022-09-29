#pragma once

#include _FAN_PATH(graphics/renderer.h)

#include _FAN_PATH(graphics/gui/button_event.h)

//namespace fan_2d {
//  namespace graphics {
//    namespace gui {
//      // text color needs to be less than 1.0 or 255 in rgb to see color change
//      struct text_renderer_clickable_t :
//        public fan_2d::opengl::gui::text_renderer_t
//      {
//
//        struct properties_t : public text_renderer_t::properties_t {
//          fan::vec2 hitbox_position; // middle
//          fan::vec2 hitbox_size; // half
//        };
//
//        void open(fan::window_t* window, fan::opengl::context_t* context) {
//          text_renderer_t::open(context);
//          m_button_event.open(window, context);
//          m_store.open();
//
//          viewport_collision_offset = 0;
//        }
//
//        void close(fan::window_t* window, fan::opengl::context_t* context) {
//          text_renderer_t::close(context);
//          m_button_event.close(window, context);
//          m_store.close();
//          if (decltype(m_button_event)::pointer_remove_flag == 1) {
//            decltype(m_button_event)::pointer_remove_flag = 0;
//          }
//        }
//
//        void lib_add_on_input(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, uint16_t key, fan::keyboard_state state, fan_2d::graphics::gui::mouse_stage stage, void* user_ptr) {
//
//          if (key != fan::mouse_left) {
//            return;
//          }
//
//          switch (stage) {
//            case mouse_stage::inside: {
//              switch (state) {
//                case fan::keyboard_state::press: {
//
//                  if (m_store[i].previous_states == 2) {
//                    text_renderer_t::set_text_color(context, i, text_renderer_t::get_text_color(context, i) + (click_strength - hover_strength));
//                    fan::color c;
//                    if ((c = text_renderer_t::get_outline_color(context, i)) != 0) {
//                      text_renderer_t::set_outline_color(context, i, text_renderer_t::get_outline_color(context, i) + (click_strength - hover_strength));
//                    }
//                  }
//                  else {
//                    text_renderer_t::set_text_color(context, i, text_renderer_t::get_text_color(context, i) + click_strength);
//                    fan::color c;
//                    if ((c = text_renderer_t::get_outline_color(context, i)) != 0) {
//                      text_renderer_t::set_outline_color(context, i, text_renderer_t::get_outline_color(context, i) + click_strength);
//                    }
//                  }
//
//                  m_store[i].previous_states = 1;
//                  // click
//                  break;
//                }
//                case fan::keyboard_state::release: {
//
//                  text_renderer_t::set_text_color(context, i, text_renderer_t::get_text_color(context, i) - (click_strength - hover_strength));
//                  fan::color c;
//                  if ((c = text_renderer_t::get_outline_color(context, i)) != 0) {
//                    text_renderer_t::set_outline_color(context, i, text_renderer_t::get_outline_color(context, i) - (click_strength - hover_strength));
//                  }
//
//                  m_store[i].previous_states = 2;
//                  // hover
//                  break;
//                }
//              }
//              break;
//            }
//            case mouse_stage::outside: {
//
//              switch (state) {
//                case fan::keyboard_state::release: {
//
//                  text_renderer_t::set_text_color(context, i, text_renderer_t::get_text_color(context, i) - click_strength);
//                  fan::color c;
//                  if ((c = text_renderer_t::get_outline_color(context, i)) != 0) {
//                    text_renderer_t::set_outline_color(context, i, text_renderer_t::get_outline_color(context, i) - click_strength);
//                  }
//
//                  m_store[i].previous_states = 0;
//
//                  // return original
//                  break;
//                }
//              }
//
//              break;
//            }
//            case mouse_stage::inside_drag: {
//
//              if (m_store[i].previous_states == 2) {
//                return;
//              }
//
//              text_renderer_t::set_text_color(context, i, text_renderer_t::get_text_color(context, i) + hover_strength);
//              fan::color c;
//              if ((c = text_renderer_t::get_outline_color(context, i)) != 0) {
//                text_renderer_t::set_outline_color(context, i, text_renderer_t::get_outline_color(context, i) + hover_strength);
//              }
//
//              m_store[i].previous_states = 2;
//
//              break;
//            }
//          }
//
//        }
//
//        void lib_add_on_mouse_move(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, fan_2d::graphics::gui::mouse_stage stage, void* user_ptr) {
//
//          switch (stage) {
//            case mouse_stage::inside: {
//
//              if (m_store[i].previous_states == 2) {
//                return;
//              }
//
//              text_renderer_t::set_text_color(context, i, text_renderer_t::get_text_color(context, i) + hover_strength);
//              fan::color c;
//              if ((c = text_renderer_t::get_outline_color(context, i)) != 0) {
//                text_renderer_t::set_outline_color(context, i, text_renderer_t::get_outline_color(context, i) + hover_strength);
//              }
//
//              m_store[i].previous_states = 2;
//
//              break;
//            }
//            default: { // outside, outside drag
//
//              if (m_button_event.holding_button() == (uint32_t)-1 || m_store[i].previous_states == 0) {
//                return;
//              }
//
//              switch (m_store[i].previous_states) {
//                case 1: {
//                  text_renderer_t::set_text_color(context, i, text_renderer_t::get_text_color(context, i) - click_strength);
//                  fan::color c;
//                  if ((c = text_renderer_t::get_outline_color(context, i)) != 0) {
//                    text_renderer_t::set_outline_color(context, i, text_renderer_t::get_outline_color(context, i) - click_strength);
//                  }
//                  break;
//                }
//                case 2: {
//                  text_renderer_t::set_text_color(context, i, text_renderer_t::get_text_color(context, i) - hover_strength);
//                  fan::color c;
//                  if ((c = text_renderer_t::get_outline_color(context, i)) != 0) {
//                    text_renderer_t::set_outline_color(context, i, text_renderer_t::get_outline_color(context, i) - hover_strength);
//                  }
//                  break;
//                }
//              }
//
//              m_store[i].previous_states = 0;
//
//              break;
//            }
//          }
//        }
//
//        void push_back(fan::window_t* window, fan::opengl::context_t* context, const text_renderer_clickable_t::properties_t& properties)
//        {
//          store_t store;
//          store.m_hitbox = hitbox_t{ properties.hitbox_position, properties.hitbox_size };
//          store.previous_states = 0;
//          m_store.push_back(store);
//
//          text_renderer_t::push_back(context, properties);
//        }
//
//        void set_hitbox(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, const fan::vec2& hitbox_position, const fan::vec2& hitbox_size)
//        {
//          m_store[i].m_hitbox.hitbox_position = hitbox_position;
//          m_store[i].m_hitbox.hitbox_size = hitbox_size;
//        }
//
//        fan::vec2 get_hitbox_position(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) const
//        {
//          return m_store[i].m_hitbox.hitbox_position;
//        }
//
//        void set_hitbox_position(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, const fan::vec2& hitbox_position)
//        {
//          m_store[i].m_hitbox.hitbox_position = hitbox_position;
//        }
//
//        fan::vec2 get_hitbox_size(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) const
//        {
//          return m_store[i].m_hitbox.hitbox_size;
//        }
//
//        void set_hitbox_size(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, const fan::vec2& hitbox_size)
//        {
//          m_store[i].m_hitbox.hitbox_size = hitbox_size;
//        }
//
//        uint32_t size(fan::window_t* window, fan::opengl::context_t* context) const {
//          return fan_2d::opengl::gui::text_renderer_t::size(context);
//        }
//
//        bool inside(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, const fan::vec2& p) const
//        {
//          const fan::vec2 src = m_store[i].m_hitbox.hitbox_position - m_store[i].m_hitbox.hitbox_size;
//          const fan::vec2 dst = m_store[i].m_hitbox.hitbox_position + m_store[i].m_hitbox.hitbox_size;
//
//          return fan_2d::collision::rectangle::point_inside_no_rotation(p, src, dst);
//        }
//        bool locked(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) const { return 0; }
//
//        fan::vec2 get_viewport_collision_offset() const {
//          return viewport_collision_offset;
//        }
//        void set_viewport_collision_offset(const fan::vec2& offset) {
//          viewport_collision_offset = offset;
//        }
//
//        static constexpr f32_t hover_strength = 0.2;
//        static constexpr f32_t click_strength = 0.3;
//
//        fan_2d::graphics::gui::button_event_t<text_renderer_clickable_t> m_button_event;
//
//        void erase(fan::window_t* window, fan::opengl::context_t* context, uintptr_t i) {
//          m_store.erase(i);
//          text_renderer_t::erase(context, i);
//        }
//
//        fan::vec2 get_position(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) const {
//          return text_renderer_t::get_position(context, i);
//        }
//        void set_position(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, const fan::vec2& position) {
//          fan::vec2 offset = position - text_renderer_t::get_position(context, i);
//          text_renderer_t::set_position(context, i, position);
//          m_store[i].m_hitbox.hitbox_position += offset;
//        }
//
//        fan::vec2 get_size(fan::window_t* window, fan::opengl::context_t* context, uint32_t i) const {
//          return m_store[i].m_hitbox.hitbox_size;
//        }
//        void set_size(fan::window_t* window, fan::opengl::context_t* context, uint32_t i, const fan::vec2& size) {
//          m_store[i].m_hitbox.hitbox_size = size;
//        }
//
//      protected:
//
//        using text_renderer_t::insert;
//
//        struct hitbox_t {
//          fan::vec2 hitbox_position; // middle
//          fan::vec2 hitbox_size; // half
//        };
//
//        struct store_t {
//          hitbox_t m_hitbox;
//          uint8_t previous_states;
//        };
//
//        fan::vec2 viewport_collision_offset;
//
//        fan::hector_t<store_t> m_store;
//      };
//    }
//  }
//}