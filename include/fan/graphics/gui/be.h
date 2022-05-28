#pragma once

#include <fan/graphics/gui/rectangle_text_box_sized.h>
#include <fan/graphics/gui/button_event.h>
#include <fan/graphics/gui/key_event.h>

namespace fan_2d {
  namespace graphics {
    namespace gui {
      struct be_t {

        typedef void(*on_input_cb)(be_t*, uint32_t index, uint16_t key, fan::key_state key_state, fan_2d::graphics::gui::mouse_stage mouse_stage);
        typedef void(*on_mouse_move_cb)(be_t*, uint32_t index, mouse_stage mouse_stage);

        struct properties_t {
          fan::vec2 position;
          fan::vec2 size;
        };

        struct key_info_t {
          be_t* be;
          uint32_t index;
          uint16_t key;
          fan::key_state key_state;
          fan_2d::graphics::gui::mouse_stage mouse_stage;
        };
        struct mouse_info_t {
          be_t* be;
          uint32_t index;
          uint16_t key;
          fan::key_state key_state;
          fan_2d::graphics::gui::mouse_stage mouse_stage;
        };

        void open() {
          m_button_data.open();
          coordinate_offset = 0;
        }
        void close() {
          m_button_data.close();
        }

        void bind_to_window(fan::window_t* w) {
          m_old_mouse_stage = fan::uninitialized;
          m_do_we_hold_button = 0;
          m_focused_button_id = fan::uninitialized;
          add_keys_callback_id = fan::uninitialized;
          add_mouse_move_callback_id = fan::uninitialized;
          on_input_function = [](be_t*, uint32_t index, uint16_t key, fan::key_state key_state, fan_2d::graphics::gui::mouse_stage mouse_stage) {};
          on_mouse_event_function = [](be_t*, uint32_t index, fan_2d::graphics::gui::mouse_stage mouse_stage) {};

          static auto inside = [](be_t* object, uint32_t i, const fan::vec2& p) {
            fan::vec2 src = object->m_button_data[i].properties.position - object->m_button_data[i].properties.size;
            fan::vec2 dst = object->m_button_data[i].properties.position + object->m_button_data[i].properties.size;
            return fan_2d::collision::rectangle::point_inside_no_rotation(p, src, dst);
          };

          add_mouse_move_callback_id = w->add_mouse_move_callback(this, [](fan::window_t* w, const fan::vec2i&, void* user_ptr) {
            be_t* object = (be_t*)user_ptr;

            if (object->m_do_we_hold_button == 1) {
              return;
            }
            if (object->m_focused_button_id != fan::uninitialized) {
              if (inside(object, object->m_focused_button_id, object->coordinate_offset + w->get_mouse_position())) {
                return;
              }
            }

            for (int i = object->size(); i--; ) {
              if (inside(object, i, object->coordinate_offset + w->get_mouse_position())) {
                if (object->m_focused_button_id != fan::uninitialized) {
                  object->on_mouse_event_function(object, object->m_focused_button_id, mouse_stage::outside);
                }
                object->m_focused_button_id = i;
                object->on_mouse_event_function(object, object->m_focused_button_id, mouse_stage::inside);
                return;
              }
            }
            if (object->m_focused_button_id != fan::uninitialized) {
              object->on_mouse_event_function(object, object->m_focused_button_id, mouse_stage::outside);
              object->m_focused_button_id = fan::uninitialized;
            }
          });

          add_keys_callback_id = w->add_keys_callback(this, [](fan::window_t* w, uint16_t key, fan::key_state state, void* user_ptr) {
            be_t* object = (be_t*)user_ptr;

            if (object->m_do_we_hold_button == 0) {
              if (state == fan::key_state::press) {
                if (object->m_focused_button_id != fan::uninitialized) {
                  object->m_do_we_hold_button = 1;
                  object->on_input_function(object, object->m_focused_button_id, key, fan::key_state::press, mouse_stage::inside);
                }
                else {
                  for (int i = object->size(); i--; ) {
                  if (inside(object, i, object->coordinate_offset + w->get_mouse_position())) {
                      object->on_input_function(object, i, key, state, mouse_stage::outside);
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
                if (inside(object, object->m_focused_button_id, object->coordinate_offset + w->get_mouse_position())) {
                  pointer_remove_flag = 1;
                  object->on_input_function(object, object->m_focused_button_id, key, fan::key_state::release, mouse_stage::inside);
                  if (pointer_remove_flag == 0) {
                    return;
                    //rtb is deleted
                  }
                }
                else {
                  for (int i = object->size(); i--; ) {
                    if (inside(object, i, object->coordinate_offset + w->get_mouse_position())) {
                      object->on_input_function(object, i, key, fan::key_state::release, mouse_stage::inside_drag);
                      object->m_focused_button_id = i;
                      break;
                    }
                  }

                  pointer_remove_flag = 1;
                  object->on_input_function(object, object->m_focused_button_id, key, fan::key_state::release, mouse_stage::outside);
                  if (pointer_remove_flag == 0) {
                    return;
                  }

                  pointer_remove_flag = 0;
                }
                object->m_do_we_hold_button = 0;
              }
            }
          });
        }
        void unbind_from_window(fan::window_t* window) {
          if (add_keys_callback_id != fan::uninitialized) {
            window->remove_keys_callback(add_keys_callback_id);
            add_keys_callback_id = fan::uninitialized;
          }
          if (add_mouse_move_callback_id != -1) {
            window->remove_mouse_move_callback(add_mouse_move_callback_id);
            add_mouse_move_callback_id = fan::uninitialized;
          }
        }

        void* get_userptr() const {
          return m_userptr;
        }
        void set_userptr(void* userptr) {
          m_userptr = userptr;
        }

        void push_back(const properties_t& p) {
          button_data_t b;
          b.properties = p;
          m_button_data.push_back(b);
        }

        // used for camera position
        fan::vec2 get_coordinate_offset() const {
          return coordinate_offset;
        }
        void set_coordinate_offset(const fan::vec2& offset) {
          coordinate_offset = offset;
        }

        uint32_t size() const {
          return m_button_data.size();
        }

        void set_on_input(on_input_cb function) {
          on_input_function = function;
        }

        void set_on_mouse_event(on_mouse_move_cb function) {
          on_mouse_event_function = function;
        }

        on_input_cb on_input_function;
        on_mouse_move_cb on_mouse_event_function;

        inline static thread_local bool pointer_remove_flag;

        uint8_t m_old_mouse_stage;
        bool m_do_we_hold_button;
        uint32_t m_focused_button_id;
        uint32_t add_keys_callback_id;
        uint32_t add_mouse_move_callback_id;
        void* key_user_ptr;
        void* mouse_user_ptr;

        struct button_data_t {
          properties_t properties;
          uint16_t old_key;

        };

        fan::hector_t<button_data_t> m_button_data;

        void* m_userptr;
        fan::vec2 coordinate_offset;

        

        /*union {
          struct{
            fan::window_t* window;
          }window;

        }bind_data;*/
      };
    }
  }
}