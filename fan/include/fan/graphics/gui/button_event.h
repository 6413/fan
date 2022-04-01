#pragma once

#include <fan/window/window.h>
#include <fan/graphics/opengl/gl_core.h>

namespace fan_2d {
  namespace graphics {
    namespace gui {

      enum class mouse_stage {
        outside,
        inside,
        outside_drag,
        inside_drag // when dragged from other element and released inside other element
      };

      // requires to have functions: 
      // size() returns amount of objects, 
      // inside() if mouse inside
      // requires push_back to be called in every shape push_back
      // requires to have either lib_add_on_input and lib_add_on_mouse_event

      template <typename T>
      struct button_event_t {

        typedef void(*on_input_cb)(fan::window_t* window, fan::opengl::context_t* context, uint32_t index, uint16_t key, fan::key_state key_state, mouse_stage mouse_stage, void* user_ptr);
        typedef void(*on_mouse_move_cb)(fan::window_t* window, fan::opengl::context_t* context, uint32_t index, mouse_stage mouse_stage, void* user_ptr);

        button_event_t() = default;

        void open(fan::window_t* window, fan::opengl::context_t* context) {
          m_old_mouse_stage = fan::uninitialized;
          m_do_we_hold_button = 0;
          m_focused_button_id = fan::uninitialized;
          add_keys_callback_id = fan::uninitialized;
          add_mouse_move_callback_id = fan::uninitialized;
          on_mouse_event_function = [](fan::window_t* window, fan::opengl::context_t* context, uint32_t index, mouse_stage mouse_stage, void* user_ptr) {};
          on_input_function = [](fan::window_t* window, fan::opengl::context_t* context, uint32_t index, uint16_t key, fan::key_state key_state, mouse_stage mouse_stage, void* user_ptr) {};
          m_context = context;

          add_mouse_move_callback_id = window->add_mouse_move_callback(this, [](fan::window_t* w, const fan::vec2i&, void* user_ptr) {
            // thanks to visual studio's internal error
            T* object = fan::offsetless(user_ptr, &T::m_button_event);
            
            fan::opengl::context_t* context = (fan::opengl::context_t*)object->m_button_event.m_context;

            if (object->m_button_event.m_do_we_hold_button == 1) {
              return;
            }
            if (object->m_button_event.m_focused_button_id != fan::uninitialized) {

              if (object->m_button_event.m_focused_button_id >= object->size(w, context)) {
                object->m_button_event.m_focused_button_id = fan::uninitialized;
              }
              else if (object->inside(w, context, object->m_button_event.m_focused_button_id, fan::vec2(context->camera.get_position()) + w->get_mouse_position()) || object->locked(w, context, object->m_button_event.m_focused_button_id)) {
                return;
              }
            }

            for (int i = object->size(w, context); i--; ) {
              if (object->inside(w, context, i, fan::vec2(context->camera.get_position()) + w->get_mouse_position()) && !object->locked(w, context, i)) {
                if (object->m_button_event.m_focused_button_id != fan::uninitialized) {
                  object->lib_add_on_mouse_move(w, context, object->m_button_event.m_focused_button_id, mouse_stage::outside, object->m_button_event.mouse_user_ptr);
                  object->m_button_event.on_mouse_event_function(w, context, object->m_button_event.m_focused_button_id, mouse_stage::outside, object->m_button_event.mouse_user_ptr);
                }
                object->m_button_event.m_focused_button_id = i;
                object->lib_add_on_mouse_move(w, context, object->m_button_event.m_focused_button_id, mouse_stage::inside, object->m_button_event.mouse_user_ptr);
                object->m_button_event.on_mouse_event_function(w, context, object->m_button_event.m_focused_button_id, mouse_stage::inside, object->m_button_event.mouse_user_ptr);
                return;
              }
            }
            if (object->m_button_event.m_focused_button_id != fan::uninitialized) {
              object->lib_add_on_mouse_move(w, context, object->m_button_event.m_focused_button_id, mouse_stage::outside, object->m_button_event.mouse_user_ptr);
              object->m_button_event.on_mouse_event_function(w, context, object->m_button_event.m_focused_button_id, mouse_stage::outside, object->m_button_event.mouse_user_ptr);
              object->m_button_event.m_focused_button_id = fan::uninitialized;
            }
          });

          add_keys_callback_id = window->add_keys_callback(this, [](fan::window_t* w, uint16_t key, fan::key_state state, void* user_ptr) {
            T* object = fan::offsetless<T>(user_ptr, &T::m_button_event);
            fan::opengl::context_t* context = (fan::opengl::context_t*)object->m_button_event.m_context;

            if (object->m_button_event.m_focused_button_id >= object->size(w, context)) {
              object->m_button_event.m_focused_button_id = fan::uninitialized;
            }

            if (object->m_button_event.m_do_we_hold_button == 0) {
              if (state == fan::key_state::press) {
                if (object->m_button_event.m_focused_button_id != fan::uninitialized) {
                  object->m_button_event.m_do_we_hold_button = 1;
                  object->lib_add_on_input(w, context, object->m_button_event.m_focused_button_id, key, fan::key_state::press, mouse_stage::inside, object->m_button_event.key_user_ptr);
                  object->m_button_event.on_input_function(w, context, object->m_button_event.m_focused_button_id, key, fan::key_state::press, mouse_stage::inside, object->m_button_event.key_user_ptr);
                }
                else {
                  for (int i = object->size(w, context); i--; ) {
                    object->lib_add_on_input(w, context, i, key, state, mouse_stage::outside, object->m_button_event.key_user_ptr);
                    object->m_button_event.on_input_function(w, context, i, key, state, mouse_stage::outside, object->m_button_event.key_user_ptr);
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
                if (object->m_button_event.m_focused_button_id >= object->size(w, context)) {
                  object->m_button_event.m_focused_button_id = fan::uninitialized;
                }
                else if (object->inside(w, context, object->m_button_event.m_focused_button_id, fan::vec2(context->camera.get_position()) + w->get_mouse_position()) && !object->locked(w, context, object->m_button_event.m_focused_button_id)) {
                  object->lib_add_on_input(w, context, object->m_button_event.m_focused_button_id, key, fan::key_state::release, mouse_stage::inside, object->m_button_event.key_user_ptr);
                  pointer_remove_flag = 1;
                  object->m_button_event.on_input_function(w, context, object->m_button_event.m_focused_button_id, key, fan::key_state::release, mouse_stage::inside, object->m_button_event.key_user_ptr);
                  if (pointer_remove_flag == 0) {
                    return;
                    //rtb is deleted
                  }
                }
                else {
                  object->lib_add_on_input(w, context, object->m_button_event.m_focused_button_id, key, fan::key_state::release, mouse_stage::outside, object->m_button_event.key_user_ptr);

                  for (int i = object->size(w, context); i--; ) {
                    if (object->inside(w, context, i, fan::vec2(context->camera.get_position()) + w->get_mouse_position()) && !object->locked(w, context, i)) {
                      object->lib_add_on_input(w, context, i, key, fan::key_state::release, mouse_stage::inside_drag, object->m_button_event.key_user_ptr);
                      object->m_button_event.m_focused_button_id = i;
                      break;
                    }
                  }

                  pointer_remove_flag = 1;
                  object->m_button_event.on_input_function(w, context, object->m_button_event.m_focused_button_id, key, fan::key_state::release, mouse_stage::outside, object->m_button_event.key_user_ptr);
                  if (pointer_remove_flag == 0) {
                    return;
                    //rtb is deleted
                  }

                  pointer_remove_flag = 0;
                }
                object->m_button_event.m_do_we_hold_button = 0;
              }
            }
          });
        }

        void close(fan::window_t* window, fan::opengl::context_t* context) {
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
        void set_on_input(void* user_ptr, on_input_cb function) {
          key_user_ptr = user_ptr;
          on_input_function = function;
        }

        // uint32_t index, bool inside
        void set_on_mouse_event(void* user_ptr, on_mouse_move_cb function) {
          mouse_user_ptr = user_ptr;
          on_mouse_event_function = function;
        }

        uint32_t holding_button() const {
          return m_focused_button_id;
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
        fan::opengl::context_t* m_context;
      };
    }
  }
}