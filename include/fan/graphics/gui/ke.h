#pragma once

#include _FAN_PATH(physics/collision/circle.h)
#include _FAN_PATH(graphics/opengl/gl_core.h)

namespace fan_2d {
  namespace graphics {
    namespace gui {
      struct ke_t {

        typedef void(*on_input_cb)(fan::opengl::context_t* context, ke_t*, fan::opengl::cid_t* object_index, uint16_t key, fan::key_state key_state, void* userptr);

        struct properties_t {
          on_input_cb on_input_function;

          void* userptr;
        };

        void open() {
          m_button_data.open();
        }
        void close() {
          m_button_data.close();
        }

        void push_back(fan::opengl::cid_t* cid, fan::opengl::cid_t* object_cid, const properties_t& p) {
          button_data_t b;
          b.properties = p;
          b.object_cid = object_cid;
          cid->id = m_button_data.push_back(b);
        }

        void feed_keyboard(fan::opengl::context_t* context, uint16_t key, fan::key_state key_state) {
          for (uint32_t i = 0; i < m_button_data.size(); i++) {
            m_button_data[i].properties.on_input_function(context, this, m_button_data[i].object_cid, key, key_state, 
              m_button_data[i].properties.userptr);
          }
        }

        struct button_data_t {
          properties_t properties;
          fan::opengl::cid_t* object_cid;
        };

        fan::hector_t<button_data_t> m_button_data;
      };
    }
  }
}