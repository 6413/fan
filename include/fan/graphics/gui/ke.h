#pragma once

#include _FAN_PATH(physics/collision/circle.h)
#include _FAN_PATH(graphics/opengl/gl_core.h)

namespace fan_2d {
  namespace graphics {
    namespace gui {
      struct ke_t {

        struct key_data_t {
          loco_t* loco;
          ke_t* key_event;
          void* element_id;
          void* userptr;
          uint32_t depth;
          uint16_t key;
          fan::key_state key_state;
        };

        typedef uint8_t(*on_input_cb_t)(const key_data_t&);

        struct properties_t {
          on_input_cb_t on_input_cb;

          void* userptr;
          void* cid;
        };

        void open() {
          m_button_data.open();
        }
        void close() {
          m_button_data.close();
        }

        uint32_t push_back(const properties_t& p, on_input_cb_t on_input_lib_cb = [](const key_data_t&) -> uint8_t { return 1; }) {
          button_data_t b;
          b.properties = p;
          b.on_input_lib_cb = on_input_lib_cb;
          return m_button_data.push_back(b);
        }

        uint32_t feed_keyboard(loco_t* loco, uint16_t key, fan::key_state key_state, uint32_t depth) {
          #define make_data(index) \
          key_data.loco = loco; \
          key_data.key_event = this; \
          key_data.element_id = m_button_data[index].properties.cid; \
          key_data.key = key; \
          key_data.key_state = key_state; \
          key_data.userptr = m_button_data[index].properties.userptr; \
          key_data.depth = depth; \
          m_button_data[index].on_input_lib_cb(key_data); \
          if (!m_button_data[index].properties.on_input_cb(key_data)) { \
            return 0; \
          }

          uint32_t it = m_button_data.rbegin();
          while (it != m_button_data.rend()) {
            key_data_t key_data;
            make_data(it);
            it = m_button_data.rnext(it);
          }

          return 1;
        }

        struct button_data_t {
          on_input_cb_t on_input_lib_cb;
          properties_t properties;
        };

        bll_t<button_data_t> m_button_data;
      };
    }
  }
}