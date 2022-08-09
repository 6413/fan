#pragma once

#include _FAN_PATH(graphics/gui/key_event.h)
#include _FAN_PATH(physics/collision/circle.h)
#include _FAN_PATH(physics/collision/rectangle.h)

namespace fan_2d {
  namespace graphics {
    namespace gui {

      enum class mouse_stage_e {
        outside,
        inside,
        outside_drag,
        inside_drag // when dragged from other element and released inside other element
      };

      struct be_t {

        //using user_global_data_t = T_user_global_data;

        struct mouse_move_data_t {
          fan::opengl::context_t* context;
          be_t* button_event;
          void* element_id;
          mouse_stage_e mouse_stage;
          void* userptr[3];
          uint32_t depth;
        };

        struct mouse_input_data_t : mouse_move_data_t{
          uint16_t key;
          fan::key_state key_state;
        };

        typedef uint8_t(*on_input_cb_t)(const mouse_input_data_t&);
        typedef uint8_t(*on_mouse_move_cb_t)(const mouse_move_data_t&);

        struct hitbox_type_t {
          static constexpr uint8_t rectangle = 0;
          static constexpr uint8_t circle = 1;
        };

        struct properties_t {

          using type_t = be_t;

          on_input_cb_t on_input_function;
          on_mouse_move_cb_t on_mouse_event_function;

          void* userptr[3];

          void* cid;

          uint8_t hitbox_type;
          union {
            struct{
              fan::vec2 position;
              fan::vec2 size;
            }hitbox_rectangle;
            struct{
              fan::vec2 position;
              f32_t radius;
            }hitbox_circle;
          };
        };

        struct key_info_t {
          be_t* be;
          uint32_t index;
          uint16_t key;
          fan::key_state key_state;
          fan_2d::graphics::gui::mouse_stage_e mouse_stage;
        };

        void open() {
          m_button_data.open();
          coordinate_offset = 0;
          m_old_mouse_stage = fan::uninitialized;
          m_do_we_hold_button = 0;
          m_focused_button_id = fan::uninitialized;
        }
        void close() {
          m_button_data.close();
        }

        auto inside(uint32_t i, const fan::vec2& p) {

          switch(m_button_data[i].properties.hitbox_type) {
            case hitbox_type_t::rectangle: {
              return fan_2d::collision::rectangle::point_inside_no_rotation(
                p, 
                m_button_data[i].properties.hitbox_rectangle.position - m_button_data[i].properties.hitbox_rectangle.size, 
                m_button_data[i].properties.hitbox_rectangle.position + m_button_data[i].properties.hitbox_rectangle.size
              );
            }
            case hitbox_type_t::circle: {
              return fan_2d::collision::circle::point_inside(
                p, 
                m_button_data[i].properties.hitbox_circle.position,
                m_button_data[i].properties.hitbox_circle.radius
              );
            }
          }
        };

        uint8_t feed_mouse_move(fan::opengl::context_t* context, const fan::vec2& mouse_position, uint32_t depth) {
          if (m_do_we_hold_button == 1) {
            return 1;
          }
          if (m_focused_button_id != fan::uninitialized) {
            if (inside(m_focused_button_id, coordinate_offset + mouse_position)) {
              return 1;
            }
          }

          uint32_t i = m_button_data.rbegin();
          while (i != m_button_data.rend()) {
            if (inside(i, coordinate_offset + mouse_position)) {
              if (m_focused_button_id != fan::uninitialized) {
                mouse_move_data_t mm_data;
                mm_data.context = context;
                mm_data.button_event = this;
                mm_data.element_id = m_button_data[m_focused_button_id].properties.cid;
                mm_data.mouse_stage = mouse_stage_e::outside;
                mm_data.userptr[0] = m_button_data[m_focused_button_id].properties.userptr[0];
                mm_data.userptr[1] = m_button_data[m_focused_button_id].properties.userptr[1];
                mm_data.userptr[2] = m_button_data[m_focused_button_id].properties.userptr[2];
                mm_data.depth = depth;
                m_button_data[m_focused_button_id].on_mouse_move_lib_cb(mm_data);
                m_button_data[m_focused_button_id].properties.on_mouse_event_function(mm_data);
              }
              m_focused_button_id = i;

              mouse_move_data_t mm_data;
              mm_data.context = context;
              mm_data.button_event = this;
              mm_data.element_id = m_button_data[m_focused_button_id].properties.cid;
              mm_data.mouse_stage = mouse_stage_e::inside;
              mm_data.userptr[0] = m_button_data[m_focused_button_id].properties.userptr[0];
              mm_data.userptr[1] = m_button_data[m_focused_button_id].properties.userptr[1];
              mm_data.userptr[2] = m_button_data[m_focused_button_id].properties.userptr[2];
              mm_data.depth = depth;
              m_button_data[m_focused_button_id].on_mouse_move_lib_cb(mm_data);
              if (!m_button_data[m_focused_button_id].properties.on_mouse_event_function(mm_data)) {
                return 0;
              }
              return 1;
            }
            i = m_button_data.rnext(i);
          }
          if (m_focused_button_id != fan::uninitialized) {
            mouse_move_data_t mm_data;
            mm_data.context = context;
            mm_data.button_event = this;
            mm_data.element_id = m_button_data[m_focused_button_id].properties.cid;
            mm_data.mouse_stage = mouse_stage_e::outside;
            mm_data.userptr[0] = m_button_data[m_focused_button_id].properties.userptr[0];
            mm_data.userptr[1] = m_button_data[m_focused_button_id].properties.userptr[1];
            mm_data.userptr[2] = m_button_data[m_focused_button_id].properties.userptr[2];
            mm_data.depth = depth;
            m_button_data[m_focused_button_id].on_mouse_move_lib_cb(mm_data);
            if (!m_button_data[m_focused_button_id].properties.on_mouse_event_function(mm_data)) {
              m_focused_button_id = fan::uninitialized;
              return 0;
            }
            m_focused_button_id = fan::uninitialized;
          }
        }

        uint8_t feed_mouse_input(fan::opengl::context_t* context, uint16_t button, fan::key_state state, const fan::vec2& mouse_position, uint32_t depth) {
          if (m_do_we_hold_button == 0) {
            if (state == fan::key_state::press) {
              if (m_focused_button_id != fan::uninitialized) {
                m_do_we_hold_button = 1;

                mouse_input_data_t ii_data;
                ii_data.context = context;
                ii_data.button_event = this;
                ii_data.element_id = m_button_data[m_focused_button_id].properties.cid;
                ii_data.mouse_stage = mouse_stage_e::inside;
                ii_data.key = button;
                ii_data.key_state = fan::key_state::press;
                ii_data.userptr[0] = m_button_data[m_focused_button_id].properties.userptr[0];
                ii_data.userptr[1] = m_button_data[m_focused_button_id].properties.userptr[1];
                ii_data.userptr[2] = m_button_data[m_focused_button_id].properties.userptr[2];
                ii_data.depth = depth;
                m_button_data[m_focused_button_id].on_input_lib_cb(ii_data);
                if (!m_button_data[m_focused_button_id].properties.on_input_function(ii_data)) {
                  return 0;
                }
              }
              else {
                uint32_t i = m_button_data.rbegin();
                while (i != m_button_data.rend()) {
                  if (inside(i, coordinate_offset + mouse_position)) {
                    mouse_input_data_t ii_data;
                    ii_data.context = context;
                    ii_data.button_event = this;
                    ii_data.element_id = m_button_data[i].properties.cid;
                    ii_data.mouse_stage = mouse_stage_e::outside;
                    ii_data.key = button;
                    ii_data.key_state = state;
                    ii_data.userptr[0] = m_button_data[i].properties.userptr[0];
                    ii_data.userptr[1] = m_button_data[i].properties.userptr[1];
                    ii_data.userptr[2] = m_button_data[i].properties.userptr[2];
                    m_button_data[i].on_input_lib_cb(ii_data);
                    if (!m_button_data[i].properties.on_input_function(ii_data)) {
                      return 0;
                    }
                  }
                  i = m_button_data.rnext(i);
                }
                return 1; // clicked at space
              }
            }
            else {
              return 1;
            }
          }
          else {
            if (state == fan::key_state::press) {
              return 1; // double press
            }
            else {
              if (inside(m_focused_button_id, coordinate_offset + mouse_position)) {
                pointer_remove_flag = 1;
                mouse_input_data_t ii_data;
                ii_data.context = context;
                ii_data.button_event = this;
                ii_data.element_id = m_button_data[m_focused_button_id].properties.cid;
                ii_data.mouse_stage = mouse_stage_e::inside;
                ii_data.key = button;
                ii_data.key_state = fan::key_state::release;
                ii_data.userptr[0] = m_button_data[m_focused_button_id].properties.userptr[0];
                ii_data.userptr[1] = m_button_data[m_focused_button_id].properties.userptr[1];
                ii_data.userptr[2] = m_button_data[m_focused_button_id].properties.userptr[2];
                ii_data.depth = depth;
                m_button_data[m_focused_button_id].on_input_lib_cb(ii_data);
                m_button_data[m_focused_button_id].properties.on_input_function(ii_data);
                if (pointer_remove_flag == 0) {
                  return 1;
                  //rtb is deleted
                }
              }
              else {
                uint32_t i = m_button_data.rbegin();
                while (i != m_button_data.rend()) {
                  if (inside(i, coordinate_offset + mouse_position)) {
                    m_focused_button_id = i;
                    mouse_input_data_t ii_data;
                    ii_data.context = context;
                    ii_data.button_event = this;
                    ii_data.element_id = m_button_data[m_focused_button_id].properties.cid;
                    ii_data.mouse_stage = mouse_stage_e::inside_drag;
                    ii_data.key = button;
                    ii_data.key_state = fan::key_state::release;
                    ii_data.userptr[0] = m_button_data[m_focused_button_id].properties.userptr[0];
                    ii_data.userptr[1] = m_button_data[m_focused_button_id].properties.userptr[1];
                    ii_data.userptr[2] = m_button_data[m_focused_button_id].properties.userptr[2];
                    ii_data.depth = depth;
                    m_button_data[m_focused_button_id].on_input_lib_cb(ii_data);
                    m_button_data[m_focused_button_id].properties.on_input_function(ii_data);
                    break;
                  }
                  i = m_button_data.rnext(i);
                }

                pointer_remove_flag = 1;
                mouse_input_data_t ii_data;
                ii_data.context = context;
                ii_data.button_event = this;
                ii_data.element_id = m_button_data[m_focused_button_id].properties.cid;
                ii_data.mouse_stage = mouse_stage_e::outside;
                ii_data.key = button;
                ii_data.key_state = fan::key_state::release;
                ii_data.userptr[0] = m_button_data[m_focused_button_id].properties.userptr[0];
                ii_data.userptr[1] = m_button_data[m_focused_button_id].properties.userptr[1];
                ii_data.userptr[2] = m_button_data[m_focused_button_id].properties.userptr[2];
                ii_data.depth = depth;
                m_button_data[m_focused_button_id].on_input_lib_cb(ii_data);
                if (!m_button_data[m_focused_button_id].properties.on_input_function(ii_data)) {
                  m_do_we_hold_button = 0;
                  return 0;
                }
                if (pointer_remove_flag == 0) {
                  m_do_we_hold_button = 0;
                  return 1;
                }

                pointer_remove_flag = 0;
              }
              m_do_we_hold_button = 0;
            }
          }
        }

        uint32_t push_back(const properties_t& p, on_input_cb_t on_input_lib_cb = [](const mouse_input_data_t&) -> uint8_t { return 1; }, on_mouse_move_cb_t on_mouse_move_lib_cb = [](const mouse_move_data_t&) -> uint8_t { return 1; }) {
          button_data_t b;
          b.properties = p;
          b.on_input_lib_cb = on_input_lib_cb;
          b.on_mouse_move_lib_cb = on_mouse_move_lib_cb;
          return m_button_data.push_back(b);
        }
        void erase(uint32_t node_reference) {
          m_button_data.erase(node_reference);
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

        void write_in(FILE* f) {
          m_button_data.write_in(f);
        }
        void write_out(FILE* f) {
          m_button_data.write_out(f);
        }

    //  protected:

        inline static thread_local bool pointer_remove_flag;

        uint8_t m_old_mouse_stage;
        bool m_do_we_hold_button;
        uint32_t m_focused_button_id;

        struct button_data_t {
          properties_t properties;
          on_mouse_move_cb_t on_mouse_move_lib_cb;
          on_input_cb_t on_input_lib_cb;
        };

        bll_t<button_data_t> m_button_data;

        fan::vec2 coordinate_offset;
      };
    }
  }
}