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

        fan::window_t* get_window(loco_t* loco);

        struct mouse_move_data_t {
          loco_t* loco;
          be_t* button_event;
          void* element_id;
          mouse_stage_e mouse_stage;
          void* userptr;
          uint32_t depth;
          bool changed;
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
          on_input_cb_t on_input_cb;
          on_mouse_move_cb_t on_mouse_event_cb;

          void* userptr;

          void* cid;

          fan::opengl::viewport_t* viewport;

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

        void open() {
          m_button_data.open();
          m_old_mouse_stage = fan::uninitialized;
          m_do_we_hold_button = 0;
          m_focused_button_id = fan::uninitialized;
        }
        void close() {
          m_button_data.close();
        }

        auto inside(loco_t* loco, uint32_t i, const fan::vec2& p) {
          switch(m_button_data[i].properties.hitbox_type) {
            case hitbox_type_t::rectangle: {
              fan::vec2 size = m_button_data[i].properties.hitbox_rectangle.size;
              return fan_2d::collision::rectangle::point_inside_no_rotation(
                p, 
                m_button_data[i].properties.hitbox_rectangle.position - size, 
                m_button_data[i].properties.hitbox_rectangle.position + size
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
 
        uint8_t feed_mouse_move(loco_t* loco, const fan::vec2& mouse_position, uint32_t depth) {
          #define move_data(index) \
          move_data.loco = loco;   \
          move_data.button_event = this;   \
          move_data.element_id = m_button_data[index].properties.cid;   \
          move_data.userptr = m_button_data[index].properties.userptr;   \
          move_data.depth = depth;   \
          m_button_data[index].on_mouse_move_lib_cb(move_data);   \
          if (!m_button_data[index].properties.on_mouse_event_cb(move_data)) {   \
            return 0;   \
          }

          auto get_mouse_position = [&](uint32_t i) {
            fan::vec2 viewport_position = m_button_data[i].properties.viewport->get_viewport_position(); 
            fan::vec2 viewport_size = m_button_data[i].properties.viewport->get_viewport_size();
            fan::vec2 rp = (mouse_position - viewport_position) / (viewport_size / 2);
            rp.x -= 1;
            rp.y += 1;
            return rp;
          };

          if (m_do_we_hold_button == 1 && m_focused_button_id != fan::uninitialized) {
            mouse_move_data_t move_data;
            move_data.changed = false;
            if (inside(loco, m_focused_button_id, get_mouse_position(m_focused_button_id))) {
              move_data.mouse_stage = mouse_stage_e::inside;
              move_data(m_focused_button_id);
            }
            else {
              uint32_t id = m_focused_button_id;
              m_focused_button_id = fan::uninitialized;
              move_data.mouse_stage = mouse_stage_e::outside;
              move_data(id);
            }
            
          }
          if (m_focused_button_id != fan::uninitialized) {
            mouse_move_data_t move_data;
            
            if (inside(loco, m_focused_button_id, get_mouse_position(m_focused_button_id))) {
              move_data.changed = false;
              move_data.mouse_stage = mouse_stage_e::inside;
              move_data(m_focused_button_id);
            }
            else {
              uint32_t id = m_focused_button_id;
              m_focused_button_id = fan::uninitialized;
              move_data.changed = true;
              move_data.mouse_stage = mouse_stage_e::outside;
              move_data(id);
            }
          }
          uint32_t i = m_button_data.rbegin();
          while (i != m_button_data.rend()) {
            if (inside(loco, i, get_mouse_position(i))) {
              m_focused_button_id = i;

              mouse_move_data_t move_data;
              move_data.changed = true;
              move_data.mouse_stage = mouse_stage_e::inside;
              move_data(m_focused_button_id);
              return 1;
            }
            else {
              mouse_move_data_t move_data;
              move_data.changed = false;
              move_data.mouse_stage = mouse_stage_e::outside;
              move_data(i);
            }
            i = m_button_data.rnext(i);
          }
          return 1;
        }

        uint8_t feed_mouse_input(loco_t* loco, uint16_t button, fan::key_state state, const fan::vec2& mouse_position, uint32_t depth, void** focus_id) {
          auto reset_focus = [&] {
            *focus_id = (void*)fan::uninitialized;
          };
          auto set_focus = [&](auto id) {
            *focus_id = (void*)m_button_data[id].properties.cid;
          };
          auto get_mouse_position = [&](uint32_t i) {
            fan::vec2 viewport_position = m_button_data[i].properties.viewport->get_viewport_position(); 
            fan::vec2 viewport_size = m_button_data[i].properties.viewport->get_viewport_size();
            fan::vec2 rp = (mouse_position - viewport_position) / (viewport_size / 2);
            rp.x -= 1;
            rp.y += 1;
            return rp;
          };

          #define input_data(index) \
          input_data.loco = loco; \
          input_data.button_event = this; \
          input_data.element_id = m_button_data[index].properties.cid; \
          input_data.key = button; \
          input_data.userptr = m_button_data[index].properties.userptr; \
          input_data.depth = depth; \
          m_button_data[index].on_input_lib_cb(input_data); \
          if (!m_button_data[index].properties.on_input_cb(input_data)) { \
            return 0; \
          }

          if (m_do_we_hold_button == 0) {
            if (state == fan::key_state::press) {
              if (m_focused_button_id != fan::uninitialized) {
                m_do_we_hold_button = 1;

                mouse_input_data_t input_data;
                input_data.key_state = fan::key_state::press;
                input_data.mouse_stage = mouse_stage_e::inside;
                set_focus(m_focused_button_id);
                input_data(m_focused_button_id);
              }
              else {
                uint32_t i = m_button_data.rbegin();
                while (i != m_button_data.rend()) {
                  if (inside(loco, i, get_mouse_position(i))) {
                    mouse_input_data_t input_data;
                    input_data.key_state = fan::key_state::press;
                    input_data.mouse_stage = mouse_stage_e::inside;
                    set_focus(i);
                    input_data(i);
                  }
                  i = m_button_data.rnext(i);
                }
                reset_focus();
                return 1; // clicked at space
              }
            }
            else {
              uint32_t i = m_button_data.rbegin();
              while (i != m_button_data.rend()) {
                if (inside(loco, i, get_mouse_position(i))) {
                  mouse_input_data_t input_data;
                  input_data.mouse_stage = mouse_stage_e::inside;
                  input_data.key_state = fan::key_state::release;
                  input_data(i);
                }
                /* maybe not possible xd
                else {
                  fan::throw_error("b");
                  m_do_we_hold_button = 0;
                  m_focused_button_id = i;
                  mouse_input_data_t input_data;
                  input_data.mouse_stage = mouse_stage_e::outside;
                  input_data.key_state = fan::key_state::release;
                  input_data(m_focused_button_id);
                }*/
                i = m_button_data.rnext(i);
              }
              return 1;
            }
          }
          else {
            if (state == fan::key_state::press) {
              return 1; // double press
            }
            else if (m_focused_button_id != fan::uninitialized){
              if (inside(loco, m_focused_button_id, get_mouse_position(m_focused_button_id))) {
                pointer_remove_flag = 1;
                m_do_we_hold_button = 0;
                mouse_input_data_t input_data;
                input_data.mouse_stage = mouse_stage_e::inside;
                input_data.key_state = fan::key_state::release;
                input_data(m_focused_button_id);
                if (pointer_remove_flag == 0) {
                  return 1;
                  //rtb is deleted
                }
              }
              else {
                uint32_t i = m_button_data.rbegin();
                while (i != m_button_data.rend()) {
                  if (inside(loco, i, get_mouse_position(i))) {
                    m_do_we_hold_button = 0;
                    m_focused_button_id = i;
                    mouse_input_data_t input_data;
                    input_data.mouse_stage = mouse_stage_e::inside_drag;
                    input_data.key_state = fan::key_state::release;
                    input_data(m_focused_button_id);
                    break;
                  }
                  i = m_button_data.rnext(i);
                }

                pointer_remove_flag = 1;
                m_do_we_hold_button = 0;
                uint32_t id = m_focused_button_id;
                m_focused_button_id = fan::uninitialized;
                mouse_input_data_t input_data;
                input_data.mouse_stage = mouse_stage_e::outside;
                input_data.key_state = fan::key_state::release;
                input_data(id);
                if (pointer_remove_flag == 0) {
                  return 1;
                }

                pointer_remove_flag = 0;
              }
              m_do_we_hold_button = 0;
            }
          }
          return 1;
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
      };
    }
  }
}

#undef input_data
#undef move_data
#undef dummy_position