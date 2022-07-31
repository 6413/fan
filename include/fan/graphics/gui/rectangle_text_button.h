#pragma once

#include _FAN_PATH(graphics/gui/rectangle_text_box.h)
#include _FAN_PATH(graphics/gui/be.h)
#include _FAN_PATH(graphics/gui/ke.h)

namespace fan_2d {
  namespace graphics {
    namespace gui {

      struct rectangle_text_button_t {

        using text_box_t = fan_2d::graphics::gui::rectangle_text_box_t;
        using text_renderer_t = fan_2d::opengl::text_renderer_t;
        using letter_t = text_renderer_t::letter_t;
        //using box_t = text_box_t::box;

        struct properties_t : text_box_t::properties_t {
          be_t::on_input_cb_t mouse_input_cb = [](const be_t::mouse_input_data_t&) -> uint8_t {return 1; };
          be_t::on_mouse_move_cb_t mouse_move_cb = [](const be_t::mouse_move_data_t&) -> uint8_t { return 1; };

          void* userptr;
        };

      protected:

        static void lib_set_theme(
          rectangle_text_button_t* rtb,
          fan::opengl::context_t* context,
          letter_t* letter,
          uint32_t element_id,
          f32_t intensity
          ) {
          rtb->text_box.set_theme(context, letter, rtb->list[element_id].cid_text_box, rtb->text_box.get_theme(context, rtb->list[element_id].cid_text_box) * intensity);
        }

      #define make_code_small_plis(d_n, i) lib_set_theme( \
        (rectangle_text_button_t*)d_n.userptr[0], \
          d_n.context, \
          (letter_t*)d_n.userptr[1], \
          d_n.element_id, \
          i \
        );

        static uint8_t mouse_move_cb(const be_t::mouse_move_data_t& mm_data) {
          switch (mm_data.mouse_stage) {
            case mouse_stage::inside: {
              make_code_small_plis(mm_data, 1.1);
              break;
            }
            case mouse_stage::outside: {
              make_code_small_plis(mm_data, 1.0 / 1.1);
              break;
            }
          }
          return 1;
        }
        static uint8_t mouse_input_cb(const be_t::mouse_input_data_t& ii_data) {
          if (ii_data.key != fan::mouse_left) {
            return 1;
          }
          switch (ii_data.mouse_stage) {
            case mouse_stage::inside: {
              switch (ii_data.key_state) {
                case fan::key_state::press: {
                  make_code_small_plis(ii_data, 1.2);
                  break;
                }
                case fan::key_state::release: {
                  make_code_small_plis(ii_data, 1.0 / 1.2);
                  break;
                }
              }
              break;
            }
            case mouse_stage::outside: {
              make_code_small_plis(ii_data, 1.0 / 1.2);
              break;
            }
          }
          return 1;
        }

      #undef make_code_small_plis

      public:

        void open(fan::opengl::context_t* context)
        {
          list.open();
          text_box.open(context);
          e.amount = 0;
        }

        void close(fan::opengl::context_t* context)
        {
          for (uint32_t i = 0; i < list.size(); i++) {
            delete list[i].cid_text_box;
          }
          list.close();
          text_box.close(context);
        }

        uint32_t push_back(fan::opengl::context_t* context, fan_2d::graphics::gui::be_t* button_event, text_renderer_t::letter_t* letters, const properties_t& p) {
          uint32_t id;
          if (e.amount != 0) {
            id = e.id;
            e.id = *(uint32_t*)&list[e.id];
            e.amount--;
          }
          else {
            id = list.resize(list.size() + 1);
          }

          list[id].cid_text_box = new fan::opengl::cid_t;
          text_box.push_back(context, letters, list[id].cid_text_box, p);

          fan_2d::graphics::gui::be_t::properties_t be_p;
          be_p.hitbox_type = fan_2d::graphics::gui::be_t::hitbox_type_t::rectangle;
          be_p.hitbox_rectangle.position = p.position;
          be_p.hitbox_rectangle.size = p.size;
          be_p.on_input_function = p.mouse_input_cb;
          be_p.on_mouse_event_function = p.mouse_move_cb;
          be_p.userptr[0] = this;
          be_p.userptr[1] = letters;
          be_p.userptr[2] = p.userptr;
          be_p.element_cid = list[id].cid_text_box;
          list[id].button_event_id = button_event->push_back(be_p, mouse_input_cb, mouse_move_cb);

          return id;
        }

        void set_theme(fan::opengl::context_t* context, letter_t* letter, fan::opengl::cid_t* cid, const fan_2d::graphics::gui::theme& theme) {
          text_box.set_theme(context, letter, cid, theme);
        }

        void enable_draw(fan::opengl::context_t* context) {
          text_box.enable_draw(context);
        }

        void disable_draw(fan::opengl::context_t* context) {
          text_box.disable_draw(context);
        }

        text_box_t text_box;

        struct{
          uint32_t id;
          uint32_t amount;
        }e;

        struct element_t {
          fan::opengl::cid_t* cid_text_box;
          uint32_t button_event_id;
        };

        fan::hector_t<element_t> list;
      };
    }
  }
}