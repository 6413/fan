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

        };

        static void mouse_move_cb(fan::opengl::context_t* context, fan_2d::graphics::gui::be_t* be, uint32_t object_index, mouse_stage ms, void* userptr) {
          switch (ms) {
            case mouse_stage::inside: {
              rectangle_text_button_t* rtb = (rectangle_text_button_t*)userptr;
              rtb->text_box.set_theme(context,  &rtb->list[object_index].cid_text_box, fan_2d::graphics::gui::themes::deep_red(1.1));
              break;
            }
            case mouse_stage::outside: {
              rectangle_text_button_t* rtb = (rectangle_text_button_t*)userptr;
              rtb->text_box.set_theme(context,  &rtb->list[object_index].cid_text_box, fan_2d::graphics::gui::themes::deep_red(1));
              break;
            }
          }
        }
        static void mouse_input_cb(fan::opengl::context_t* context, be_t*, uint32_t object_index, uint16_t key, fan::key_state key_state, fan_2d::graphics::gui::mouse_stage ms, void* userptr) {
          if (key != fan::mouse_left) {
            return;
          }
          switch (ms) {
            case mouse_stage::inside: {
              rectangle_text_button_t* rtb = (rectangle_text_button_t*)userptr;
              switch (key_state) {
                case fan::key_state::press: {
                  rtb->text_box.set_theme(context, &rtb->list[object_index].cid_text_box, fan_2d::graphics::gui::themes::deep_red(1.2));
                  break;
                }
                case fan::key_state::release: {
                  rtb->text_box.set_theme(context,  &rtb->list[object_index].cid_text_box, fan_2d::graphics::gui::themes::deep_red(1));
                  break;
                }
              }
              break;
            }
            case mouse_stage::outside: {
              rectangle_text_button_t* rtb = (rectangle_text_button_t*)userptr;
              rtb->text_box.set_theme(context,  &rtb->list[object_index].cid_text_box, fan_2d::graphics::gui::themes::deep_red(1));
              break;
            }
          }
        }

        void open(fan::opengl::context_t* context)
        {
          list.open();
          text_box.open(context);
          e.amount = 0;
        }

        void close(fan::opengl::context_t* context)
        {
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

          text_box.push_back(context, letters, &list[id].cid_text_box, p);

          fan_2d::graphics::gui::be_t::properties_t be_p;
          be_p.hitbox_type = fan_2d::graphics::gui::be_t::hitbox_type_t::rectangle;
          be_p.hitbox_rectangle.position = p.position;
          be_p.hitbox_rectangle.size = p.size;
          be_p.on_mouse_event_function = mouse_move_cb;
          be_p.on_input_function = mouse_input_cb;
          be_p.userptr = this;
          list[id].button_event_id = button_event->push_back(be_p);

          return id;
        }

        void set_theme(fan::opengl::context_t* context, fan::opengl::cid_t* cid, const fan_2d::graphics::gui::theme& theme) {
          text_box.set_theme(context, cid, theme);
        }

        void enable_draw(fan::opengl::context_t* context) {
          text_box.enable_draw(context);
        }

        void disable_draw(fan::opengl::context_t* context) {
          text_box.disable_draw(context);
        }

        void bind_matrices(fan::opengl::context_t* context, fan::opengl::matrices_t* matrices) {
          text_box.bind_matrices(context, matrices);
        }

        text_box_t text_box;

        struct{
          uint32_t id;

          uint32_t amount;
        }e;

        struct element_t {
          fan::opengl::cid_t cid_text_box;
          uint32_t button_event_id;
        };

        fan::hector_t<element_t> list;
      };
    }
  }
}