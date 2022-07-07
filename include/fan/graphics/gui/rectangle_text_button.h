#pragma once

#include _FAN_PATH(graphics/gui/rectangle_text_box.h)
#include _FAN_PATH(graphics/gui/be.h)

namespace fan_2d {
  namespace graphics {
    namespace gui {


      template <typename T_user_global_data, typename T_user_instance_data>
      struct rectangle_text_button_t {

        using user_global_data_t = T_user_global_data;
        using user_instance_data_t = T_user_instance_data;

        using text_box_t = fan_2d::graphics::gui::rectangle_text_box_t<user_global_data_t, user_instance_data_t>;
        using text_renderer_t = text_box_t::text_renderer_t;
        using letter_t = text_box_t::letter_t;
        using box_t = text_box_t::box_t;

        struct properties_t : text_box_t::properties_t {

        };

        void open(fan::opengl::context_t* context, box_t::rectangle_t::move_cb_t move_cb_, const T_user_global_data& gd)
        {
          text_box.open(context, move_cb_, gd);
        }

        void close(fan::opengl::context_t* context)
        {
          text_box.close(context);
        }

        void push_back(fan::opengl::context_t* context, fan_2d::graphics::gui::be_t* button_event, text_renderer_t::letter_t* letters, const properties_t& p) {
          text_box.push_back(context, letters, p);
          fan_2d::graphics::gui::be_t::properties_t be_p;
          be_p.hitbox_type = fan_2d::graphics::gui::be_t::hitbox_type_t::rectangle;
          be_p.hitbox_rectangle.position = p.position;
          be_p.hitbox_rectangle.size = p.size;
          button_event->push_back(be_p);
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
      };
    }
  }
}