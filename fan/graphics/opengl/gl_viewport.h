#pragma once

#include <fan/types/vector.h>
#include <fan/physics/collision/rectangle.h>

namespace fan {
  namespace opengl {

    struct viewport_t;

    #include "viewport_list_builder_settings.h"
    #define BLL_set_declare_NodeReference 1
    #define BLL_set_declare_rest 0
    #include <fan/BLL/BLL.h>

    struct viewport_t {

      void open();
      void close();

      fan::vec2 get_position() const
      {
        return viewport_position;
      }

      fan::vec2 get_size() const
      {
        return viewport_size;
      }

      void set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size);
      void zero();
      static void set_viewport(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size);

      bool inside(const fan::vec2& position) const {
        return fan_2d::collision::rectangle::point_inside_no_rotation(position, viewport_position + viewport_size / 2, viewport_size / 2);
      }

      bool inside_wir(const fan::vec2& position) const {
        return fan_2d::collision::rectangle::point_inside_no_rotation(position, viewport_size / 2, viewport_size / 2);
      }

      fan::vec2 viewport_position;
      fan::vec2 viewport_size;

      fan::opengl::viewport_list_NodeReference_t viewport_reference;
    };

    #include "viewport_list_builder_settings.h"
    #define BLL_set_declare_NodeReference 0
    #define BLL_set_declare_rest 1
    #include <fan/BLL/BLL.h>
  }
}