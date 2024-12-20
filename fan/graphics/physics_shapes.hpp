#pragma once

// for shapes
#include <fan/graphics/loco.h>
#include <fan/physics/b2_integration.hpp>

namespace fan {
  namespace graphics {
    namespace physics_shapes {

      void shape_physics_update(const loco_t::physics_update_data_t& data);

      struct rectangle_t : loco_t::shape_t, fan::physics::entity_t {
        struct properties_t {
          camera_impl_t* camera = &gloco->orthographic_camera;
          fan::vec3 position = fan::vec3(0, 0, 0);
          fan::vec2 size = fan::vec2(0.1, 0.1);
          fan::color color = fan::color(1, 1, 1, 1);
          fan::vec3 angle = 0;
          fan::vec2 rotation_point = 0;
          bool blending = false;
          operator fan::graphics::rectangle_properties_t() const {
            return fan::graphics::rectangle_properties_t{
              .camera = camera,
              .position = position,
              .size = size,
              .color = color,
              .angle = angle,
              .rotation_point = rotation_point,
              .blending = blending
            };
          }
          uint8_t body_type = fan::physics::body_type_e::static_body;
        };
        rectangle_t() = default;
        rectangle_t(const properties_t& p) : 
          loco_t::shape_t(fan::graphics::rectangle_t{p}),
          fan::physics::entity_t(gloco->physics_context.create_box(p.position, p.size, p.body_type))
        {
          gloco->add_physics_update({
            .shape = this,
            .cb = shape_physics_update
          });
        }
      };
    }
  }
}