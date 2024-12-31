#pragma once

// for shapes
#include <fan/graphics/loco.h>
#include <fan/physics/b2_integration.hpp>

namespace fan {
  namespace graphics {
    namespace physics_shapes {

      void shape_physics_update(const loco_t::physics_update_data_t& data);

      struct mass_data_t {
        // kgs
        f32_t mass = -1.f;
        fan::vec2 center_of_mass = 0.f;
        f32_t rotational_inertia = -1.f;
        operator b2MassData() const {
          return b2MassData{.mass = mass, .center = center_of_mass, .rotationalInertia = rotational_inertia};
        }
      };

      struct base_shape_t : loco_t::shape_t, fan::physics::entity_t{
        base_shape_t() = default;
        base_shape_t(loco_t::shape_t&& shape, fan::physics::entity_t&& entity, const mass_data_t& mass_data) :
          loco_t::shape_t(std::move(shape)),
          fan::physics::entity_t(std::move(entity)){
          physics_update_nr = gloco->add_physics_update({
            .shape_id = *this,
            .body_id = body_id,
            .cb = (void*)shape_physics_update
          });
          b2MassData md = b2Body_GetMassData(*this);
          mass_data_t md_copy = mass_data;
          if (mass_data.mass < 0.f) {
            md_copy.mass = md.mass;
          }
          if (mass_data.center_of_mass == 0) {
            md_copy.center_of_mass = md.center;
          }
          if (mass_data.rotational_inertia < 0.f) {
            md_copy.rotational_inertia = md.rotationalInertia;
          }
          b2Body_SetMassData(body_id, md_copy);
        }
        base_shape_t(const base_shape_t& r) : loco_t::shape_t(r), fan::physics::entity_t(r) {
           physics_update_nr = gloco->add_physics_update({
            .shape_id = *this,
            .body_id = body_id,
            .cb = (void*)shape_physics_update
          });
        }
        base_shape_t(base_shape_t&& r) : loco_t::shape_t(std::move(r)), fan::physics::entity_t(std::move(r)) {
          physics_update_nr = r.physics_update_nr;
          r.physics_update_nr.sic();
        }
        ~base_shape_t() {
          if (physics_update_nr.iic()) {
            return;
          }
          gloco->remove_physics_update(physics_update_nr);
          physics_update_nr.sic();
        }
        base_shape_t& operator=(const base_shape_t& r) {
          loco_t::shape_t::operator=(r);
          fan::physics::entity_t::operator=(r);
          physics_update_nr =  gloco->add_physics_update({
            .shape_id = *this,
            .body_id = r.body_id,
            .cb = (void*)shape_physics_update
          });
          return *this;
        }
        base_shape_t& operator=(base_shape_t&& r) {
          loco_t::shape_t::operator=(std::move(r));
          fan::physics::entity_t::operator=(std::move(r));
          physics_update_nr = r.physics_update_nr;
          r.physics_update_nr.sic();
          return *this;
        }
        operator fan::physics::body_id_t() const {
          return body_id;
        }

        loco_t::physics_update_cbs_t::nr_t physics_update_nr;
      };

      struct rectangle_t : base_shape_t {
        struct properties_t {
          camera_impl_t* camera = &gloco->orthographic_camera;
          fan::vec3 position = fan::vec3(0, 0, 0);
          fan::vec2 size = fan::vec2(0.1, 0.1);
          fan::color color = fan::color(1, 1, 1, 1);
          fan::color outline_color = color;
          fan::vec3 angle = 0;
          fan::vec2 rotation_point = 0;
          bool blending = false;
          operator fan::graphics::rectangle_properties_t() const {
            return fan::graphics::rectangle_properties_t{
              .camera = camera,
              .position = position,
              .size = size,
              .color = color,
              .outline_color = outline_color,
              .angle = angle,
              .rotation_point = rotation_point,
              .blending = blending
            };
          }
          uint8_t body_type = fan::physics::body_type_e::static_body;
          mass_data_t mass_data;
          fan::physics::shape_properties_t shape_properties;
        };
        rectangle_t() = default;
        rectangle_t(const properties_t& p) : base_shape_t(
          loco_t::shape_t(fan::graphics::rectangle_t{p}),
          fan::physics::entity_t(gloco->physics_context.create_box(p.position, p.size, p.body_type, p.shape_properties)),
          p.mass_data
        ) {
        }
        rectangle_t(const rectangle_t& r) : base_shape_t(r) {}
        rectangle_t(rectangle_t&& r) : base_shape_t(std::move(r)) {}
        rectangle_t& operator=(const rectangle_t& r) {
          base_shape_t::operator=(r);
          return *this;
        }
        rectangle_t& operator=(rectangle_t&& r) {
          base_shape_t::operator=(std::move(r));
          return *this;
        }
      };

      struct circle_t : base_shape_t {
        struct properties_t {
          camera_impl_t* camera = &gloco->orthographic_camera;
          fan::vec3 position = fan::vec3(0, 0, 0);
          f32_t radius = 0.1f;
          fan::color color = fan::color(1, 1, 1, 1);
          bool blending = true;
          uint32_t flags = 0;
          operator fan::graphics::circle_properties_t() const {
            return fan::graphics::circle_properties_t{
              .camera = camera,
              .position = position,
              .radius = radius,
              .color = color,
              .blending = blending,
              .flags = flags
            };
          }
          uint8_t body_type = fan::physics::body_type_e::static_body;
          mass_data_t mass_data;
          fan::physics::shape_properties_t shape_properties;
        };
        circle_t() = default;
        circle_t(const properties_t& p) : base_shape_t(
          loco_t::shape_t(fan::graphics::circle_t{p}),
          fan::physics::entity_t(gloco->physics_context.create_circle(p.position, p.radius, p.body_type, p.shape_properties)),
          p.mass_data
        ) {
        }
        circle_t(const circle_t& r) : base_shape_t(r) {}
        circle_t(circle_t&& r) : base_shape_t(std::move(r)) {}
        circle_t& operator=(const circle_t& r) {
          base_shape_t::operator=(r);
          return *this;
        }
        circle_t& operator=(circle_t&& r) {
          base_shape_t::operator=(std::move(r));
          return *this;
        }
      };
      struct capsule_t : base_shape_t {
        struct properties_t {
          camera_impl_t* camera = &gloco->orthographic_camera;
          fan::vec3 position = fan::vec3(0, 0, 0);
          fan::vec2 center0 = 0;
          fan::vec2 center1{0, 128.f};
          f32_t radius = 32.f;
          fan::color color = fan::color(1, 1, 1, 1);
          fan::color outline_color = color;
          bool blending = true;
          uint32_t flags = 0;
          operator fan::graphics::capsule_properties_t() const {
            return fan::graphics::capsule_properties_t{
              .camera = camera,
              .position = position,
              .center0 = center0,
              .center1 = center1,
              .radius = radius,
              .color = color,
              .outline_color = outline_color,
              .blending = blending,
              .flags = flags
            };
          }
          uint8_t body_type = fan::physics::body_type_e::static_body;
          mass_data_t mass_data;
          fan::physics::shape_properties_t shape_properties;
        };
        capsule_t() = default;
        capsule_t(const properties_t& p) : base_shape_t(
          loco_t::shape_t(fan::graphics::capsule_t{p}),
          fan::physics::entity_t(gloco->physics_context.create_capsule(p.position, b2Capsule{.center1 = p.center0, .center2 = p.center1, .radius = p.radius}, p.body_type, p.shape_properties)),
          p.mass_data
        ) {
        }
        capsule_t(const capsule_t& r) : base_shape_t(r) {}
        capsule_t(capsule_t&& r) : base_shape_t(std::move(r)) {}
        capsule_t& operator=(const capsule_t& r) {
          base_shape_t::operator=(r);
          return *this;
        }
        capsule_t& operator=(capsule_t&& r) {
          base_shape_t::operator=(std::move(r));
          return *this;
        }
      };
      std::array<fan::graphics::physics_shapes::rectangle_t, 4> create_stroked_rectangle(
        const fan::vec2& center_position,
        const fan::vec2& half_size,
        f32_t thickness,
        const fan::color& color = fan::color::hex(0x6e8d6eff),
        std::array<fan::physics::shape_properties_t, 4> shape_properties = { {
          {.friction = 0},
          {.friction = 0.6},
          {.friction = 0},
          {.friction = 0}
        } });
    }

    struct character2d_t {

      inline character2d_t(auto&& shape) : character(std::move(shape)) {
          gloco->input_action.add(fan::key_a, "move_left");
          gloco->input_action.add(fan::key_d, "move_right");
          gloco->input_action.add(fan::key_space, "move_up");
      }

      void process_movement(f32_t friction = 12);

      f32_t force = 25.f;
      f32_t impulse = 10.f;
      f32_t jump_delay = 0.25f;
      bool jumping = false;
      fan::graphics::physics_shapes::capsule_t character;
    };
  }
}