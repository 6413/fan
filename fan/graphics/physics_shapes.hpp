#pragma once

// for shapes
#include <fan/graphics/loco.h>
#include <fan/physics/b2_integration.hpp>

#include <array>
#include <fan/time/timer.h>

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
          return b2MassData{ .mass = mass, .center = center_of_mass, .rotationalInertia = rotational_inertia };
        }
      };

      struct base_shape_t : loco_t::shape_t, fan::physics::entity_t {
        base_shape_t() = default;
        base_shape_t(loco_t::shape_t&& shape, fan::physics::entity_t&& entity, const mass_data_t& mass_data) :
          loco_t::shape_t(std::move(shape)),
          fan::physics::entity_t(std::move(entity)) {
          if (physics_update_nr.iic() == false) {
            gloco->remove_physics_update(physics_update_nr);
          }
          physics_update_nr = gloco->add_physics_update({
            .shape_id = *this,
            .body_id = *this,
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
          b2Body_SetMassData(*this, md_copy);
        }
        base_shape_t(const base_shape_t& r) : loco_t::shape_t(r), fan::physics::entity_t(r) {
          //if (this != )
          fan::physics::body_id_t new_body_id = fan::physics::deep_copy_body(gloco->physics_context.world_id, r);
          if (!B2_ID_EQUALS(r, (*this))) {
            destroy();
          }
          set_body(new_body_id);
          b2Body_GetWorldPoint(*this, fan::vec2(0));
          if (physics_update_nr.iic() == false) {
            gloco->remove_physics_update(physics_update_nr);
            physics_update_nr.sic();
          }
          if (!fan::physics::entity_t::is_valid()) {
            return;
          }
          physics_update_nr = gloco->add_physics_update({
            .shape_id = *this,
            .body_id = *this,
            .cb = (void*)shape_physics_update
          });
        }
        base_shape_t(base_shape_t&& r) : loco_t::shape_t(std::move(r)), fan::physics::entity_t(std::move(r)) {
          if (!B2_ID_EQUALS(r, (*this))) {
            destroy();
          }
          physics_update_nr = r.physics_update_nr;
          r.physics_update_nr.sic();
          r.set_body(b2_nullBodyId);
        }
        ~base_shape_t() {

          erase();
        }
        base_shape_t& operator=(const base_shape_t& r) {
          if (physics_update_nr.iic() == false) {
            gloco->remove_physics_update(physics_update_nr);
            physics_update_nr.sic();
          }
          if (this != &r) {
            loco_t::shape_t::operator=(r);

            fan::physics::body_id_t new_body_id = fan::physics::deep_copy_body(gloco->physics_context.world_id, r);
            if (!B2_ID_EQUALS(r, (*this))) {
              destroy();
            }
            set_body(new_body_id);
            if (!fan::physics::entity_t::is_valid()) {
              return *this;
            }
            physics_update_nr =  gloco->add_physics_update({
              .shape_id = *this,
              .body_id = *this,
              .cb = (void*)shape_physics_update
            });
          }
          return *this;
        }
        base_shape_t& operator=(base_shape_t&& r) {
          if (!B2_ID_EQUALS(r, (*this))) {
            destroy();
          }
          if (physics_update_nr.iic() == false) {
            gloco->remove_physics_update(physics_update_nr);
            physics_update_nr.sic();
          }
          if (this != &r) {
            loco_t::shape_t::operator=(std::move(r));
            fan::physics::entity_t::operator=(std::move(r));
            r.set_body(b2_nullBodyId);
            physics_update_nr = r.physics_update_nr;
            r.physics_update_nr.sic();
          }
          return *this;
        }
        void erase() {
          loco_t::shape_t::erase();
          fan::physics::entity_t::destroy();
          if (physics_update_nr.iic() == false) {
            gloco->remove_physics_update(physics_update_nr);
          }
          physics_update_nr.sic();
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

      struct sprite_t : base_shape_t {
        struct properties_t {
          camera_impl_t* camera = &gloco->orthographic_camera;
          fan::vec3 position = fan::vec3(0, 0, 0);
          fan::vec2 size = fan::vec2(0.1, 0.1);
          fan::vec3 angle = 0;
          fan::color color = fan::color(1, 1, 1, 1);
          fan::vec2 rotation_point = 0;
          loco_t::image_t image = gloco->default_texture;
          std::array<loco_t::image_t, 30> images;
          f32_t parallax_factor = 0;
          bool blending = true;
          uint32_t flags = 0;
          operator fan::graphics::sprite_properties_t() const {
            return fan::graphics::sprite_properties_t{
              .camera = camera,
              .position = position,
              .size = size,
              .angle = angle,
              .color = color,
              .rotation_point = rotation_point,
              .image = image,
              .images = images,
              .parallax_factor = parallax_factor,
              .blending = blending,
              .flags = flags
            };
          }
          uint8_t body_type = fan::physics::body_type_e::static_body;
          mass_data_t mass_data;
          fan::physics::shape_properties_t shape_properties;
        };
        sprite_t() = default;
        sprite_t(const properties_t& p) : base_shape_t(
          loco_t::shape_t(fan::graphics::sprite_t{ p }),
          fan::physics::entity_t(gloco->physics_context.create_box(p.position, p.size, p.body_type, p.shape_properties)),
          p.mass_data
        ) {
        }
        sprite_t(const sprite_t& r) : base_shape_t(r) {}
        sprite_t(sprite_t&& r) : base_shape_t(std::move(r)) {}
        sprite_t& operator=(const sprite_t& r) {
          base_shape_t::operator=(r);
          return *this;
        }
        sprite_t& operator=(sprite_t&& r) {
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
          fan::vec2 center0{0, -32.f};
          fan::vec2 center1{0, 32.f};
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

      struct capsule_sprite_t : base_shape_t {
        struct properties_t {
          camera_impl_t* camera = &gloco->orthographic_camera;
          fan::vec3 position = fan::vec3(0, 0, 0);
          fan::vec2 center0{0, -32.f};
          fan::vec2 center1{ 0, 32.f };
          f32_t radius = 64.0f;
          fan::vec3 angle = 0;
          fan::color color = fan::color(1, 1, 1, 1);
          fan::vec2 rotation_point = 0;
          loco_t::image_t image = gloco->default_texture;
          std::array<loco_t::image_t, 30> images;
          f32_t parallax_factor = 0;
          bool blending = true;
          uint32_t flags = 0;

          uint8_t body_type = fan::physics::body_type_e::static_body;
          mass_data_t mass_data;
          fan::physics::shape_properties_t shape_properties;

          operator fan::graphics::sprite_properties_t() const {
            return fan::graphics::sprite_properties_t{
              .camera = camera,
              .position = position,
              .size = radius,
              .angle = angle,
              .color = color,
              .rotation_point = rotation_point,
              .image = image,
              .images = images,
              .parallax_factor = parallax_factor,
              .blending = blending,
              .flags = flags
            };
          }
        };
        capsule_sprite_t() = default;
        capsule_sprite_t(const properties_t& p) : base_shape_t(
          loco_t::shape_t(fan::graphics::sprite_t{p}),
          fan::physics::entity_t(gloco->physics_context.create_capsule(p.position, b2Capsule{.center1 = p.center0, .center2 = p.center1, .radius = p.radius}, p.body_type, p.shape_properties)),
          p.mass_data
        ) {
        }
        capsule_sprite_t(const capsule_sprite_t& r) : base_shape_t(r) {}
        capsule_sprite_t(capsule_sprite_t&& r) : base_shape_t(std::move(r)) {}
        capsule_sprite_t& operator=(const capsule_sprite_t& r) {
          base_shape_t::operator=(r);
          return *this;
        }
        capsule_sprite_t& operator=(capsule_sprite_t&& r) {
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

    struct character2d_t : physics_shapes::base_shape_t {
      character2d_t();
      inline character2d_t(auto&& shape) : base_shape_t(std::move(shape)) {
        add_inputs();
      }
      static bool is_on_ground(fan::physics::body_id_t main, std::array<fan::physics::body_id_t, 2> feet, bool jumping);
      void add_inputs();
      void process_movement(f32_t friction = 12);
      f32_t force = 25.f;
      f32_t impulse = 3.f;
      f32_t max_speed = 500.f;
      f32_t jump_delay = 0.25f;
      bool jumping = false;
      fan::physics::body_id_t feet[2];
      f32_t walk_force = 0;
      bool handle_jump = true;
    };

    struct bone_e {
      enum {
        hip = 0,
        torso = 1,
        head = 2,
        upper_left_leg = 3,
        lower_left_leg = 4,
        upper_right_leg = 5,
        lower_right_leg = 6,
        upper_left_arm = 7,
        lower_left_arm = 8,
        upper_right_arm = 9,
        lower_right_arm = 10,
        bone_count = 11,
      };
    };
    inline std::string bone_to_string(int bone) {
      switch (bone) {
      case bone_e::hip: return "hip";
      case bone_e::torso: return "torso";
      case bone_e::head: return "head";
      case bone_e::upper_left_leg: return "upper_left_leg";
      case bone_e::lower_left_leg: return "lower_left_leg";
      case bone_e::upper_right_leg: return "upper_right_leg";
      case bone_e::lower_right_leg: return "lower_right_leg";
      case bone_e::upper_left_arm: return "upper_left_arm";
      case bone_e::lower_left_arm: return "lower_left_arm";
      case bone_e::upper_right_arm: return "upper_right_arm";
      case bone_e::lower_right_arm: return "lower_right_arm";
      default: return "unknown";
      }
    }

    struct bone_t {
      fan::graphics::physics_shapes::base_shape_t visual;
      b2JointId joint_id = b2_nullJointId;
      f32_t friction_scale;
      int parent_index;
      // local
      fan::vec3 position = 0;
      fan::vec2 size = 1;
      fan::vec2 pivot = 0;
      fan::vec2 offset = 0;
      f32_t scale = 1;
      f32_t lower_angle = 0;
      f32_t upper_angle = 0;
      f32_t reference_angle = 0;
      fan::vec2 center0 = 0;
      fan::vec2 center1 = 0;
    };

     void update_reference_angle(b2WorldId world, b2JointId& joint_id, float new_reference_angle);

    struct human_t {
      using bone_images_t = std::array<loco_t::image_t, bone_e::bone_count>;
      using bones_t = std::array<bone_t, bone_e::bone_count>;

      human_t() = default;
      human_t(const fan::vec2& position, const f32_t scale = 1.f, const bone_images_t& images={}, const fan::color& color=fan::colors::white);

      static void load_bones(const fan::vec2& position, f32_t scale, std::array<fan::graphics::bone_t, fan::graphics::bone_e::bone_count>& bones);
      static bone_images_t load_character_images(const std::string& character_folder_path, const loco_t::image_load_properties_t& lp);

      void load_preset(const fan::vec2& position, const f32_t scale, const bone_images_t& images, std::array<bone_t, bone_e::bone_count>& bones, const fan::color& color = fan::colors::white);
      void load(const fan::vec2& position, const f32_t scale = 1.f, const bone_images_t& images={}, const fan::color& color=fan::colors::white);

      void animate_walk(f32_t force, f32_t dt);
      void animate_jump(f32_t impulse, f32_t dt, bool is_jumping);

      void erase();

      bones_t bones;
      f32_t scale=1.f;
      bool is_spawned=false;
      int direction = 1;
      int look_direction = direction;
      int go_up = 0;
      fan::time::clock jump_animation_timer;
    };

    struct mouse_joint_t {

      struct QueryContext
      {
        b2Vec2 point;
        b2BodyId bodyId = b2_nullBodyId;
      };

      static bool QueryCallback(b2ShapeId shapeId, void* context);

      mouse_joint_t(fan::physics::body_id_t tb);

      void update_mouse(b2WorldId world_id, const fan::vec2& position);

      fan::physics::body_id_t target_body;
      b2JointId mouse_joint = b2_nullJointId;
    };
  }
}