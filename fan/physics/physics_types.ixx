module;

#if defined(fan_physics)
  #include <box2d/box2d.h>
  #include <functional>
#endif

export module fan.physics.types;

export import fan.types.vector;

#if defined(fan_physics)

export namespace fan::physics {
  struct aabb_t {
    fan::vec2 min;
    fan::vec2 max;
  };

  inline double length_units_per_meter = 256.0;

  fan::vec2d physics_to_render(const fan::vec2d& p) {
    return p * fan::physics::length_units_per_meter;
  }

  fan::vec2d render_to_physics(const fan::vec2d& p) {
    return p / fan::physics::length_units_per_meter;
  }

  struct shape_properties_t {
    f32_t friction = 0.6f;
    f32_t density = 1.0f;
    f32_t restitution = 0.0f;
    bool fixed_rotation = false;
    bool presolve_events = false;
    bool contact_events = false;
    bool is_sensor = false;
    f32_t linear_damping = 0.0f;
    f32_t angular_damping = 0.0f;
    fan::vec2 collision_multiplier = 1; // possibility to change multiplier of collision size
    b2Filter filter = b2DefaultFilter();
    bool fast_rotation = false;
  };

  struct body_type_e {
    enum : uint8_t {
      static_body = b2_staticBody,
      kinematic_body = b2_kinematicBody,
      dynamic_body = b2_dynamicBody,
      count = b2_bodyTypeCount
    };
  };

  using body_type = b2BodyType;
}
#endif