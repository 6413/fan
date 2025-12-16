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

    static constexpr aabb_t normalized_impl(const fan::vec2& a,
      const fan::vec2& b) {
      const bool x_lt = a.x < b.x;
      const bool y_lt = a.y < b.y;

      return {
        { x_lt ? a.x : b.x, y_lt ? a.y : b.y },
        { x_lt ? b.x : a.x, y_lt ? b.y : a.y }
      };
    }
    constexpr void normalize() {
      *this = normalized_impl(min, max);
    }

    constexpr aabb_t normalized() const {
      return normalized_impl(min, max);
    }

    static constexpr aabb_t normalized(const fan::vec2& a,
      const fan::vec2& b) {
      return normalized_impl(a, b);
    }

    constexpr bool intersects(const aabb_t& o) const {
      return !(max.x < o.min.x || min.x > o.max.x ||
        max.y < o.min.y || min.y > o.max.y);
    }

    constexpr bool intersects(const fan::vec2& a, const fan::vec2& b) const {
      return intersects(normalized(a, b));
    }

    constexpr bool contains(const aabb_t& o) const {
      return o.min.x >= min.x && o.max.x <= max.x &&
        o.min.y >= min.y && o.max.y <= max.y;
    }

    constexpr bool contains_point(const fan::vec2& p) const {
      return p.x >= min.x && p.x <= max.x &&
        p.y >= min.y && p.y <= max.y;
    }
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