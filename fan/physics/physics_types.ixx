module;

#if defined(FAN_PHYSICS_2D)
  #include <box2d/box2d.h>
#endif

export module fan.physics.types;

import std;

import fan.types;
import fan.types.vector;
import fan.math;

export namespace fan::physics {
  // allow aabb build without flag
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

    constexpr aabb_t normalize() const {
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

    constexpr bool segment_intersect(const fan::vec2& start, const fan::vec2& end) const {
      fan::vec2 dir = end - start;
      f32_t tmin = 0.f, tmax = 1.f;
      auto check = [&](f32_t p, f32_t d, f32_t bmin, f32_t bmax) {
        if (fan::math::abs(d) < 1e-6f) { return p >= bmin && p <= bmax; }
        f32_t t0 = (bmin - p) / d, t1 = (bmax - p) / d;
        if (t0 > t1) { std::swap(t0, t1); }
        tmin = fan::math::max(tmin, t0);
        tmax = fan::math::min(tmax, t1);
        return tmax >= tmin;
      };
      return check(start.x, dir.x, min.x, max.x) &&
             check(start.y, dir.y, min.y, max.y);
    }
    static constexpr aabb_t from_center(const fan::vec2& center, const fan::vec2& half_size) {
      return aabb_t {center - half_size, center + half_size};
    }
    constexpr auto center() const {
      return (min + max) * 0.5f;
    }
    constexpr auto extents() const {
      return (max - min) * 0.5f;
    }
    bool push_out(fan::vec2& point, f32_t distance) const {
      fan::vec2 c = center();
      fan::vec2 e = extents();
      fan::vec2 d = point - c;

      if (std::abs(d.x) < e.x && std::abs(d.y) < e.y) {
        if (d.x == 0.f && d.y == 0.f) { d.y = 1.f; }
        point += d.normalize() * distance;
        return true;
      }
      return false;
    }
  };

#if defined(FAN_PHYSICS_2D)

  inline double length_units_per_meter = 256.0;

  fan::vec2d physics_to_render(const fan::vec2d& p) {
    return p * fan::physics::length_units_per_meter;
  }

  fan::vec2d render_to_physics(const fan::vec2d& p) {
    return p / fan::physics::length_units_per_meter;
  }

  struct shape_properties_t {
    f32_t friction = 0.f;
    f32_t density = 1.0f;
    f32_t restitution = 0.0f;
    bool fixed_rotation = false;
    bool presolve_events = false;
    bool contact_events = false;
    bool is_sensor = false;
    f32_t linear_damping = 0.0f;
    f32_t angular_damping = 0.0f;
    fan::vec2 collision_multiplier = 1;
    b2Filter filter = b2DefaultFilter();
    bool fast_rotation = false;
    bool is_bullet = false;
    bool allow_sleep = true;
  };

  struct body_type_e {
    enum : std::uint8_t {
      static_body = b2_staticBody,
      kinematic_body = b2_kinematicBody,
      dynamic_body = b2_dynamicBody,
      count = b2_bodyTypeCount
    };
  };

  using body_type = b2BodyType;

  // TODO think some other way to make this, not multi engine safe
  auto& debug_draw_cb() {
    static std::function<void(bool enabled, void* render_view)> func;
    return func;
  }
  auto& debug_draw_init_cb() {
    static std::function<b2DebugDraw(bool enabled)> func;
    return func;
  }
  bool& is_debug_draw_enabled() {
    static bool enabled = false;
    return enabled;
  }
  #endif
}