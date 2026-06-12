export module fan.math.intersection;

import std;
import fan.types;
import fan.types.vector;
import fan.math;

namespace fm = fan::math;

export namespace fan::math::d2 {
  constexpr bool triangle_point_inside(const fan::vec2& v1, const fan::vec2& v2, const fan::vec2& v3, const fan::vec2& point) {
    f32_t d1 = fm::sign(point, v1, v2);
    f32_t d2 = fm::sign(point, v2, v3);
    f32_t d3 = fm::sign(point, v3, v1);
    return !(((d1 < 0) || (d2 < 0) || (d3 < 0)) && ((d1 > 0) || (d2 > 0) || (d3 > 0)));
  }
  constexpr bool rectangle_point_inside(const fan::vec2& p1, const fan::vec2& p2, const fan::vec2& p3, const fan::vec2& p4, const fan::vec2& point) {
    return triangle_point_inside(p1, p2, p4, point) || triangle_point_inside(p1, p3, p4, point);
  }
  constexpr bool aabb_point_inside(const fan::vec2& point, const fan::vec2& position, const fan::vec2& size) {
    return point.x >= position.x - size.x &&
           point.x <= position.x + size.x &&
           point.y >= position.y - size.y &&
           point.y <= position.y + size.y;
  }
  constexpr bool aabb_intersects_aabb(const fan::vec2& center1, const fan::vec2& half_size1,
    const fan::vec2& center2, const fan::vec2& half_size2) {
    fan::vec2 abs_half_size1 = { fm::abs(half_size1.x), fm::abs(half_size1.y) };
    fan::vec2 abs_half_size2 = { fm::abs(half_size2.x), fm::abs(half_size2.y) };

    bool overlap_x = fm::abs(center1.x - center2.x) < (abs_half_size1.x + abs_half_size2.x);
    bool overlap_y = fm::abs(center1.y - center2.y) < (abs_half_size1.y + abs_half_size2.y);

    return overlap_x && overlap_y;
  }
  constexpr bool point_inside_rotated(const fan::vec2& point, const fan::vec2& position, const fan::vec2& size, const fan::vec3& angle, const fan::vec2& rotation_point) {
    // TODO BROKEN
    return 0;
  }
  enum struct quadrant_e : std::uint8_t {
    top_left = 0,
    top_right = 1,
    bottom_right = 2,
    bottom_left = 3
  };
  constexpr quadrant_e get_quadrant(const fan::vec2& point, const fan::vec2& p) {
    if (point.x <= p.x && point.y <= p.y) {
      return quadrant_e::top_left;
    }
    if (point.x >= p.x && point.y <= p.y) {
      return quadrant_e::top_right;
    }
    if (point.x >= p.x && point.y >= p.y) {
      return quadrant_e::bottom_right;
    }
    return quadrant_e::bottom_left;
  }
  constexpr bool circle_point_inside(const fan::vec2& p, const fan::vec2& c, f64_t r) {
    f64_t dx = c.x - p.x;
    f64_t dy = c.y - p.y;
    dx *= dx;
    dy *= dy;
    f64_t distance_squared = dx + dy;
    f64_t radius_squared = r * r;
    return distance_squared <= radius_squared;
  }
  constexpr bool circle_intersects_circle(const fan::vec2& c0, f64_t r0, const fan::vec2& c1, f64_t r1) {
    f64_t dx = c1.x - c0.x;
    f64_t dy = c1.y - c0.y;
    f64_t rs = r0 + r1;
    return dx * dx + dy * dy <= rs * rs;
  }
}

export namespace fan::math::d3 {
  struct aabb_t {
    fan::vec3 min, max;
  };

  constexpr aabb_t triangle_bounds(const fan::vec3& v0, const fan::vec3& v1, const fan::vec3& v2) {
    return {v0.min(v1).min(v2), v0.max(v1).max(v2)};
  }

  constexpr bool aabb_intersects_aabb(const aabb_t& a, const aabb_t& b) {
    return a.min.x <= b.max.x && a.max.x >= b.min.x
        && a.min.y <= b.max.y && a.max.y >= b.min.y
        && a.min.z <= b.max.z && a.max.z >= b.min.z;
  }

  constexpr bool aabb_intersects_aabb(const aabb_t& a, const fan::vec3& min, const fan::vec3& max) {
    return aabb_intersects_aabb(a, {min, max});
  }

  constexpr bool triangle_intersects_aabb(const fan::vec3& v0, const fan::vec3& v1, const fan::vec3& v2, const fan::vec3& bc, const fan::vec3& hs) {
    fan::vec3 a[3] = {v0 - bc, v1 - bc, v2 - bc};
    if (fm::min(a[0].x, a[1].x, a[2].x) > hs.x || fm::max(a[0].x, a[1].x, a[2].x) < -hs.x) { return false; }
    if (fm::min(a[0].y, a[1].y, a[2].y) > hs.y || fm::max(a[0].y, a[1].y, a[2].y) < -hs.y) { return false; }
    if (fm::min(a[0].z, a[1].z, a[2].z) > hs.z || fm::max(a[0].z, a[1].z, a[2].z) < -hs.z) { return false; }
    fan::vec3 e[3] = {a[1] - a[0], a[2] - a[1], a[0] - a[2]};
    fan::vec3 n = e[0].cross(e[1]);
    if (fm::abs(n.dot(a[0])) > hs.x * fm::abs(n.x) + hs.y * fm::abs(n.y) + hs.z * fm::abs(n.z)) { return false; }
    auto t_cr = [](f32_t eu, f32_t ev, f32_t a0u, f32_t a0v, f32_t a1u, f32_t a1v, f32_t a2u, f32_t a2v, f32_t hu, f32_t hv) {
      f32_t p0 = a0u * ev - a0v * eu, p1 = a1u * ev - a1v * eu, p2 = a2u * ev - a2v * eu, rad = hu * fm::abs(ev) + hv * fm::abs(eu);
      return fm::min(p0, p1, p2) > rad || fm::max(p0, p1, p2) < -rad;
    };
    for (int i = 0; i < 3; ++i) {
      if (t_cr(e[i].z, e[i].y, a[0].y, a[0].z, a[1].y, a[1].z, a[2].y, a[2].z, hs.y, hs.z) ||
          t_cr(e[i].z, e[i].x, a[0].x, a[0].z, a[1].x, a[1].z, a[2].x, a[2].z, hs.x, hs.z) ||
          t_cr(e[i].y, e[i].x, a[0].x, a[0].y, a[1].x, a[1].y, a[2].x, a[2].y, hs.x, hs.y)) {
        return false;
      }
    }
    return true;
  }

  constexpr bool is_ray_intersecting_cube(const fan::ray3_t& ray, const fan::vec3& position, const fan::vec3& size) {
    fan::vec3 min_bounds = position - size;
    fan::vec3 max_bounds = position + size;

    fan::vec3 t_min = (min_bounds - ray.origin) / (ray.direction + fan::vec3(1e-6f));
    fan::vec3 t_max = (max_bounds - ray.origin) / (ray.direction + fan::vec3(1e-6f));

    fan::vec3 t1 = t_min.min(t_max);
    fan::vec3 t2 = t_min.max(t_max);

    f32_t t_near = std::max({t1.x, t1.y, t1.z});
    f32_t t_far = std::min({t2.x, t2.y, t2.z});

    return t_near <= t_far && t_far >= 0.0f;
  }

  constexpr fan::vec3 barycentric(const fan::vec3& p, const fan::vec3& a, const fan::vec3& b, const fan::vec3& c) {
    fan::vec3 v0 = b - a, v1 = c - a, v2 = p - a;
    f32_t d00 = v0.dot(v0), d01 = v0.dot(v1), d11 = v1.dot(v1), d20 = v2.dot(v0), d21 = v2.dot(v1);
    f32_t denom = d00 * d11 - d01 * d01;
    f32_t v = (d11 * d20 - d01 * d21) / denom;
    f32_t w = (d00 * d21 - d01 * d20) / denom;
    return {1.0f - v - w, v, w};
  }

  constexpr fan::vec3 closest_barycentric(const fan::vec3& p, const fan::vec3& a, const fan::vec3& b, const fan::vec3& c) {
    fan::vec3 ab = b - a;
    fan::vec3 ac = c - a;
    fan::vec3 ap = p - a;

    f32_t d1 = ab.dot(ap);
    f32_t d2 = ac.dot(ap);
    if (d1 <= 0.f && d2 <= 0.f) { return {1.f, 0.f, 0.f}; }

    fan::vec3 bp = p - b;
    f32_t d3 = ab.dot(bp);
    f32_t d4 = ac.dot(bp);
    if (d3 >= 0.f && d4 <= d3) { return {0.f, 1.f, 0.f}; }

    f32_t vc = d1 * d4 - d3 * d2;
    if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
      f32_t v = d1 / (d1 - d3);
      return {1.f - v, v, 0.f};
    }

    fan::vec3 cp = p - c;
    f32_t d5 = ab.dot(cp);
    f32_t d6 = ac.dot(cp);
    if (d6 >= 0.f && d5 <= d6) { return {0.f, 0.f, 1.f}; }

    f32_t vb = d5 * d2 - d1 * d6;
    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
      f32_t w = d2 / (d2 - d6);
      return {1.f - w, 0.f, w};
    }

    f32_t va = d3 * d6 - d5 * d4;
    if (va <= 0.f && d4 - d3 >= 0.f && d5 - d6 >= 0.f) {
      f32_t w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      return {0.f, 1.f - w, w};
    }

    f32_t denom = 1.f / (va + vb + vc);
    f32_t v = vb * denom;
    f32_t w = vc * denom;
    return {1.f - v - w, v, w};
  }
}