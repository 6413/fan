export module fan.math.intersection;

import std;
import fan.types;
import fan.types.vector;
import fan.types.matrix;
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

  constexpr aabb_t merge_aabb(const aabb_t& a, const aabb_t& b) {
    return {a.min.min(b.min), a.max.max(b.max)};
  }

  constexpr aabb_t transform_aabb(const aabb_t& aabb, const fan::mat4& transform) {
    fan::vec3 first = transform * fan::vec3(aabb.min.x, aabb.min.y, aabb.min.z);
    aabb_t result{first, first};
    for (std::uint32_t corner = 1; corner < 8; ++corner) {
      fan::vec3 local(
        (corner & 1) ? aabb.max.x : aabb.min.x,
        (corner & 2) ? aabb.max.y : aabb.min.y,
        (corner & 4) ? aabb.max.z : aabb.min.z
      );
      fan::vec3 world = transform * local;
      result.min = result.min.min(world);
      result.max = result.max.max(world);
    }
    return result;
  }

  constexpr bool ray_intersects_aabb(const fan::ray3_t& ray, const fan::vec3& min, const fan::vec3& max, f32_t& hit_t, f32_t epsilon = 1e-6f) {
    f32_t t_min = 0.f;
    f32_t t_max = std::numeric_limits<f32_t>::max();
    for (std::uint32_t axis = 0; axis < 3; ++axis) {
      f32_t origin = ray.origin[axis];
      f32_t direction = ray.direction[axis];
      if (fm::abs(direction) < epsilon) {
        if (origin < min[axis] || origin > max[axis]) {
          return false;
        }
        continue;
      }
      f32_t inv_direction = 1.f / direction;
      f32_t t0 = (min[axis] - origin) * inv_direction;
      f32_t t1 = (max[axis] - origin) * inv_direction;
      if (t0 > t1) {
        std::swap(t0, t1);
      }
      t_min = std::max(t_min, t0);
      t_max = std::min(t_max, t1);
      if (t_min > t_max) {
        return false;
      }
    }
    hit_t = t_min;
    return true;
  }

  constexpr bool ray_intersects_aabb(const fan::ray3_t& ray, const aabb_t& aabb, f32_t& hit_t, f32_t epsilon = 1e-6f) {
    return ray_intersects_aabb(ray, aabb.min, aabb.max, hit_t, epsilon);
  }

  inline bool ray_intersects_sphere(const fan::ray3_t& ray, const fan::vec3& center, f32_t radius, f32_t& hit_t) {
    fan::vec3 to_center = ray.origin - center;
    f32_t a = ray.direction.dot(ray.direction);
    f32_t b = 2.f * to_center.dot(ray.direction);
    f32_t c = to_center.dot(to_center) - radius * radius;
    f32_t discriminant = b * b - 4.f * a * c;
    if (discriminant < 0.f) {
      return false;
    }
    f32_t sqrt_discriminant = std::sqrt(discriminant);
    f32_t inv_denom = 1.f / (2.f * a);
    f32_t t0 = (-b - sqrt_discriminant) * inv_denom;
    f32_t t1 = (-b + sqrt_discriminant) * inv_denom;
    if (t0 >= 0.f) {
      hit_t = t0;
      return true;
    }
    if (t1 >= 0.f) {
      hit_t = t1;
      return true;
    }
    return false;
  }

  inline bool ray_intersects_sphere(const fan::ray3_t& ray, const fan::vec3& center, f32_t radius) {
    f32_t hit_t = 0.f;
    return ray_intersects_sphere(ray, center, radius, hit_t);
  }

  constexpr bool triangle_intersects_aabb(const fan::vec3& v0, const fan::vec3& v1, const fan::vec3& v2, const fan::vec3& bc, const fan::vec3& hs) {
    f32_t a0x = v0.x - bc.x, a0y = v0.y - bc.y, a0z = v0.z - bc.z;
    f32_t a1x = v1.x - bc.x, a1y = v1.y - bc.y, a1z = v1.z - bc.z;
    f32_t a2x = v2.x - bc.x, a2y = v2.y - bc.y, a2z = v2.z - bc.z;

    if (std::max(a0x, std::max(a1x, a2x)) < -hs.x || std::min(a0x, std::min(a1x, a2x)) > hs.x) { return false; }
    if (std::max(a0y, std::max(a1y, a2y)) < -hs.y || std::min(a0y, std::min(a1y, a2y)) > hs.y) { return false; }
    if (std::max(a0z, std::max(a1z, a2z)) < -hs.z || std::min(a0z, std::min(a1z, a2z)) > hs.z) { return false; }

    f32_t e0x = a1x - a0x, e0y = a1y - a0y, e0z = a1z - a0z;
    f32_t e1x = a2x - a1x, e1y = a2y - a1y, e1z = a2z - a1z;
    f32_t e2x = a0x - a2x, e2y = a0y - a2y, e2z = a0z - a2z;

    f32_t p0, p1, p2, r;

    p0 = a0z * e0y - a0y * e0z; p2 = a2z * e0y - a2y * e0z; r = hs.y * std::abs(e0z) + hs.z * std::abs(e0y);
    if (std::max(p0, p2) < -r || std::min(p0, p2) > r) { return false; }
    p0 = a0x * e0z - a0z * e0x; p2 = a2x * e0z - a2z * e0x; r = hs.x * std::abs(e0z) + hs.z * std::abs(e0x);
    if (std::max(p0, p2) < -r || std::min(p0, p2) > r) { return false; }
    p0 = a0y * e0x - a0x * e0y; p2 = a2y * e0x - a2x * e0y; r = hs.x * std::abs(e0y) + hs.y * std::abs(e0x);
    if (std::max(p0, p2) < -r || std::min(p0, p2) > r) { return false; }

    p0 = a0z * e1y - a0y * e1z; p1 = a1z * e1y - a1y * e1z; r = hs.y * std::abs(e1z) + hs.z * std::abs(e1y);
    if (std::max(p0, p1) < -r || std::min(p0, p1) > r) { return false; }
    p0 = a0x * e1z - a0z * e1x; p1 = a1x * e1z - a1z * e1x; r = hs.x * std::abs(e1z) + hs.z * std::abs(e1x);
    if (std::max(p0, p1) < -r || std::min(p0, p1) > r) { return false; }
    p0 = a0y * e1x - a0x * e1y; p1 = a1y * e1x - a1x * e1y; r = hs.x * std::abs(e1y) + hs.y * std::abs(e1x);
    if (std::max(p0, p1) < -r || std::min(p0, p1) > r) { return false; }

    p0 = a0z * e2y - a0y * e2z; p1 = a1z * e2y - a1y * e2z; r = hs.y * std::abs(e2z) + hs.z * std::abs(e2y);
    if (std::max(p0, p1) < -r || std::min(p0, p1) > r) { return false; }
    p0 = a0x * e2z - a0z * e2x; p1 = a1x * e2z - a1z * e2x; r = hs.x * std::abs(e2z) + hs.z * std::abs(e2x);
    if (std::max(p0, p1) < -r || std::min(p0, p1) > r) { return false; }
    p0 = a0y * e2x - a0x * e2y; p1 = a1y * e2x - a1x * e2y; r = hs.x * std::abs(e2y) + hs.y * std::abs(e2x);
    if (std::max(p0, p1) < -r || std::min(p0, p1) > r) { return false; }

    f32_t nx = e0y * e1z - e0z * e1y;
    f32_t ny = e0z * e1x - e0x * e1z;
    f32_t nz = e0x * e1y - e0y * e1x;
    r = hs.x * std::abs(nx) + hs.y * std::abs(ny) + hs.z * std::abs(nz);
    if (std::abs(nx * a0x + ny * a0y + nz * a0z) > r) { return false; }

    return true;
  }

  constexpr bool is_ray_intersecting_cube(const fan::ray3_t& ray, const fan::vec3& position, const fan::vec3& size) {
    f32_t hit_t = 0.f;
    return ray_intersects_aabb(ray, position - size, position + size, hit_t);
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
