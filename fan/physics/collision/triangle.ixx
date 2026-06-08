module;

export module fan.physics.collision.triangle;

import std;

import fan.types.vector;
import fan.math;

namespace fm = fan::math;

export namespace fan_2d {
  namespace collision {
    namespace triangle {
      constexpr bool point_inside(const fan::vec2& v1, const fan::vec2& v2, const fan::vec2& v3, const fan::vec2& point) {
        f32_t d1 = 0, d2 = 0, d3 = 0;
        bool has_neg = 0, has_pos = 0;

        d1 = fm::sign(point, v1, v2);
        d2 = fm::sign(point, v2, v3);
        d3 = fm::sign(point, v3, v1);

        has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
        has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

        return !(has_neg && has_pos);
      }
      constexpr bool intersects_aabb(const fan::vec3& v0, const fan::vec3& v1, const fan::vec3& v2, const fan::vec3& bc, const fan::vec3& hs) {
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
    }
  }
}