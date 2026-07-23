module;

export module fan.graphics.algorithm.raymarch;

import std;

import fan.types;
import fan.types.vector;

export namespace fan::graphics::algorithm {
  template<typename F>
  fan::vec2 raymarch(fan::vec2 start, fan::vec2 end, f32_t cell_size, F&& is_solid) {
    fan::vec2 diff = end - start;
    f32_t dist = diff.length();
    if (dist < 1e-4f) return end;
    fan::vec2 dir = diff / dist;
    f32_t step_size = cell_size * 0.5f;
    for (f32_t d = 0.f; d <= dist; d += step_size) {
      fan::vec2 p = start + dir * d;
      if (is_solid((int)std::floor(p.x / cell_size), (int)std::floor(p.y / cell_size))) return p;
    }
    return end;
  }

  template<typename F>
  fan::vec2 raymarch_thick(fan::vec2 start, fan::vec2 end, f32_t cell_size, f32_t radius, F&& is_solid) {
    fan::vec2 diff = end - start;
    f32_t dist = diff.length();
    if (dist < 1e-4f) return end;
    fan::vec2 dir = diff / dist;
    f32_t step_size = cell_size * 0.5f;
    f32_t radius_sq = radius * radius;
    for (f32_t d = 0.f; d <= dist; d += step_size) {
      fan::vec2 p = start + dir * d;
      int gx0 = (int)std::floor((p.x - radius) / cell_size);
      int gx1 = (int)std::ceil((p.x + radius) / cell_size);
      int gy0 = (int)std::floor((p.y - radius) / cell_size);
      int gy1 = (int)std::ceil((p.y + radius) / cell_size);
      for (int gy = gy0; gy <= gy1; ++gy) {
        for (int gx = gx0; gx <= gx1; ++gx) {
          if (!is_solid(gx, gy)) continue;
          fan::vec2 cell_min = fan::vec2(gx, gy) * cell_size;
          fan::vec2 cell_max = cell_min + cell_size;
          f32_t cx = std::clamp(p.x, cell_min.x, cell_max.x);
          f32_t cy = std::clamp(p.y, cell_min.y, cell_max.y);
          fan::vec2 d2 = fan::vec2(cx, cy) - p;
          if (d2.x * d2.x + d2.y * d2.y <= radius_sq) return p;
        }
      }
    }
    return end;
  }
}
