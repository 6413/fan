module;

module fan.graphics.algorithm.raycast_grid;

import std;

import fan.types.vector;

namespace fan::graphics::algorithm {

  void ray_grid_solve(
    fan::vec2 src,
    fan::vec2 dst,
    f32_t block_size,
    const std::function<void(ray_grid_result_t&)>& l
  ) {
    src /= block_size;
    dst /= block_size;
    fan::vec2 dir = dst - src;
    f32_t len_sq = dir.x * dir.x + dir.y * dir.y;
    if (len_sq == 0.0f) return;
    f32_t inv_len = 1.0f / std::sqrt(len_sq);
    dir *= inv_len;

    fan::vec2i gi((int)std::floor(src.x), (int)std::floor(src.y));
    fan::vec2i dst_gi((int)std::floor(dst.x), (int)std::floor(dst.y));
    fan::vec2i step(dir.x > 0 ? 1 : (dir.x < 0 ? -1 : 0), dir.y > 0 ? 1 : (dir.y < 0 ? -1 : 0));
    fan::vec2 t_max(
      step.x > 0 ? (gi.x + 1 - src.x) / dir.x : (step.x < 0 ? (src.x - gi.x) / -dir.x : 1e30f),
      step.y > 0 ? (gi.y + 1 - src.y) / dir.y : (step.y < 0 ? (src.y - gi.y) / -dir.y : 1e30f)
    );
    fan::vec2 t_delta(
      step.x != 0 ? 1.0f / std::abs(dir.x) : 1e30f,
      step.y != 0 ? 1.0f / std::abs(dir.y) : 1e30f
    );

    ray_grid_result_t r;
    while (true) {
      r.np = fan::vec2(gi) * block_size;
      r.at = r.np;
      r.gi = gi;
      l(r);
      if (gi == dst_gi) return;
      if (t_max.x < t_max.y) { t_max.x += t_delta.x; gi.x += step.x; }
      else                    { t_max.y += t_delta.y; gi.y += step.y; }
    }
  }

  std::vector<fan::vec2i> grid_raycast(const fan::line& line, const fan::vec2& tile_size) {
    std::vector<fan::vec2i> v;
    ray_grid_solve(line.first, line.second, tile_size.x, [&](ray_grid_result_t& result) {
      v.push_back(result.gi);
    });
    return v;
  }
}