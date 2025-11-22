module;

#include <functional>
#include <vector>

export module fan.graphics.algorithm.raycast_grid;

import fan.types.vector;
import fan.utility;
import fan.graphics.common_types;

export namespace fan::graphics::algorithm {
  struct ray_grid_result_t {
    fan::vec2 np;
    fan::vec2 at;
    fan::vec2_wrap_t<sint32_t> gi;
  };

  void ray_grid_solve(
    fan::vec2 line_src,
    fan::vec2 line_dst,
    f32_t GridBlockSize,
    const std::function<void(ray_grid_result_t&)>& l
  );

  std::vector<fan::vec2i> grid_raycast(const fan::line& line, const fan::vec2& tile_size);
}
