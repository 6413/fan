module;

#include <fan/types/types.h>
#include <cmath>
#include <vector>

export module fan.graphics.algorithm.raycast_grid;

export import fan.types.vector;
import fan.graphics.common_types;

export namespace fan {
  namespace graphics {
    namespace algorithm {
      struct ray_grid_result_t {
        fan::vec2 np;
        fan::vec2 at;
        fan::vec2_wrap_t<sint32_t> gi;
      };

      void ray_grid_solve(
        fan::vec2 line_src,
        fan::vec2 line_dst,
        f32_t GridBlockSize,
        auto l
      ) {
        line_src /= GridBlockSize;
        line_dst /= GridBlockSize;

        fan::vec2 direction = (line_dst - line_src).normalized();

        for (uint32_t d = 0; d < fan::vec2::size(); d++) {
          if (direction[d] == 0) {
            direction[d] = 0.00001;
          }
        }

        ray_grid_result_t grid_result;
        grid_result.np = line_src;
        grid_result.at = grid_result.np;
        for (uint32_t d = 0; d < fan::vec2::size(); d++) {
          grid_result.gi[d] = grid_result.at[d] + (grid_result.at[d] < f32_t(0) ? f32_t(-1) : f32_t(0));
        }

        fan::vec2 r = grid_result.at - grid_result.gi;
        while (1) {
          {
            if ((grid_result.at - line_src).length() >= (line_dst - line_src).length()) {
              return;
            }

            l(grid_result);

          }

          fan::vec2 left;
          for (uint32_t i = 0; i < fan::vec2::size(); i++) {
            if (direction[i] > 0) {
              left[i] = f32_t(1) - r[i];
            }
            else {
              left[i] = r[i];
            }
          }
          fan::vec2 multiplers = left / direction.abs();

          f32_t min_multipler = multiplers.min();
          for (uint32_t i = 0; i < fan::vec2::size(); i++) {
            if (multiplers[i] == min_multipler) {
              grid_result.gi[i] += copysign((sint32_t)1, direction[i]);
              r[i] -= copysign((f32_t)1, direction[i]);
            }
          }
          fan::vec2 min_dir = direction * min_multipler;
          grid_result.at += min_dir;
          r += min_dir;
        }
      }

      std::vector<fan::vec2i> grid_raycast(const fan::line& line, const fan::vec2& tile_size) {
        std::vector<fan::vec2i> v;
        ray_grid_solve(line.first, line.second, tile_size.x, [&](ray_grid_result_t& result)
          {
            v.push_back(result.gi);
          });
        return v;
      }
    }
  }
}