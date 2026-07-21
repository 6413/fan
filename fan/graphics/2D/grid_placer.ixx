export module fan.graphics.grid_placer;

import std;
import fan.types.vector;
import fan.math;
import fan.graphics.algorithm.raycast_grid;
import fan.graphics.common_types;

export namespace fan::graphics {
  struct grid_placer_t {
    fan::vec2 tile_size;
    
    float z_min = 1.f;
    float z_max = 65535.f;
    float z_offset = 2000.f;

    grid_placer_t(fan::vec2 ts) : tile_size(ts) {}

    inline float get_z_depth(float ground_y) const {
      return std::max(z_min, std::min(z_max, ground_y + z_offset));
    }

    inline fan::vec2i get_cell(fan::vec2 pos) const {
      return pos.grid_floor(tile_size);
    }

    inline fan::vec2i cells_occupied(fan::vec2 custom_scale) const {
      return {std::max(1, (int)std::round(custom_scale.x)),
              std::max(1, (int)std::round(custom_scale.y))};
    }

    inline fan::vec3 get_placement(fan::vec2i cell, fan::vec2 object_size, float custom_scale_x = 1.0f) const {
      auto co = cells_occupied({custom_scale_x, 1});
      fan::vec2 pos(
        cell.x + (co.x * tile_size.x) / 2.f,
        cell.y + tile_size.y - object_size.y / 2.f
      );
      return fan::vec3(pos, get_z_depth(pos.y + object_size.y / 2.f));
    }
    
    inline fan::vec2 get_fit_size(fan::vec2 original_size, fan::vec2 custom_scale = 1.0f) const {
      fan::vec2 base_size(tile_size.x, tile_size.x * (original_size.y / original_size.x));
      return base_size * custom_scale;
    }
  };

  struct grid_drag_painter_t {
    fan::vec2 prev_pos{std::numeric_limits<f32_t>::max(), std::numeric_limits<f32_t>::max()};
    fan::vec2i prev_cell{std::numeric_limits<int>::min(), std::numeric_limits<int>::min()};

    void reset() {
      prev_pos = {std::numeric_limits<f32_t>::max(), std::numeric_limits<f32_t>::max()};
      prev_cell = {std::numeric_limits<int>::min(), std::numeric_limits<int>::min()};
    }

    std::vector<fan::vec2i> update(const fan::vec2& pos, const fan::vec2& tile_size) {
      if (prev_pos.x == std::numeric_limits<f32_t>::max()) {
        prev_pos = pos;
        prev_cell = pos.grid_floor(tile_size);
        return {prev_cell};
      }
      auto cells = fan::graphics::algorithm::grid_raycast({prev_pos, pos}, tile_size);
      prev_pos = pos;
      if (!cells.empty()) prev_cell = cells.back() * tile_size;
      for (auto& c : cells) c = c * tile_size;
      return cells;
    }
  };
}
