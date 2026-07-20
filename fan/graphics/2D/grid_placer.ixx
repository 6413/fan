export module fan.graphics.grid_placer;

import std;
import fan.types.vector;
import fan.math;

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
}
