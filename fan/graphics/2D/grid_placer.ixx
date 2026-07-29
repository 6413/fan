export module fan.graphics.grid_placer;

import std;
import fan.types.vector;
import fan.math;
import fan.graphics.algorithm.raycast_grid;
import fan.graphics.common_types;

export namespace fan::graphics {
  struct grid_placer_t {
    fan::vec2 tile_size;
    
    f32_t z_min = 1.f;
    f32_t z_max = 65535.f;
    f32_t z_offset = 0.f;

    grid_placer_t(fan::vec2 ts) : tile_size(ts) {}

    inline f32_t get_z_depth(f32_t ground_y) const {
      return std::clamp(ground_y + z_offset, z_min, z_max);
    }

    inline fan::vec2i get_cell(fan::vec2 pos) const {
      return pos.grid_floor(tile_size);
    }

    inline fan::vec2i cells_occupied(fan::vec2 custom_scale) const {
      return {std::max(1, (int)std::round(custom_scale.x)),
              std::max(1, (int)std::round(custom_scale.y))};
    }

    inline fan::vec3 get_placement(fan::vec2i cell, fan::vec2 object_size, f32_t custom_scale_x = 1.f) const {
      auto co = cells_occupied({custom_scale_x, 1});
      fan::vec2 pos(
        cell.x + (co.x * tile_size.x) / 2.f,
        cell.y + tile_size.y - object_size.y / 2.f
      );
      return fan::vec3(pos, get_z_depth(cell.y / tile_size.y + 1));
    }
    
    inline fan::vec2 get_fit_size(fan::vec2 original_size, fan::vec2 custom_scale = 1.f) const {
      return fan::vec2(tile_size.x, tile_size.x * (original_size.y / original_size.x)) * custom_scale;
    }
  };

  struct grid_drag_painter_t {
    fan::vec2 prev_pos{std::numeric_limits<f32_t>::max(), std::numeric_limits<f32_t>::max()};

    void reset() {
      prev_pos = {std::numeric_limits<f32_t>::max(), std::numeric_limits<f32_t>::max()};
    }

    std::vector<fan::vec2i> update(const fan::vec2& pos, const fan::vec2& tile_size) {
      if (prev_pos.x == std::numeric_limits<f32_t>::max()) {
        prev_pos = pos;
        return {fan::vec2i((int)std::floor(pos.x / tile_size.x), (int)std::floor(pos.y / tile_size.y))};
      }
      if (prev_pos == pos) return {};
      auto cells = fan::graphics::algorithm::grid_raycast({prev_pos, pos}, tile_size);
      prev_pos = pos;
      return cells;
    }
  };

  template <typename T>
  struct grid_brush_t {
    struct config_t {
      bool z_by_y = true;   
      f32_t z_offset = 0.f; 
      f32_t z_min = 1.f;    
      f32_t z_max = 65535.f;
    } config;

    fan::vec2 tile_size;
    grid_drag_painter_t painter;
    std::unordered_map<fan::vec2i, T> placed;

    grid_brush_t(fan::vec2 ts, config_t conf = {}) : tile_size(ts), config(conf) {}

    inline f32_t get_z_depth(f32_t ground_y) const {
      if (!config.z_by_y) return config.z_offset;
      return std::clamp(ground_y + config.z_offset, config.z_min, config.z_max);
    }

    std::vector<fan::vec3> paint_update(const fan::vec2& pos) {
      std::vector<fan::vec3> result;
      auto cells = painter.update(pos, tile_size);
      result.reserve(cells.size());
      for (auto& cell : cells) {
        if (!placed.contains(cell)) {
          result.push_back(fan::vec3(
            cell.x * tile_size.x + tile_size.x / 2.f,
            cell.y * tile_size.y + tile_size.y / 2.f,
            get_z_depth(cell.y + 1)
          ));
        }
      }
      return result;
    }

    std::vector<fan::vec3> paint_overwrite(const fan::vec2& pos) {
      std::vector<fan::vec3> result;
      auto cells = painter.update(pos, tile_size);
      result.reserve(cells.size());
      for (auto& cell : cells) {
        result.push_back(fan::vec3(
          cell.x * tile_size.x + tile_size.x / 2.f,
          cell.y * tile_size.y + tile_size.y / 2.f,
          get_z_depth(cell.y + 1)
        ));
      }
      return result;
    }

    void erase_update(const fan::vec2& pos) {
      for (auto& cell : painter.update(pos, tile_size)) {
        placed.erase(cell);
      }
    }

    void insert(const fan::vec3& pos, T obj) {
      placed.insert_or_assign(cell_at(pos), std::move(obj));
    }

    fan::vec2i cell_at(const fan::vec2& pos) const { return {(int)std::floor(pos.x / tile_size.x), (int)std::floor(pos.y / tile_size.y)}; }
    fan::vec2i cell_at(const fan::vec3& pos) const { return cell_at(fan::vec2(pos.x, pos.y)); }

    T* get(fan::vec2i cell) { auto it = placed.find(cell); return it != placed.end() ? &it->second : nullptr; }
    const T* get(fan::vec2i cell) const { auto it = placed.find(cell); return it != placed.end() ? &it->second : nullptr; }
    T* get(const fan::vec3& pos) { return get(cell_at(pos)); }
    const T* get(const fan::vec3& pos) const { return get(cell_at(pos)); }
    T* get(const fan::vec2& pos) { return get(cell_at(pos)); }
    const T* get(const fan::vec2& pos) const { return get(cell_at(pos)); }

    bool has(fan::vec2i cell) const { return placed.contains(cell); }

    void reset() { painter.reset(); }
  };

  inline fan::vec2i8 cell_direction(fan::vec2i delta) {
    if (std::abs(delta.x) > std::abs(delta.y)) return {(int8_t)(delta.x > 0 ? 1 : -1), 0};
    if (delta.y) return {0, (int8_t)(delta.y > 0 ? 1 : -1)};
    return {1, 0};
  }
}