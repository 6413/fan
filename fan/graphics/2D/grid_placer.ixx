export module fan.graphics.grid_placer;

import std;
import fan.types;
import fan.types.vector;
import fan.math;
import fan.graphics.algorithm.raycast_grid;
import fan.graphics.common_types;

export namespace fan::graphics {
  inline fan::vec2i8 cell_direction(fan::vec2i delta) {
    if (std::abs(delta.x) > std::abs(delta.y)) {
      return {(sint8_t)(delta.x > 0 ? 1 : -1), 0};
    }
    if (delta.y) {
      return {0, (sint8_t)(delta.y > 0 ? 1 : -1)};
    }
    return {1, 0};
  }

  struct grid_placer_t {
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

    fan::vec2 tile_size;
    f32_t z_min = 1.f;
    f32_t z_max = 65535.f;
    f32_t z_offset = 0.f;
  };

  struct grid_drag_painter_t {
    void reset() {
      prev_pos = {std::numeric_limits<f32_t>::max(), std::numeric_limits<f32_t>::max()};
    }

    std::vector<fan::vec2i> update(const fan::vec2& pos, const fan::vec2& tile_size) {
      if (prev_pos.x == std::numeric_limits<f32_t>::max()) {
        prev_pos = pos;
        return {fan::vec2i((int)std::floor(pos.x / tile_size.x), (int)std::floor(pos.y / tile_size.y))};
      }
      if (prev_pos == pos) {
        return {};
      }
      auto cells = fan::graphics::algorithm::grid_raycast({prev_pos, pos}, tile_size);
      prev_pos = pos;
      return cells;
    }

    fan::vec2 prev_pos{std::numeric_limits<f32_t>::max(), std::numeric_limits<f32_t>::max()};
  };

  template <typename T>
  struct grid_brush_t {
    struct config_t {
      bool z_by_y = true;   
      f32_t z_offset = 0.f; 
      f32_t z_min = 1.f;    
      f32_t z_max = 65535.f;
    };

    grid_brush_t(fan::vec2 ts, config_t conf = {}) : tile_size(ts), config(conf) {}

    inline f32_t get_z_depth(f32_t ground_y) const {
      if (!config.z_by_y) {
        return config.z_offset;
      }
      return std::clamp(ground_y + config.z_offset, config.z_min, config.z_max);
    }

    std::vector<fan::vec2i> paint_update(const fan::vec2& pos) {
      std::vector<fan::vec2i> result;
      auto cells = painter.update(pos, tile_size);
      result.reserve(cells.size());
      for (auto& cell : cells) {
        if (!placed.contains(cell)) {
          result.push_back(cell);
        }
      }
      return result;
    }

    std::vector<fan::vec2i> paint_overwrite(const fan::vec2& pos) {
      return painter.update(pos, tile_size);
    }

    template <typename F>
    void paint_directional(const fan::vec2& mouse_pos, F&& on_cell) {
      for (auto& cell : paint_overwrite(mouse_pos)) {
        if (cell == last_cell) continue;
        auto facing = cell_direction(cell - last_cell);
        auto prev = last_cell;
        last_cell = cell;
        on_cell(cell, facing, prev);
      }
    }

    bool follow_path(fan::vec2& pos, fan::vec2i8& facing, f32_t speed, f32_t dt, auto&& get_facing) {
      f32_t dist = tile_size.x * speed * dt;
      f32_t half = tile_size.x * 0.5f;
      int steps = std::max(1, (int)(dist / (half * 0.5f)) + 1);
      f32_t step_dist = dist / steps;
      f32_t snap = dt * speed / steps;

      fan::vec2i cell = cell_at(pos);
      fan::vec2 center(cell.x * tile_size.x + half, cell.y * tile_size.y + half);

      for (int i = 0; i < steps; ++i) {
        pos += fan::vec2(facing) * step_dist;
        if (facing.x) pos.y = std::clamp(pos.y, cell.y * tile_size.y, cell.y * tile_size.y + tile_size.y);
        if (facing.y) pos.x = std::clamp(pos.x, cell.x * tile_size.x, cell.x * tile_size.x + tile_size.x);

        auto new_cell = cell_at(pos);
        if (new_cell != cell) {
          if (!get_facing(new_cell, facing)) return false;
          cell = new_cell;
          center = fan::vec2(cell.x * tile_size.x + half, cell.y * tile_size.y + half);
        }

        if (facing.x) pos.y += (center.y - pos.y) * snap;
        if (facing.y) pos.x += (center.x - pos.x) * snap;
      }
      return true;
    }

    void erase_update(const fan::vec2& pos) {
      for (auto& cell : painter.update(pos, tile_size)) {
        placed.erase(cell);
      }
    }

    template <typename... Args>
    void insert(fan::vec2i cell, Args&&... args) {
      placed.erase(cell);
      placed.try_emplace(cell, cell_center(cell), std::forward<Args>(args)...);
    }

    fan::vec2 cell_center(fan::vec2i cell) const {
      return {cell.x * tile_size.x + tile_size.x / 2.f, cell.y * tile_size.y + tile_size.y / 2.f};
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

    void reset() { 
      painter.reset(); 
      last_cell = {std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};
    }

    config_t config;
    fan::vec2 tile_size;
    grid_drag_painter_t painter;
    std::unordered_map<fan::vec2i, T> placed;
    fan::vec2i last_cell{std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};
  };
}
