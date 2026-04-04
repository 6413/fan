module;
#if defined(FAN_2D)
#include <vector>
#include <unordered_map>
#include <limits>
#include <cstdint>
#include <algorithm>
#include <optional>
#endif

export module fan.spatial;

#if defined(FAN_2D)

import fan.types;
import fan.types.vector;
import fan.print.error;
import fan.physics.types;

export namespace fan::spatial {

  template<typename id_t>
  struct object_t {
    id_t id;
    fan::vec2 min;
    fan::vec2 max;
    int cell;
  };

  template<typename id_t>
  struct static_cell_t {
    std::vector<id_t> objects;
  };

  template<typename id_t>
  struct static_grid_t {
    fan::vec2 world_min = fan::vec2(-10000);
    fan::vec2 cell_size = fan::vec2(256);
    fan::vec2i grid_size = fan::vec2i(256);
    std::vector<static_cell_t<id_t>> cells;
  };

  template<typename id_t>
  struct dynamic_grid_t {
    fan::vec2 world_min = fan::vec2(-10000);
    fan::vec2 cell_size = fan::vec2(256);
    fan::vec2i grid_size = fan::vec2i(256);
    std::vector<std::vector<uint32_t>> cells;
    std::vector<object_t<id_t>> objects;
    std::unordered_map<id_t, uint32_t> id_to_object;
  };

  template<typename id_t>
  struct registry_t {
    std::vector<id_t> static_objects;
    std::unordered_map<id_t, std::vector<int>> static_object_cells;
    std::unordered_map<id_t, uint32_t> id_to_dynamic;
    std::unordered_map<id_t, uint8_t> id_to_movement;
    std::unordered_map<id_t, fan::physics::aabb_t> aabb_cache;
  };

  enum movement_type_t : uint8_t {
    movement_static,
    movement_dynamic
  };

  constexpr int cell_index(const fan::vec2i& c, const fan::vec2i& grid_size){
    return c.y * grid_size.x + c.x;
  }

  constexpr fan::vec2i world_to_cell_clamped(const fan::vec2& p, const fan::vec2& world_min, const fan::vec2& cell_size, const fan::vec2i& grid_size){
    fan::vec2i c = fan::vec2i((p - world_min) / cell_size);
    c.x = std::max(0, std::min(grid_size.x - 1, c.x));
    c.y = std::max(0, std::min(grid_size.y - 1, c.y));
    return c;
  }

  constexpr bool is_aabb_in_view(const fan::physics::aabb_t& aabb, const fan::vec2& view_min, const fan::vec2& view_max){
    return (aabb.max.x >= view_min.x) && (aabb.min.x <= view_max.x) && 
           (aabb.max.y >= view_min.y) && (aabb.min.y <= view_max.y);
  }

  template<typename id_t>
  void static_grid_init(static_grid_t<id_t>& grid, const fan::vec2& world_min, const fan::vec2& cell_size, const fan::vec2i& grid_size){
    grid.world_min = world_min;
    grid.cell_size = cell_size;
    grid.grid_size = grid_size;
    grid.cells.assign(grid_size.x * grid_size.y, {});
  }

  template<typename id_t>
  void dynamic_grid_init(dynamic_grid_t<id_t>& grid, const fan::vec2& world_min, const fan::vec2& cell_size, const fan::vec2i& grid_size){
    grid.world_min = world_min;
    grid.cell_size = cell_size;
    grid.grid_size = grid_size;
    grid.cells.assign(grid_size.x * grid_size.y, {});
    grid.objects.clear();
    grid.id_to_object.clear();
  }

  template<typename id_t>
  void add_object(registry_t<id_t>& registry, static_grid_t<id_t>& static_grid, dynamic_grid_t<id_t>& dynamic_grid, id_t id, const fan::physics::aabb_t& aabb, movement_type_t movement){
    registry.id_to_movement[id] = movement;
    registry.aabb_cache[id] = aabb;

    if (movement == movement_static) {
      registry.static_objects.push_back(id);

      auto min_cell = world_to_cell_clamped(aabb.min, static_grid.world_min, static_grid.cell_size, static_grid.grid_size);
      auto max_cell = world_to_cell_clamped(aabb.max, static_grid.world_min, static_grid.cell_size, static_grid.grid_size);
      int grid_width = static_grid.grid_size.x;

      auto& list = registry.static_object_cells[id];
      list.clear();

      for (int y = min_cell.y; y <= max_cell.y; ++y) {
        for (int x = min_cell.x; x <= max_cell.x; ++x) {
          int idx = y * grid_width + x;
          static_grid.cells[idx].objects.push_back(id);
          list.push_back(idx);
        }
      }
    }
    else {
      auto center = (aabb.min + aabb.max) * 0.5f;
      auto cell = world_to_cell_clamped(center, dynamic_grid.world_min, dynamic_grid.cell_size, dynamic_grid.grid_size);
      uint32_t obj_index = (uint32_t)dynamic_grid.objects.size();
      registry.id_to_dynamic[id] = obj_index;
      int idx = cell_index(cell, dynamic_grid.grid_size);
      dynamic_grid.objects.push_back({id, aabb.min, aabb.max, idx});

      dynamic_grid.cells[idx].push_back(obj_index);
      dynamic_grid.id_to_object[id] = obj_index;
    }
  }
  template<typename id_t>
  void remove_object(
    registry_t<id_t>& registry,
    static_grid_t<id_t>& static_grid,
    dynamic_grid_t<id_t>& dynamic_grid,
    id_t id
  ) {
    auto m_it = registry.id_to_movement.find(id);
    if (m_it == registry.id_to_movement.end()) {
      return;
    }

    movement_type_t m = (movement_type_t)m_it->second;

    if (m == movement_static) {
      auto cells_it = registry.static_object_cells.find(id);
      if (cells_it != registry.static_object_cells.end()) {
        for (int cell_idx : cells_it->second) {
          auto& v = static_grid.cells[cell_idx].objects;
          for (size_t i = 0; i < v.size(); ++i) {
            if (v[i] == id) {
              v[i] = v.back();
              v.pop_back();
              break;
            }
          }
        }
        registry.static_object_cells.erase(cells_it);
      }

      auto& v = registry.static_objects;
      for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] == id) {
          v[i] = v.back();
          v.pop_back();
          break;
        }
      }
    }
    else {
      auto dyn_it = registry.id_to_dynamic.find(id);
      if (dyn_it != registry.id_to_dynamic.end()) {
        uint32_t obj_idx = dyn_it->second;
        uint32_t last_idx = (uint32_t)dynamic_grid.objects.size() - 1;

        auto& obj = dynamic_grid.objects[obj_idx];
        auto& cell_vec = dynamic_grid.cells[obj.cell];

        for (size_t i = 0; i < cell_vec.size(); ++i) {
          if (cell_vec[i] == obj_idx) {
            cell_vec[i] = cell_vec.back();
            cell_vec.pop_back();
            break;
          }
        }

        if (obj_idx != last_idx) {
          auto& last_obj = dynamic_grid.objects[last_idx];
          dynamic_grid.objects[obj_idx] = last_obj;

          registry.id_to_dynamic[last_obj.id] = obj_idx;
          dynamic_grid.id_to_object[last_obj.id] = obj_idx;

          auto& last_cell_vec = dynamic_grid.cells[last_obj.cell];
          for (size_t i = 0; i < last_cell_vec.size(); ++i) {
            if (last_cell_vec[i] == last_idx) {
              last_cell_vec[i] = obj_idx;
              break;
            }
          }
        }

        dynamic_grid.objects.pop_back();
        registry.id_to_dynamic.erase(dyn_it);
      }

      dynamic_grid.id_to_object.erase(id);
    }

    registry.id_to_movement.erase(m_it);
    registry.aabb_cache.erase(id);
  }


  template<typename id_t>
  void update_dynamic_object(registry_t<id_t>& registry, dynamic_grid_t<id_t>& dynamic_grid, id_t id, const fan::physics::aabb_t& new_aabb) {
    auto dyn_it = registry.id_to_dynamic.find(id);
    if (dyn_it == registry.id_to_dynamic.end()) {
      return;
    }

    uint32_t obj_idx = dyn_it->second;
    auto& obj = dynamic_grid.objects[obj_idx];
    auto center = (new_aabb.min + new_aabb.max) * 0.5f;
    auto cell = world_to_cell_clamped(center, dynamic_grid.world_min, dynamic_grid.cell_size, dynamic_grid.grid_size);
    int new_cell = cell_index(cell, dynamic_grid.grid_size);

    if (new_cell != obj.cell) {
      auto& old = dynamic_grid.cells[obj.cell];
      for (size_t i = 0; i < old.size(); ++i) {
        if (old[i] == obj_idx) {
          old[i] = old.back();
          old.pop_back();
          break;
        }
      }
      dynamic_grid.cells[new_cell].push_back(obj_idx);
      obj.cell = new_cell;
    }

    obj.min = new_aabb.min;
    obj.max = new_aabb.max;
    registry.aabb_cache[id] = new_aabb;
  }

  template<typename id_t, typename callback_t>
  void query_area(const static_grid_t<id_t>& static_grid, const dynamic_grid_t<id_t>& dynamic_grid, const fan::vec2& view_min, const fan::vec2& view_max, callback_t&& callback){
    auto min_cell = world_to_cell_clamped(view_min, static_grid.world_min, static_grid.cell_size, static_grid.grid_size);
    auto max_cell = world_to_cell_clamped(view_max, static_grid.world_min, static_grid.cell_size, static_grid.grid_size);
    int grid_width = static_grid.grid_size.x;

    for (int y = min_cell.y; y <= max_cell.y; ++y) {
      int row_offset = y * grid_width;
      for (int x = min_cell.x; x <= max_cell.x; ++x) {
        int idx = row_offset + x;
        for (auto id : static_grid.cells[idx].objects) {
          callback(id);
        }
      }
    }

    for (auto& obj : dynamic_grid.objects) {
      if (is_aabb_in_view({obj.min, obj.max}, view_min, view_max)) {
        callback(obj.id);
      }
    }
  }

  template<typename id_t, typename callback_t>
  void query_radius(const dynamic_grid_t<id_t>& dynamic_grid, const fan::vec2& center, f32_t radius, callback_t&& callback){
    fan::vec2 min = center - fan::vec2(radius);
    fan::vec2 max = center + fan::vec2(radius);
    auto min_cell = world_to_cell_clamped(min, dynamic_grid.world_min, dynamic_grid.cell_size, dynamic_grid.grid_size);
    auto max_cell = world_to_cell_clamped(max, dynamic_grid.world_min, dynamic_grid.cell_size, dynamic_grid.grid_size);

    f32_t radius_sq = radius * radius;

    for (int y = min_cell.y; y <= max_cell.y; ++y) {
      for (int x = min_cell.x; x <= max_cell.x; ++x) {
        int idx = cell_index(fan::vec2i(x, y), dynamic_grid.grid_size);
        for (auto obj_idx : dynamic_grid.cells[idx]) {
          auto& obj = dynamic_grid.objects[obj_idx];
          fan::vec2 obj_center = (obj.min + obj.max) * 0.5f;
          if ((obj_center - center).length_squared() <= radius_sq) {
            callback(obj.id);
          }
        }
      }
    }
  }

  template <typename map_t>
  void clear_rehash(map_t& m) {
    m.clear();
    m.rehash(0);
  }

  template <typename vec_t>
  void clear_shrink(vec_t& v) {
    vec_t empty;
    v.swap(empty);
  }

  template<typename id_t>
  void reset(
    static_grid_t<id_t>& static_grid,
    dynamic_grid_t<id_t>& dynamic_grid,
    registry_t<id_t>& registry
  ) {
    clear_shrink(static_grid.cells);
    static_grid.cells.resize(
      static_grid.grid_size.x * static_grid.grid_size.y
    );

    clear_shrink(dynamic_grid.cells);
    dynamic_grid.cells.resize(
      dynamic_grid.grid_size.x * dynamic_grid.grid_size.y
    );
    clear_shrink(dynamic_grid.objects);
    clear_rehash(dynamic_grid.id_to_object);

    clear_shrink(registry.static_objects);
    clear_rehash(registry.static_object_cells);
    clear_rehash(registry.id_to_dynamic);
    clear_rehash(registry.id_to_movement);
    clear_rehash(registry.aabb_cache);
  }

  template<typename id_t>
  void upsert_object(
    registry_t<id_t>& registry,
    static_grid_t<id_t>& static_grid,
    dynamic_grid_t<id_t>& dynamic_grid,
    id_t id,
    const fan::physics::aabb_t& aabb,
    movement_type_t movement
  ) {
    auto it = registry.id_to_movement.find(id);
    if (it == registry.id_to_movement.end()) {
      add_object(registry, static_grid, dynamic_grid, id, aabb, movement);
      return;
    }
    if (movement == movement_static) {
      registry.aabb_cache[id] = aabb;
      remove_object(registry, static_grid, dynamic_grid, id);
      add_object(registry, static_grid, dynamic_grid, id, aabb, movement);
      return;
    }
    update_dynamic_object(registry, dynamic_grid, id, aabb);
  }

  template<typename id_t, typename callback_t>
  void query_aabb(
    const dynamic_grid_t<id_t>& grid,
    const fan::physics::aabb_t& aabb,
    callback_t&& cb
  ) {
    fan::vec2 min = aabb.min;
    fan::vec2 max = aabb.max;
    auto c0 = world_to_cell_clamped(min, grid.world_min, grid.cell_size, grid.grid_size);
    auto c1 = world_to_cell_clamped(max, grid.world_min, grid.cell_size, grid.grid_size);
    for (int y = c0.y; y <= c1.y; ++y) {
      for (int x = c0.x; x <= c1.x; ++x) {
        int idx = cell_index({x, y}, grid.grid_size);
        for (auto oi : grid.cells[idx]) {
          auto& o = grid.objects[oi];
          if (!(o.max.x < min.x || o.min.x > max.x || o.max.y < min.y || o.min.y > max.y)) {
            cb(o.id);
          }
        }
      }
    }
  }

  template<typename id_t, typename predicate_t>
  std::optional<id_t> query_nearest(
    const dynamic_grid_t<id_t>& grid,
    const fan::vec2& center,
    f32_t radius,
    predicate_t&& pred
  ) {
    fan::vec2 min = center - fan::vec2(radius);
    fan::vec2 max = center + fan::vec2(radius);
    auto c0 = world_to_cell_clamped(min, grid.world_min, grid.cell_size, grid.grid_size);
    auto c1 = world_to_cell_clamped(max, grid.world_min, grid.cell_size, grid.grid_size);
    f32_t best = radius * radius;
    std::optional<id_t> out;
    for (int y = c0.y; y <= c1.y; ++y) {
      for (int x = c0.x; x <= c1.x; ++x) {
        int idx = cell_index({x, y}, grid.grid_size);
        for (auto oi : grid.cells[idx]) {
          auto& o = grid.objects[oi];
          if (!pred(o.id)) continue;
          fan::vec2 p = (o.min + o.max) * 0.5f;
          f32_t d = (p - center).length_squared();
          if (d < best) {
            best = d;
            out = o.id;
          }
        }
      }
    }
    return out;
  }

  template <typename id_t, typename resolve_pos_fn_t>
  fan::vec2 separation_force(
    dynamic_grid_t<id_t>& grid,
    id_t                  self_id,
    fan::vec2             pos,
    f32_t                 radius,
    resolve_pos_fn_t&& resolve_fn) {
    fan::vec2 sep {};
    query_radius(grid, pos, radius, [&](id_t id) {
      if (id == self_id) return;
      fan::vec2 other = resolve_fn(id);
      fan::vec2 d = pos - other;
      f32_t len = d.length();
      if (len > 0.001f && len < radius)
        sep += d / len * (radius - len);
    });
    return sep;
  }

  template<typename id_t>
  struct world_t {
    static_grid_t<id_t>  static_grid;
    dynamic_grid_t<id_t> dynamic_grid;
    registry_t<id_t>     registry;

    void init(const fan::vec2& world_min, const fan::vec2& cell_size, const fan::vec2i& grid_size) {
      static_grid_init(static_grid, world_min, cell_size, grid_size);
      dynamic_grid_init(dynamic_grid, world_min, cell_size, grid_size);
    }

    void reset() { fan::spatial::reset(static_grid, dynamic_grid, registry); }

    void upsert(id_t id, const fan::physics::aabb_t& aabb, movement_type_t m) {
      upsert_object(registry, static_grid, dynamic_grid, id, aabb, m);
    }

    void remove(id_t id) {
      remove_object(registry, static_grid, dynamic_grid, id);
    }

    template<typename cb_t>
    void query_radius(const fan::vec2& center, f32_t r, cb_t&& cb) {
      fan::vec2 min = center - fan::vec2(r);
      fan::vec2 max = center + fan::vec2(r);
      f32_t r_sq = r * r;
      std::vector<id_t> seen;
      seen.reserve(64);

      fan::spatial::query_area(static_grid, dynamic_grid, min, max, [&](id_t id) {
        for (auto& s : seen) if (s == id) return;
        seen.push_back(id);

        auto it = registry.aabb_cache.find(id);
        if (it != registry.aabb_cache.end()) {
          fan::vec2 obj_center = (it->second.min + it->second.max) * 0.5f;
          fan::vec2 extents = (it->second.max - it->second.min) * 0.5f;
          fan::vec2 closest = center.clamp(obj_center - extents, obj_center + extents);
          if ((closest - center).length_squared() <= r_sq) {
            cb(id);
          }
        }
      });
    }

    template <typename cb_t>
    void query_point(const fan::vec2& p, cb_t&& cb) {
      auto c = world_to_cell_clamped(p, static_grid.world_min, static_grid.cell_size, static_grid.grid_size);
      int idx = cell_index(c, static_grid.grid_size);
      for (auto id : static_grid.cells[idx].objects) {
        auto it = registry.aabb_cache.find(id);
        if (it != registry.aabb_cache.end() && p.x >= it->second.min.x && p.x <= it->second.max.x && p.y >= it->second.min.y && p.y <= it->second.max.y) cb(id);
      }
      for (auto oi : dynamic_grid.cells[idx]) {
        auto& o = dynamic_grid.objects[oi];
        if (p.x >= o.min.x && p.x <= o.max.x && p.y >= o.min.y && p.y <= o.max.y) cb(o.id);
      }
    }

    template<typename pred_t>
    std::optional<id_t> query_nearest(const fan::vec2& center, f32_t r, pred_t&& pred) {
      std::optional<id_t> best_id;
      f32_t best_dist = r * r;
      fan::vec2 min = center - fan::vec2(r);
      fan::vec2 max = center + fan::vec2(r);
      std::vector<id_t> seen;
      seen.reserve(64);

      fan::spatial::query_area(static_grid, dynamic_grid, min, max, [&](id_t id) {
        for (auto& s : seen) if (s == id) return;
        seen.push_back(id);

        if (!pred(id)) return;
        auto it = registry.aabb_cache.find(id);
        if (it != registry.aabb_cache.end()) {
          fan::vec2 obj_center = (it->second.min + it->second.max) * 0.5f;
          fan::vec2 extents = (it->second.max - it->second.min) * 0.5f;
          fan::vec2 closest = center.clamp(obj_center - extents, obj_center + extents);
          f32_t d = (closest - center).length_squared();
          if (d <= best_dist) {
            best_dist = d;
            best_id = id;
          }
        }
      });
      return best_id;
    }

    template<typename resolve_fn_t>
    fan::vec2 separation_force(id_t id, const fan::vec2& pos, f32_t r, resolve_fn_t&& fn) {
      return fan::spatial::separation_force(dynamic_grid, id, pos, r, std::forward<resolve_fn_t>(fn));
    }

    template <typename callback_t>
    void raycast(const fan::vec2& start, const fan::vec2& end, callback_t&& callback) {
      fan::vec2 view_min(std::min(start.x, end.x), std::min(start.y, end.y));
      fan::vec2 view_max(std::max(start.x, end.x), std::max(start.y, end.y));
      std::vector<id_t> seen;
      seen.reserve(64);

      fan::spatial::query_area(static_grid, dynamic_grid, view_min, view_max, [&](id_t id) {
        for (auto& s : seen) if (s == id) return;
        seen.push_back(id);

        auto it = registry.aabb_cache.find(id);
        if (it != registry.aabb_cache.end()) {
          fan::vec2 dir = end - start;
          f32_t tmin = 0.f, tmax = 1.f;
          auto check = [&](f32_t p, f32_t d, f32_t bmin, f32_t bmax) {
            if (std::abs(d) < 1e-6f) { return p >= bmin && p <= bmax; }
            f32_t t0 = (bmin - p) / d, t1 = (bmax - p) / d;
            if (t0 > t1) { std::swap(t0, t1); }
            tmin = std::max(tmin, t0);
            tmax = std::min(tmax, t1);
            return tmax >= tmin;
          };
          if (check(start.x, dir.x, it->second.min.x, it->second.max.x) &&
              check(start.y, dir.y, it->second.min.y, it->second.max.y)) {
            callback(id);
          }
        }
      });
    }
  };

  template <typename id_t, typename registry_t>
  void auto_clean(world_t<id_t>& world, registry_t& reg) {
    reg.on_destroy_hooks.push_back([&world](uint32_t id) {
      world.remove(id);
    });
  }

  template <typename pos_t, typename tag_t, typename world_t, typename registry_t>
  constexpr void sync_grid(registry_t& reg, world_t& world, fan::vec2 half_size) {
    reg.template each<pos_t, tag_t>([&](uint32_t e, pos_t& pos, tag_t&) {
      world.upsert(e, fan::physics::aabb_t::from_center(pos.v, half_size), spatial::movement_dynamic);
    });
  }

} // namespace fan::graphics::spatial
#endif