module;
#if defined(FAN_2D)
#include <vector>
#include <unordered_map>
#include <limits>
#include <cstdint>
#include <algorithm>
 #include <numeric>
#endif

export module fan.graphics.spatial;

#if defined(FAN_2D)

import fan.types.vector;
import fan.print;
import fan.physics.types;

export namespace fan::graphics::spatial {

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
  void clean_removed(
    static_grid_t<id_t>& static_grid,
    dynamic_grid_t<id_t>& dynamic_grid,
    registry_t<id_t>& registry
  ) {
    for (auto& cell : static_grid.cells) {
      size_t w = 0;
      for (size_t i = 0; i < cell.objects.size(); ++i) {
        id_t id = cell.objects[i];
        if (registry.id_to_movement.find(id) != registry.id_to_movement.end()) {
          cell.objects[w++] = id;
        }
      }
      cell.objects.resize(w);
    }

    {
      size_t w = 0;
      for (size_t i = 0; i < registry.static_objects.size(); ++i) {
        id_t id = registry.static_objects[i];
        if (registry.id_to_movement.find(id) != registry.id_to_movement.end()) {
          registry.static_objects[w++] = id;
        }
      }
      registry.static_objects.resize(w);
    }

    {
      std::vector<id_t> dead;
      for (auto& [id, list] : registry.static_object_cells) {
        if (registry.id_to_movement.find(id) == registry.id_to_movement.end()) {
          dead.push_back(id);
          continue;
        }

        size_t w = 0;
        for (size_t i = 0; i < list.size(); ++i) {
          int idx = list[i];
          if (idx < 0 || idx >= (int)static_grid.cells.size()) {
            continue;
          }
          auto& cell = static_grid.cells[idx].objects;
          bool found = false;
          for (auto sid : cell) {
            if (sid == id) { found = true; break; }
          }
          if (found) {
            list[w++] = idx;
          }
        }
        list.resize(w);
      }
      for (auto id : dead) {
        registry.static_object_cells.erase(id);
      }
    }

    std::vector<object_t<id_t>> new_objects;
    new_objects.reserve(dynamic_grid.objects.size());

    std::unordered_map<id_t, uint32_t> new_id_to_object;
    new_id_to_object.reserve(dynamic_grid.id_to_object.size());

    for (auto& [id, movement] : registry.id_to_movement) {
      if ((movement_type_t)movement != movement_dynamic) {
        continue;
      }
      auto aabb_it = registry.aabb_cache.find(id);
      if (aabb_it == registry.aabb_cache.end()) {
        continue;
      }
      const auto& aabb = aabb_it->second;
      object_t<id_t> obj;
      obj.id = id;
      obj.min = aabb.min;
      obj.max = aabb.max;

      obj.cell = -1;
      uint32_t new_index = (uint32_t)new_objects.size();
      new_objects.push_back(obj);
      new_id_to_object[id] = new_index;
    }

    dynamic_grid.objects.swap(new_objects);
    dynamic_grid.id_to_object.swap(new_id_to_object);

    for (auto& cell : dynamic_grid.cells) {
      cell.clear();
    }

    for (uint32_t i = 0; i < dynamic_grid.objects.size(); ++i) {
      auto& obj = dynamic_grid.objects[i];

      fan::vec2 center = (obj.min + obj.max) * 0.5f;
      auto cell_coord = world_to_cell_clamped(
        center,
        dynamic_grid.world_min,
        dynamic_grid.cell_size,
        dynamic_grid.grid_size
      );
      int idx = cell_index(cell_coord, dynamic_grid.grid_size);
      obj.cell = idx;
      dynamic_grid.cells[idx].push_back(i);
    }

    {
      std::vector<id_t> dead;
      for (auto& [id, idx] : registry.id_to_dynamic) {
        auto it = dynamic_grid.id_to_object.find(id);
        if (it == dynamic_grid.id_to_object.end()) {
          dead.push_back(id);
          continue;
        }
        idx = it->second;
      }
      for (auto id : dead) {
        registry.id_to_dynamic.erase(id);
      }
    }

    {
      std::vector<id_t> dead;
      for (auto& [id, aabb] : registry.aabb_cache) {
        if (registry.id_to_movement.find(id) == registry.id_to_movement.end()) {
          dead.push_back(id);
        }
      }
      for (auto id : dead) {
        registry.aabb_cache.erase(id);
      }
    }
  }
}
#endif