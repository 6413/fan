module;

#if defined(FAN_2D)
  #include <cstdint>
  #include <algorithm>
  #include <string>
#endif

module fan.graphics.culling;

#if defined(FAN_2D)

import fan.graphics.shapes;
import fan.print;

namespace fan::graphics::culling {

  template <typename... Args>
  static inline void dbg(const char* tag, Args&&... args) {
    //std::string t(tag);
    /* if (t.contains("show") || t.contains("[push_vram]") || t.contains("keep") || t.contains("test") || t.contains("span") || t.contains("[cull]") || t.contains("remove") || t == "[remove_static_shape_from_grid] AABB" || t == "[AABB]" || t == "[add_static_shape_to_grid]" || t == "[add_shape]" || t == "[remove_static_shape_from_grid]" || t == "[update_dynamic]" || t == "[zremove_shape]" || t == "[add_static_shape_to_grid] AABB" || t == "[dyn-test]" || t.contains("->") || t == "[update_dynamic] new AABB") {
    return;
    }

    fan::print_throttled(tag, std::forward<Args>(args)...);*/
  }

  constexpr int cell_index(const fan::vec2i& c, const fan::vec2i& grid_size) {
    return c.y * grid_size.x + c.x;
  }
  constexpr fan::vec2i world_to_cell_clamped(const fan::vec2& p, const fan::vec2& world_min, const fan::vec2& cell_size, const fan::vec2i& grid_size) {
    fan::vec2i c = fan::vec2i((p - world_min) / cell_size);
    c.x = std::max(0, std::min(grid_size.x - 1, c.x));
    c.y = std::max(0, std::min(grid_size.y - 1, c.y));
    return c;
  }
  static inline void normalize_view_bounds(fan::vec2& vmin, fan::vec2& vmax) {
    fan::vec2 nmin(std::min(vmin.x, vmax.x), std::min(vmin.y, vmax.y));
    fan::vec2 nmax(std::max(vmin.x, vmax.x), std::max(vmin.y, vmax.y));
    vmin = nmin;
    vmax = nmax;
  }
  static inline void dbg_aabb(const char* tag, uint32_t sid, const fan::physics::aabb_t& a) {
    dbg(tag, "sid", sid, "min", a.min.x, a.min.y, "max", a.max.x, a.max.y);
  }
  static inline void dbg_cells(const char* tag, const fan::vec2i& minc, const fan::vec2i& maxc) {
    dbg(tag, "cells [", minc.x, ",", minc.y, "]..[", maxc.x, ",", maxc.y, "]");
  }
  fan::physics::aabb_t get_shape_aabb(shaper_t::ShapeID_t sid) {
    if (sid.iic()) {
      dbg("[AABB]", "warning: empty sid");
      return {};
    }
    fan::vec2 position{};
    fan::vec2 size{};
    bool has_bounds = false;
    const char* matched = "none";
    g_shapes->visit_shape_draw_data(sid.NRI, [&](auto& properties) {
      if constexpr (requires { properties.position; properties.size; }) {
        position = properties.position;
        size = properties.size;
        has_bounds = true;
        matched = "size";
      }
      else if constexpr (requires { properties.position; properties.radius; }) {
        position = properties.position;
        size = fan::vec2(properties.radius, properties.radius);
        has_bounds = true;
        matched = "radius";
      }
      else if constexpr (requires { properties.src; properties.dst; }) {
        position = (properties.src + properties.dst) * 0.5f;
        size = fan::vec2(std::abs(properties.dst.x - properties.src.x), std::abs(properties.dst.y - properties.src.y)) * 0.5f;
        has_bounds = true;
        matched = "srcdst";
      }
      else {
        fan::print_warning("shape aabb not found");
      }
    });
    if (!has_bounds) {
      dbg("[AABB]", "sid", sid.NRI, "NO_BOUNDS properties match");
      return fan::physics::aabb_t{};
    }
    fan::physics::aabb_t aabb{position - size, position + size};
    dbg("[AABB]", "sid", sid.NRI, matched, "pos", position.x, position.y, "size", size.x, size.y);
    dbg_aabb("         ->", sid.NRI, aabb);
    return aabb;
  }
  void static_grid_init(static_grid_t& grid, const fan::vec2& world_min, const fan::vec2& cell_size, const fan::vec2i& grid_size) {
    grid.world_min = world_min;
    grid.cell_size = cell_size;
    grid.grid_size = grid_size;
    grid.cells.clear();
    grid.cells.resize(grid_size.x * grid_size.y);
    dbg("[static_grid_init]", "world_min", world_min.x, world_min.y, "cell_size", cell_size.x, cell_size.y, "grid_size", grid_size.x, grid_size.y, "cells", (uint32_t)grid.cells.size());
  }
  void dynamic_grid_init(dynamic_grid_t& grid, const fan::vec2& world_min, const fan::vec2& cell_size, const fan::vec2i& grid_size) {
    grid.world_min = world_min;
    grid.cell_size = cell_size;
    grid.grid_size = grid_size;
    grid.cells.clear();
    grid.cells.resize(grid_size.x * grid_size.y);
    grid.objects.clear();
    grid.shapeid_to_object.clear();
    dbg("[dynamic_grid_init]", "world_min", world_min.x, world_min.y, "cell_size", cell_size.x, cell_size.y, "grid_size", grid_size.x, grid_size.y, "cells", (uint32_t)grid.cells.size());
  }
  static inline void mark_cameras_dirty(culling_t& culling) {
    for (auto& [cam_id, cam_state] : culling.camera_states) {
      cam_state.view_dirty = true;
    }
  }
  static inline void ensure_registry_size(culling_t& culling, uint32_t nr) {
    if (nr >= culling.registry.shapeid_to_movement.size()) {
      size_t new_size = nr + 1;
      culling.registry.shapeid_to_movement.resize(new_size, movement_dynamic);
      culling.registry.shapeid_to_dynamic.resize(new_size, std::numeric_limits<uint32_t>::max());
      culling.registry.aabb_cache.resize(new_size);
      dbg("[ensure_registry_size]", "resize to", (uint32_t)new_size);
    }
  }
  static inline void ensure_camera_visible_size(per_camera_state_t& cam_state, uint32_t nr) {
    if (nr >= cam_state.visible.size()) {
      cam_state.visible.resize(nr + 1, 0);
    }
  }
  static inline bool is_aabb_in_view(const fan::physics::aabb_t& aabb, const fan::vec2& view_min, const fan::vec2& view_max) {
    return (aabb.max.x >= view_min.x) && (aabb.min.x <= view_max.x) && (aabb.max.y >= view_min.y) && (aabb.min.y <= view_max.y);
  }
  void update_shape_vram_if_camera_matches(shaper_t::ShapeID_t sid, const fan::graphics::camera_t& culling_camera, bool push) {
    auto* shape_ptr = (fan::graphics::shapes::shape_t*)&sid;
    auto sc = shape_ptr->get_camera();
    if (sc == culling_camera) {
      dbg(push ? "[push_vram]" : "[erase_vram]", "sid", sid.NRI, "cam", sc.NRI);
      if (push) {
        shape_ptr->push_vram();
      }
      else {
        shape_ptr->erase_vram();
      }
    }
    else {
      dbg("[camera_mismatch]", "sid", sid.NRI, "shape_cam", sc.NRI, "cull_cam", culling_camera.NRI);
    }
  }
  void add_static_shape_to_grid(culling_t& culling, shaper_t::ShapeID_t sid) {
    uint32_t nr = sid.NRI;
    if (nr >= culling.registry.aabb_cache.size()) {
      culling.registry.aabb_cache.resize(nr + 1);
      dbg("[add_static_shape_to_grid]", "resize aabb_cache", (uint32_t)culling.registry.aabb_cache.size());
    }
    fan::physics::aabb_t aabb = get_shape_aabb(sid);
    culling.registry.aabb_cache[nr] = aabb;
    auto min_cell = world_to_cell_clamped(aabb.min, culling.static_grid.world_min, culling.static_grid.cell_size, culling.static_grid.grid_size);
    auto max_cell = world_to_cell_clamped(aabb.max, culling.static_grid.world_min, culling.static_grid.cell_size, culling.static_grid.grid_size);
    dbg_aabb("[add_static_shape_to_grid] AABB", nr, aabb);
    dbg_cells("                               span", min_cell, max_cell);
    int grid_width = culling.static_grid.grid_size.x;
    for (int y = min_cell.y; y <= max_cell.y; ++y) {
      for (int x = min_cell.x; x <= max_cell.x; ++x) {
        culling.static_grid.cells[y * grid_width + x].shapes.push_back(sid);
        dbg("  -> cell", x, y, "add sid", nr);
      }
    }
    mark_cameras_dirty(culling);
  }
  static inline void check_and_push_shape_to_cameras(culling_t& culling, shaper_t::ShapeID_t sid, const fan::physics::aabb_t& aabb) {
    uint32_t nr = sid.NRI;
    for (auto& [cam_id, cam_state] : culling.camera_states) {
      if (cam_state.cached_view_min != cam_state.cached_view_max && is_aabb_in_view(aabb, cam_state.cached_view_min, cam_state.cached_view_max)) {
        fan::graphics::camera_t cam;
        cam.NRI = cam_id;
        update_shape_vram_if_camera_matches(sid, cam, true);
        ensure_camera_visible_size(cam_state, nr);
        cam_state.visible[nr] = 1;
        dbg("[check_and_push]", "in view, pushed to vram");
      }
    }
  }
  void add_shape(culling_t& culling, shaper_t::ShapeID_t sid, movement_type_t movement) {
    uint32_t nr = sid.NRI;
    auto* sp = (fan::graphics::shapes::shape_t*)&sid;
    auto sc = sp->get_camera();
    dbg("[add_shape]", "sid", nr, "shape_camera", sc.NRI);
    ensure_registry_size(culling, nr);
    culling.registry.shapeid_to_movement[nr] = movement;
    dbg("[add_shape]", "sid", nr, "movement", movement == movement_static ? "static" : "dynamic");
    auto aabb = get_shape_aabb(sid);
    if (movement == movement_static) {
      culling.registry.static_shapes.push_back(sid);
      dbg("[add_shape]", "static_shapes.size", (uint32_t)culling.registry.static_shapes.size());
      add_static_shape_to_grid(culling, sid);
      check_and_push_shape_to_cameras(culling, sid, aabb);
    }
    else {
      auto center = (aabb.min + aabb.max) * 0.5f;
      auto cell = world_to_cell_clamped(center, culling.dynamic_grid.world_min, culling.dynamic_grid.cell_size, culling.dynamic_grid.grid_size);
      uint32_t id = (uint32_t)culling.dynamic_grid.objects.size();
      culling.registry.shapeid_to_dynamic[nr] = id;
      int idx = cell_index(cell, culling.dynamic_grid.grid_size);
      culling.dynamic_grid.objects.push_back({sid, aabb.min, aabb.max, idx});
      culling.dynamic_grid.cells[idx].push_back(id);
      dbg("[add_shape]", "dynamic id", id, "sid", nr, "center", center.x, center.y, "cell", cell.x, cell.y, "idx", idx);
      dbg_aabb("           dyn AABB", nr, aabb);
      check_and_push_shape_to_cameras(culling, sid, aabb);
    }
    for (auto& [cam_id, cam_state] : culling.camera_states) {
      cam_state.view_dirty = true;
      ensure_camera_visible_size(cam_state, nr);
    }
  }
  void remove_static_shape_from_grid(culling_t& culling, shaper_t::ShapeID_t sid) {
    uint32_t nr = sid.NRI;
    if (nr >= culling.registry.aabb_cache.size()) {
      dbg("[remove_static_shape_from_grid]", "sid", nr, "aabb_cache missing");
      return;
    }
    const auto& aabb = culling.registry.aabb_cache[nr];
    dbg_aabb("[remove_static_shape_from_grid] AABB", nr, aabb);
    auto min_cell = world_to_cell_clamped(aabb.min, culling.static_grid.world_min, culling.static_grid.cell_size, culling.static_grid.grid_size);
    auto max_cell = world_to_cell_clamped(aabb.max, culling.static_grid.world_min, culling.static_grid.cell_size, culling.static_grid.grid_size);
    dbg_cells("                                      span", min_cell, max_cell);
    int grid_width = culling.static_grid.grid_size.x;
    for (int y = min_cell.y; y <= max_cell.y; ++y) {
      for (int x = min_cell.x; x <= max_cell.x; ++x) {
        auto& v = culling.static_grid.cells[y * grid_width + x].shapes;
        auto it = std::find(v.begin(), v.end(), sid);
        if (it != v.end()) {
          *it = v.back();
          v.pop_back();
          dbg("  -> cell", x, y, "remove sid", nr);
        }
      }
    }
  }
  void remove_shape(culling_t& c, shaper_t::ShapeID_t sid) {
    if (sid.iic()) {
      dbg("[remove_shape]", "empty sid");
      return;
    }
    uint32_t nr = sid.NRI;
    if (nr >= c.registry.shapeid_to_movement.size()) {
      dbg("[remove_shape]", "sid", nr, "movement missing");
      return;
    }
    auto m = (movement_type_t)c.registry.shapeid_to_movement[nr];
    dbg("[remove_shape]", "sid", nr, "movement", m == movement_static ? "static" : "dynamic");
    if (m == movement_static) {
      remove_static_shape_from_grid(c, sid);
      auto& v = c.registry.static_shapes;
      auto it = std::find(v.begin(), v.end(), sid);
      if (it != v.end()) {
        *it = v.back();
        v.pop_back();
        dbg("  -> static_shapes.size", (uint32_t)v.size(), "removed sid", nr);
      }
    }
    else {
      if (nr < c.registry.shapeid_to_dynamic.size()) {
        uint32_t id = c.registry.shapeid_to_dynamic[nr];
        if (id != std::numeric_limits<uint32_t>::max() && id < c.dynamic_grid.objects.size()) {
          auto& grid = c.dynamic_grid;
          auto& obj = grid.objects[id];
          auto& cell = grid.cells[obj.cell];
          auto it = std::find(cell.begin(), cell.end(), id);
          if (it != cell.end()) {
            *it = cell.back();
            cell.pop_back();
            dbg("  -> dyn cell idx", obj.cell, "remove id", id);
          }
          if (id != grid.objects.size() - 1) {
            dbg("  -> dyn swap id", id, "<- last", (uint32_t)(grid.objects.size() - 1));
            grid.objects[id] = grid.objects.back();
            c.registry.shapeid_to_dynamic[grid.objects[id].sid.NRI] = id;
          }
          grid.objects.pop_back();
          dbg("  -> dyn objects.size", (uint32_t)grid.objects.size());
        }
        c.registry.shapeid_to_dynamic[nr] = std::numeric_limits<uint32_t>::max();
      }
    }
    for (auto& [cam_id, cam_state] : c.camera_states) {
      if (nr < cam_state.visible.size()) {
        cam_state.visible[nr] = 0;
      }
      cam_state.view_dirty = true;
      dbg("  -> cam", cam_id, "visible[nr]=0 view_dirty=true");
    }
    c.registry.shapeid_to_movement[nr] = movement_dynamic;
    dbg("  -> movement reset to dynamic for sid", nr);
  }
  movement_type_t get_movement(const culling_t& c, shaper_t::ShapeID_t sid) {
    uint32_t nr = sid.NRI;
    if (nr >= c.registry.shapeid_to_movement.size()) {
      return movement_dynamic;
    }
    return (movement_type_t)c.registry.shapeid_to_movement[nr];
  }
  void update_dynamic(culling_t& culling, shaper_t::ShapeID_t sid) {
    uint32_t nr = sid.NRI;
    if (nr >= culling.registry.shapeid_to_dynamic.size()) {
      dbg("[update_dynamic]", "sid", nr, "dynamic missing");
      return;
    }
    uint32_t id = culling.registry.shapeid_to_dynamic[nr];
    if (id == std::numeric_limits<uint32_t>::max()) {
      dbg("[update_dynamic]", "sid", nr, "not dynamic (static/unregistered)");
      return;
    }
    auto& obj = culling.dynamic_grid.objects[id];
    auto aabb = get_shape_aabb(sid);
    auto center = (aabb.min + aabb.max) * 0.5f;
    auto cell = world_to_cell_clamped(center, culling.dynamic_grid.world_min, culling.dynamic_grid.cell_size, culling.dynamic_grid.grid_size);
    int new_cell = cell_index(cell, culling.dynamic_grid.grid_size);
    if (new_cell != obj.cell) {
      auto& old = culling.dynamic_grid.cells[obj.cell];
      auto it = std::find(old.begin(), old.end(), id);
      if (it != old.end()) {
        *it = old.back();
        old.pop_back();
      }
      culling.dynamic_grid.cells[new_cell].push_back(id);
      dbg("[update_dynamic]", "move id", id, "cell", obj.cell, "->", new_cell, "center", center.x, center.y);
      obj.cell = new_cell;
    }
    obj.min = aabb.min;
    obj.max = aabb.max;
    dbg_aabb("[update_dynamic] new AABB", nr, aabb);
    mark_cameras_dirty(culling);
  }
  static inline void process_cell_range_hide(culling_t& culling, const fan::graphics::camera_t& camera_nr, int y_start, int y_end, int x_start, int x_end, bool y_visible) {
    int grid_width = culling.static_grid.grid_size.x;
    auto& cells = culling.static_grid.cells;
    auto& visible = culling.camera_states[camera_nr.NRI].visible;
    for (int y = y_start; y <= y_end; ++y) {
      if (y_visible) {
        for (int x = x_start; x <= x_end; ++x) {
          int idx = y * grid_width + x;
          for (auto sid : cells[idx].shapes) {
            uint32_t nr = sid.NRI;
            if (nr < visible.size() && visible[nr]) {
              dbg("[hide]", "cell", x, y, "sid", nr);
              update_shape_vram_if_camera_matches(sid, camera_nr, false);
              visible[nr] = 0;
            }
          }
        }
      }
      else {
        for (int x = x_start; x <= x_end; ++x) {
          int idx = y * grid_width + x;
          for (auto sid : cells[idx].shapes) {
            uint32_t nr = sid.NRI;
            if (nr < visible.size() && visible[nr]) {
              dbg("[hide-row]", "cell", x, y, "sid", nr);
              update_shape_vram_if_camera_matches(sid, camera_nr, false);
              visible[nr] = 0;
            }
          }
        }
      }
    }
  }
  void cull(culling_t& culling, shaper_t& , const fan::vec2& view_min_in, const fan::vec2& view_max_in, const fan::graphics::camera_t& camera_nr) {
    if (!culling.enabled) {
      dbg("[cull]", "disabled");
      return;
    }
    fan::vec2 view_min = view_min_in;
    fan::vec2 view_max = view_max_in;
    normalize_view_bounds(view_min, view_max);
    uint32_t cam_id = camera_nr.NRI;
    auto& cam_state = culling.camera_states[cam_id];
    size_t required_size = std::max(culling.registry.shapeid_to_movement.size(), culling.registry.aabb_cache.size());
    if (cam_state.visible.size() < required_size) {
      cam_state.visible.resize(required_size, 0);
      dbg("[cull]", "cam", cam_id, "visible resized", (uint32_t)cam_state.visible.size());
    }
    dbg("[cull]", "cam", cam_id, "view_min", view_min.x, view_min.y, "view_max", view_max.x, view_max.y, "view_dirty", cam_state.view_dirty, "cached_min", cam_state.cached_view_min.x, cam_state.cached_view_min.y, "cached_max", cam_state.cached_view_max.x, cam_state.cached_view_max.y);
    if (!cam_state.view_dirty && view_min == cam_state.cached_view_min && view_max == cam_state.cached_view_max) {
      dbg("[cull]", "cam", cam_id, "skipped (unchanged view)");
      return;
    }
    culling.current_visible = 0;
    culling.current_total = (uint32_t)culling.registry.static_shapes.size() + (uint32_t)culling.dynamic_grid.objects.size();
    dbg("[cull]", "totals static", (uint32_t)culling.registry.static_shapes.size(), "dynamic", (uint32_t)culling.dynamic_grid.objects.size(), "total", culling.current_total);
    auto min_cell = world_to_cell_clamped(view_min, culling.static_grid.world_min, culling.static_grid.cell_size, culling.static_grid.grid_size);
    auto max_cell = world_to_cell_clamped(view_max, culling.static_grid.world_min, culling.static_grid.cell_size, culling.static_grid.grid_size);
    auto prev_min_cell = world_to_cell_clamped(cam_state.cached_view_min, culling.static_grid.world_min, culling.static_grid.cell_size, culling.static_grid.grid_size);
    auto prev_max_cell = world_to_cell_clamped(cam_state.cached_view_max, culling.static_grid.world_min, culling.static_grid.cell_size, culling.static_grid.grid_size);
    dbg_cells("[cull] prev span", prev_min_cell, prev_max_cell);
    dbg_cells("[cull] curr span", min_cell, max_cell);
    int grid_width = culling.static_grid.grid_size.x;
    auto& cells = culling.static_grid.cells;
    auto& visible = cam_state.visible;
    auto& aabb_cache = culling.registry.aabb_cache;
    for (int y = prev_min_cell.y; y <= prev_max_cell.y; ++y) {
      bool y_still_visible = (y >= min_cell.y && y <= max_cell.y);
      if (y_still_visible) {
        if (min_cell.x > prev_min_cell.x) {
          process_cell_range_hide(culling, camera_nr, y, y, prev_min_cell.x, min_cell.x - 1, true);
        }
        if (max_cell.x < prev_max_cell.x) {
          process_cell_range_hide(culling, camera_nr, y, y, max_cell.x + 1, prev_max_cell.x, true);
        }
      }
      else {
        process_cell_range_hide(culling, camera_nr, y, y, prev_min_cell.x, prev_max_cell.x, false);
      }
    }
    for (int y = min_cell.y; y <= max_cell.y; ++y) {
      int row_offset = y * grid_width;
      for (int x = min_cell.x; x <= max_cell.x; ++x) {
        int idx = row_offset + x;
        auto& cell_shapes = cells[idx].shapes;
        for (auto sid : cell_shapes) {
          uint32_t nr = sid.NRI;
          if (nr < visible.size() && visible[nr]) {
            ++culling.current_visible;
            dbg("[keep]", "cell", x, y, "sid", nr);
            continue;
          }
          auto& aabb = aabb_cache[nr];
          bool overlaps = is_aabb_in_view(aabb, view_min, view_max);
          dbg("[test]", "cell", x, y, "sid", nr, "aabb.min", aabb.min.x, aabb.min.y, "aabb.max", aabb.max.x, aabb.max.y, "overlaps", overlaps ? "yes" : "no");
          if (overlaps) {
            update_shape_vram_if_camera_matches(sid, camera_nr, true);
            visible[nr] = 1;
            ++culling.current_visible;
            dbg("[show]", "sid", nr, "visible=1");
          }
        }
      }
    }
    for (auto& obj : culling.dynamic_grid.objects) {
      uint32_t nr = obj.sid.NRI;
      bool now_visible = is_aabb_in_view({obj.min, obj.max}, view_min, view_max);
      dbg("[dyn-test]", "sid", nr, "min", obj.min.x, obj.min.y, "max", obj.max.x, obj.max.y, "overlaps", now_visible ? "yes" : "no");
      if (now_visible) {
        ++culling.current_visible;
        if (!visible[nr]) {
          update_shape_vram_if_camera_matches(obj.sid, camera_nr, true);
          visible[nr] = 1;
          dbg("[dyn-show]", "sid", nr, "visible=1");
        }
      }
      else if (visible[nr]) {
        update_shape_vram_if_camera_matches(obj.sid, camera_nr, false);
        visible[nr] = 0;
        dbg("[dyn-hide]", "sid", nr, "visible=0");
      }
    }
    dbg("[cull] result", "cam", cam_id, "visible", culling.current_visible, "total", culling.current_total, "culled", (culling.current_total >= culling.current_visible ? culling.current_total - culling.current_visible : 0));
    dbg("[cull] cache-set", "cam", cam_id, "prev_min", cam_state.cached_view_min.x, cam_state.cached_view_min.y, "prev_max", cam_state.cached_view_max.x, cam_state.cached_view_max.y);
    cam_state.cached_view_min = view_min;
    cam_state.cached_view_max = view_max;
    cam_state.view_dirty = false;
    dbg("[cull] cache-now", "cam", cam_id, "min", cam_state.cached_view_min.x, cam_state.cached_view_min.y, "max", cam_state.cached_view_max.x, cam_state.cached_view_max.y, "view_dirty=false");
  }
  void cull_camera(culling_t& culling, shaper_t& shaper, const fan::graphics::camera_t& camera_nr) {
    const auto& cam = fan::graphics::ctx()->camera_get(fan::graphics::ctx(), camera_nr);
    fan::vec2 pos = fan::graphics::ctx()->camera_get_position(fan::graphics::ctx(), camera_nr);
    fan::vec2 view_min(cam.coordinates.left / cam.zoom, cam.coordinates.top / cam.zoom);
    fan::vec2 view_max(cam.coordinates.right / cam.zoom, cam.coordinates.bottom / cam.zoom);
    view_min += pos;
    view_max += pos;
    fan::vec2 scaled_padding = culling.padding  / cam.zoom;
    view_min -= scaled_padding;
    view_max += scaled_padding;
    dbg("[cull_camera]", "cam", camera_nr.NRI, "pos", pos.x, pos.y, "raw", cam.coordinates.left, cam.coordinates.top, "..", cam.coordinates.right, cam.coordinates.bottom, "padded view_min", view_min.x, view_min.y, "view_max", view_max.x, view_max.y, "padding", culling.padding.x, culling.padding.y, "zoom", cam.zoom);
    cull(culling, shaper, view_min, view_max, camera_nr);
  }
  void rebuild_static(culling_t& culling) {
    dbg("[rebuild_static]", "begin");
    for (auto& cell : culling.static_grid.cells) {
      cell.shapes.clear();
    }
    int grid_width = culling.static_grid.grid_size.x;
    for (auto sid : culling.registry.static_shapes) {
      uint32_t nr = sid.NRI;
      if (nr >= culling.registry.aabb_cache.size()) {
        culling.registry.aabb_cache.resize(nr + 1);
        dbg("  -> resize aabb_cache", (uint32_t)culling.registry.aabb_cache.size());
      }
      fan::physics::aabb_t aabb = get_shape_aabb(sid);
      culling.registry.aabb_cache[nr] = aabb;
      dbg_aabb("  -> AABB", nr, aabb);
      auto min_cell = world_to_cell_clamped(aabb.min, culling.static_grid.world_min, culling.static_grid.cell_size, culling.static_grid.grid_size);
      auto max_cell = world_to_cell_clamped(aabb.max, culling.static_grid.world_min, culling.static_grid.cell_size, culling.static_grid.grid_size);
      dbg_cells("     span", min_cell, max_cell);
      for (int y = min_cell.y; y <= max_cell.y; ++y) {
        for (int x = min_cell.x; x <= max_cell.x; ++x) {
          culling.static_grid.cells[y * grid_width + x].shapes.push_back(sid);
          dbg("     -> cell", x, y, "add sid", nr);
        }
      }
    }
    mark_cameras_dirty(culling);
    dbg("[rebuild_static]", "end");
  }
  void set_enabled(culling_t& culling, bool flag) {
    culling.enabled = flag;
    if (!culling.enabled) {
      for (auto sid : culling.registry.static_shapes) {
        auto* shape = (fan::graphics::shapes::shape_t*)&sid;
        if (!shape->get_visual_id().iic()) {
          shape->push_vram();
        }
      }
      for (auto& obj : culling.dynamic_grid.objects) {
        auto* shape = (fan::graphics::shapes::shape_t*)&obj.sid;
        if (!shape->get_visual_id().iic()) {
          shape->push_vram();
        }
      }
    }
  }

}

#endif