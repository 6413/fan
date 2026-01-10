module;

#if defined(FAN_2D)
  #include <cstdint>
  #include <algorithm>
  #include <string>
  #include <unordered_map>
#endif

module fan.graphics.culling;

#if defined(FAN_2D)

import fan.graphics.shapes;
import fan.print;

namespace fan::graphics::culling {

  template <typename... Args>
  static inline void dbg(const char* tag, Args&&... args) {
    //std::string t(tag);
    /* if (t.contains("show") || t.contains("[push_shaper]") || t.contains("keep") || t.contains("test") || t.contains("span") || t.contains("[cull]") || t == "[remove_static_shape_from_grid] AABB" || t == "[AABB]" || t == "[add_static_shape_to_grid]" || t == "[remove_static_shape_from_grid]" || t == "[update_dynamic]" || t == "[zremove_shape]" || t == "[add_static_shape_to_grid] AABB" || t == "[dyn-test]" || t.contains("->") || t == "[update_dynamic] new AABB") {
    return;
    }
    */
    //if (t.contains("[AABB]")) {
    //  fan::print_throttled(tag, std::forward<Args>(args)...);
    //}
    //if (t.contains("[cull_camera]")) {
    //  fan::print_throttled(tag, std::forward<Args>(args)...);
    //}
    //fan::print_throttled(tag, std::forward<Args>(args)...);
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
      else if constexpr (requires { properties.position; properties.start_size; }) {
        position = properties.position;
        size = properties.start_size;
        has_bounds = true;
        matched = "start_size";
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

  void static_grid_init(fan::graphics::spatial::static_grid_t<shaper_t::ShapeID_t>& grid, const fan::vec2& world_min, const fan::vec2& cell_size, const fan::vec2i& grid_size) {
    fan::graphics::spatial::static_grid_init(grid, world_min, cell_size, grid_size);
    dbg("[static_grid_init]", "world_min", world_min.x, world_min.y, "cell_size", cell_size.x, cell_size.y, "grid_size", grid_size.x, grid_size.y, "cells", (uint32_t)grid.cells.size());
  }

  void dynamic_grid_init(fan::graphics::spatial::dynamic_grid_t<shaper_t::ShapeID_t>& grid, const fan::vec2& world_min, const fan::vec2& cell_size, const fan::vec2i& grid_size) {
    fan::graphics::spatial::dynamic_grid_init(grid, world_min, cell_size, grid_size);
    dbg("[dynamic_grid_init]", "world_min", world_min.x, world_min.y, "cell_size", cell_size.x, cell_size.y, "grid_size", grid_size.x, grid_size.y, "cells", (uint32_t)grid.cells.size());
  }

  static inline void mark_cameras_dirty(culling_t& culling) {
    for (auto& [cam_id, cam_state] : culling.camera_states) {
      cam_state.view_dirty = true;
    }
  }

  static inline void ensure_camera_visible_size(per_camera_state_t& cam_state, uint32_t nr) {
    if (cam_state.visible.find(nr) == cam_state.visible.end()) {
      cam_state.visible[nr] = 0;
    }
  }

  void update_shape_vram_if_camera_matches(shaper_t::ShapeID_t sid, const fan::graphics::camera_t& culling_camera, bool push) {
    auto* shape_ptr = (fan::graphics::shapes::shape_t*)&sid;
    auto sc = shape_ptr->get_camera();
    if (sc == culling_camera) {
      dbg(push ? "[push_shaper]" : "[erase_shaper]", "sid", sid.NRI, "cam", sc.NRI);
      if (push) {
        shape_ptr->push_shaper();
      }
      else {
        shape_ptr->erase_shaper();
      }
    }
    else {
      dbg("[camera_mismatch]", "sid", sid.NRI, "shape_cam", sc.NRI, "cull_cam", culling_camera.NRI);
    }
  }

  static inline void check_and_push_shape_to_cameras(culling_t& culling, shaper_t::ShapeID_t sid, const fan::physics::aabb_t& aabb) {
    uint32_t nr = sid.NRI;
    for (auto& [cam_id, cam_state] : culling.camera_states) {
      if (cam_state.cached_view_min != cam_state.cached_view_max && fan::graphics::spatial::is_aabb_in_view(aabb, cam_state.cached_view_min, cam_state.cached_view_max)) {
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

    auto aabb = get_shape_aabb(sid);
    
    fan::graphics::spatial::add_object(
      culling.registry,
      culling.static_grid,
      culling.dynamic_grid,
      sid,
      aabb,
      (fan::graphics::spatial::movement_type_t)movement
    );
    
    dbg("[add_shape]", "sid", nr, "movement", movement == movement_static ? "static" : "dynamic");
    
    if (movement == movement_static) {
      dbg("[add_shape]", "static_shapes.size", (uint32_t)culling.registry.static_objects.size());
      dbg_aabb("[add_static_shape_to_grid] AABB", nr, aabb);
      auto min_cell = fan::graphics::spatial::world_to_cell_clamped(aabb.min, culling.static_grid.world_min, culling.static_grid.cell_size, culling.static_grid.grid_size);
      auto max_cell = fan::graphics::spatial::world_to_cell_clamped(aabb.max, culling.static_grid.world_min, culling.static_grid.cell_size, culling.static_grid.grid_size);
      dbg_cells("                               span", min_cell, max_cell);
      if (!culling.enabled) {
        sp->push_shaper();
      }
      else {
        check_and_push_shape_to_cameras(culling, sid, aabb);
      }
    }
    else {
      auto center = (aabb.min + aabb.max) * 0.5f;
      auto cell = fan::graphics::spatial::world_to_cell_clamped(center, culling.dynamic_grid.world_min, culling.dynamic_grid.cell_size, culling.dynamic_grid.grid_size);
      uint32_t id = (uint32_t)culling.dynamic_grid.objects.size() - 1;
      int idx = fan::graphics::spatial::cell_index(cell, culling.dynamic_grid.grid_size);
      dbg("[add_shape]", "dynamic id", id, "sid", nr, "center", center.x, center.y, "cell", cell.x, cell.y, "idx", idx);
      dbg_aabb("           dyn AABB", nr, aabb);
      if (!culling.enabled) {
        sp->push_shaper();
      }
      else {
        check_and_push_shape_to_cameras(culling, sid, aabb);
      }
    }

    for (auto& [cam_id, cam_state] : culling.camera_states) {
      cam_state.view_dirty = true;
      ensure_camera_visible_size(cam_state, nr);
    }
  }

  void remove_shape(culling_t& culling, shaper_t::ShapeID_t sid) {
    if (sid.iic()) {
      dbg("[remove_shape]", "empty sid");
      return;
    }

    uint32_t nr = sid.NRI;
    auto it = culling.registry.id_to_movement.find(sid);
    if (it == culling.registry.id_to_movement.end()) {
      dbg("[remove_shape]", "sid", nr, "movement missing");
      return;
    }

    auto m = (movement_type_t)it->second;
    dbg("[remove_shape]", "sid", nr, "movement", m == movement_static ? "static" : "dynamic");

    if (m == movement_static) {
      auto aabb_it = culling.registry.aabb_cache.find(sid);
      if (aabb_it != culling.registry.aabb_cache.end()) {
        dbg_aabb("[remove_static_shape_from_grid] AABB", nr, aabb_it->second);
        auto min_cell = fan::graphics::spatial::world_to_cell_clamped(aabb_it->second.min, culling.static_grid.world_min, culling.static_grid.cell_size, culling.static_grid.grid_size);
        auto max_cell = fan::graphics::spatial::world_to_cell_clamped(aabb_it->second.max, culling.static_grid.world_min, culling.static_grid.cell_size, culling.static_grid.grid_size);
        dbg_cells("                                      span", min_cell, max_cell);
      }
    }
    else {
      auto dyn_it = culling.registry.id_to_dynamic.find(sid);
      if (dyn_it != culling.registry.id_to_dynamic.end()) {
        uint32_t id = dyn_it->second;
        dbg("  -> dyn remove id", id);
      }
    }

    fan::graphics::spatial::remove_object(
      culling.registry,
      culling.static_grid,
      culling.dynamic_grid,
      sid
    );

    for (auto& [cam_id, cam_state] : culling.camera_states) {
      cam_state.visible.erase(nr);
      cam_state.view_dirty = true;
      dbg("  -> cam", cam_id, "visible[nr]=0 view_dirty=true");
    }
  }

  movement_type_t get_movement(const culling_t& c, shaper_t::ShapeID_t sid) {
    auto it = c.registry.id_to_movement.find(sid);
    if (it == c.registry.id_to_movement.end()) {
      return movement_dynamic;
    }
    return (movement_type_t)it->second;
  }

  void update_dynamic(culling_t& culling, shaper_t::ShapeID_t sid) {
    uint32_t nr = sid.NRI;
    auto dyn_it = culling.registry.id_to_dynamic.find(sid);
    if (dyn_it == culling.registry.id_to_dynamic.end()) {
      dbg("[update_dynamic]", "sid", nr, "dynamic missing");
      return;
    }

    auto aabb = get_shape_aabb(sid);
    dbg_aabb("[update_dynamic] new AABB", nr, aabb);
    
    fan::graphics::spatial::update_dynamic_object(
      culling.registry,
      culling.dynamic_grid,
      sid,
      aabb
    );

    mark_cameras_dirty(culling);
  }

  void cull(culling_t& culling, shaper_t& shaper, const fan::vec2& view_min_in, const fan::vec2& view_max_in, const fan::graphics::camera_t& camera_nr) {
    if (!culling.enabled) {
      dbg("[cull]", "disabled");
      return;
    }

    fan::vec2 view_min = view_min_in;
    fan::vec2 view_max = view_max_in;
    normalize_view_bounds(view_min, view_max);
    uint32_t cam_id = camera_nr.NRI;
    auto& cam_state = culling.camera_states[cam_id];

    dbg("[cull]", "cam", cam_id, "view_min", view_min.x, view_min.y, "view_max", view_max.x, view_max.y, "view_dirty", cam_state.view_dirty, "cached_min", cam_state.cached_view_min.x, cam_state.cached_view_min.y, "cached_max", cam_state.cached_view_max.x, cam_state.cached_view_max.y);

    if (!cam_state.view_dirty && view_min == cam_state.cached_view_min && view_max == cam_state.cached_view_max) {
      dbg("[cull]", "cam", cam_id, "skipped (unchanged view)");
      return;
    }

    culling.current_visible = 0;
    culling.current_total = (uint32_t)culling.registry.static_objects.size() + (uint32_t)culling.dynamic_grid.objects.size();
    dbg("[cull]", "totals static", (uint32_t)culling.registry.static_objects.size(), "dynamic", (uint32_t)culling.dynamic_grid.objects.size(), "total", culling.current_total);

    auto min_cell = fan::graphics::spatial::world_to_cell_clamped(view_min, culling.static_grid.world_min, culling.static_grid.cell_size, culling.static_grid.grid_size);
    auto max_cell = fan::graphics::spatial::world_to_cell_clamped(view_max, culling.static_grid.world_min, culling.static_grid.cell_size, culling.static_grid.grid_size);
    dbg_cells("[cull] curr span", min_cell, max_cell);

    std::unordered_map<uint32_t, uint8_t> new_visible;

    fan::graphics::spatial::query_area(
      culling.static_grid,
      culling.dynamic_grid,
      view_min,
      view_max,
      [&](shaper_t::ShapeID_t sid) {
        uint32_t nr = sid.NRI;
        auto aabb_it = culling.registry.aabb_cache.find(sid);
        if (aabb_it == culling.registry.aabb_cache.end()) {
          return;
        }

        bool overlaps = fan::graphics::spatial::is_aabb_in_view(aabb_it->second, view_min, view_max);
        dbg("[test]", "sid", nr, "aabb.min", aabb_it->second.min.x, aabb_it->second.min.y, "aabb.max", aabb_it->second.max.x, aabb_it->second.max.y, "overlaps", overlaps ? "yes" : "no");

        if (overlaps) {
          bool was_visible = cam_state.visible[nr];
          new_visible[nr] = 1;
          ++culling.current_visible;

          if (!was_visible) {
            update_shape_vram_if_camera_matches(sid, camera_nr, true);
            dbg("[show]", "sid", nr, "visible=1");
          }
          else {
            dbg("[keep]", "sid", nr);
          }
        }
      }
    );

    for (auto& [nr, visible] : cam_state.visible) {
      if (visible && new_visible.find(nr) == new_visible.end()) {
        shaper_t::ShapeID_t sid;
        sid.NRI = nr;
        update_shape_vram_if_camera_matches(sid, camera_nr, false);
        dbg("[hide]", "sid", nr, "visible=0");
      }
    }

    dbg("[cull] result", "cam", cam_id, "visible", culling.current_visible, "total", culling.current_total, "culled", (culling.current_total >= culling.current_visible ? culling.current_total - culling.current_visible : 0));
    dbg("[cull] cache-set", "cam", cam_id, "prev_min", cam_state.cached_view_min.x, cam_state.cached_view_min.y, "prev_max", cam_state.cached_view_max.x, cam_state.cached_view_max.y);

    cam_state.visible = std::move(new_visible);
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
    fan::vec2 scaled_padding = culling.padding / cam.zoom;
    view_min -= scaled_padding;
    view_max += scaled_padding;
    dbg("[cull_camera]", "cam", camera_nr.NRI, "pos", pos.x, pos.y, "raw", cam.coordinates.left, cam.coordinates.top, "..", cam.coordinates.right, cam.coordinates.bottom, "padded view_min", view_min.x, view_min.y, "view_max", view_max.x, view_max.y, "padding", culling.padding.x, culling.padding.y, "zoom", cam.zoom);
    cull(culling, shaper, view_min, view_max, camera_nr);
  }

  void rebuild_static(culling_t& culling) {
    dbg("[rebuild_static]", "begin");

    culling.static_grid.cells.clear();
    culling.static_grid.cells.resize(
      culling.static_grid.grid_size.x * culling.static_grid.grid_size.y
    );

    for (auto sid : culling.registry.static_objects) {
      auto aabb_it = culling.registry.aabb_cache.find(sid);
      if (aabb_it == culling.registry.aabb_cache.end()) {
        continue;
      }

      const auto& aabb = aabb_it->second;
      uint32_t nr = sid.NRI;

      dbg_aabb("  -> AABB", nr, aabb);

      auto min_cell = fan::graphics::spatial::world_to_cell_clamped(
        aabb.min,
        culling.static_grid.world_min,
        culling.static_grid.cell_size,
        culling.static_grid.grid_size
      );

      auto max_cell = fan::graphics::spatial::world_to_cell_clamped(
        aabb.max,
        culling.static_grid.world_min,
        culling.static_grid.cell_size,
        culling.static_grid.grid_size
      );

      dbg_cells("     span", min_cell, max_cell);

      int grid_width = culling.static_grid.grid_size.x;

      auto& list = culling.registry.static_object_cells[sid];
      list.clear();

      for (int y = min_cell.y; y <= max_cell.y; ++y) {
        for (int x = min_cell.x; x <= max_cell.x; ++x) {
          int idx = y * grid_width + x;
          culling.static_grid.cells[idx].objects.push_back(sid);
          list.push_back(idx);
        }
      }
    }

    mark_cameras_dirty(culling);
    dbg("[rebuild_static]", "end");
  }

  void set_enabled(culling_t& culling, bool flag) {
    if (culling.enabled == flag) {
      return;
    }

    if (!flag) {
      for (auto sid : culling.registry.static_objects) {
        auto* shape = (fan::graphics::shapes::shape_t*)&sid;
        if (!shape->get_visual_id().iic()) {
          shape->push_shaper();
        }
      }
      for (auto& obj : culling.dynamic_grid.objects) {
        auto* shape = (fan::graphics::shapes::shape_t*)&obj.id;
        if (!shape->get_visual_id().iic()) {
          shape->push_shaper();
        }
      }
    }

    culling.enabled = flag;

    if (flag) {
      rebuild_static(culling);
    }
  }

  void reset(culling_t& culling) {
    for (auto& [cam_id, cam_state] : culling.camera_states) {
      cam_state.visible.clear();
      cam_state.visible.rehash(0);
      cam_state.cached_view_min = fan::vec2(0);
      cam_state.cached_view_max = fan::vec2(0);
      cam_state.view_dirty = true;
    }

    culling.current_visible = 0;
    culling.current_total = 0;

    fan::graphics::spatial::reset(
      culling.static_grid,
      culling.dynamic_grid,
      culling.registry
    );

    fan::graphics::spatial::static_grid_init(
      culling.static_grid,
      culling.static_grid.world_min,
      culling.static_grid.cell_size,
      culling.static_grid.grid_size
    );

    fan::graphics::spatial::dynamic_grid_init(
      culling.dynamic_grid,
      culling.dynamic_grid.world_min,
      culling.dynamic_grid.cell_size,
      culling.dynamic_grid.grid_size
    );
  }
}

#endif