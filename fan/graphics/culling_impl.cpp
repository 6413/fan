module;

#include <climits>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <cstdint>

module fan.graphics.culling;

import fan.graphics.shapes;

namespace fan::graphics::culling {

  static std::vector<uint32_t> visible_stamp;
  static uint32_t frame_id = 1;

  struct block_bucket_t {
    shaper_t::bmid_t bmid;
    struct block_range_t {
      shaper_t::blid_t blid;
      std::vector<uint16_t> idx;
    };
    std::vector<block_range_t> blocks;
  };

  static std::array<std::unordered_map<uint64_t, block_bucket_t>, fan::graphics::shape_type_t::last> buckets;
  static std::array<std::vector<uint64_t>, fan::graphics::shape_type_t::last> bucket_keys;
  static std::array<size_t, fan::graphics::shape_type_t::last> last_bucket_cap{};
  static std::array<size_t, fan::graphics::shape_type_t::last> last_idx_cap{};
  static std::array<size_t, fan::graphics::shape_type_t::last> last_ranges_cap{};

  static inline uint64_t make_block_key(
    const shaper_t::bmid_t& bmid,
    const shaper_t::blid_t& blid
  ) {
    auto& mbmid = const_cast<shaper_t::bmid_t&>(bmid);

    return uint64_t(uint32_t(mbmid.gint()));
  }

  void draw_list_t::reserve(size_t count) { ranges.reserve(count); }
  void draw_list_t::clear() { ranges.clear(); }

  void result_t::clear() {
    for (auto& list : draw_lists) list.clear();
    total_visible = 0;
    total_culled = 0;
  }

  fan::physics::aabb_t get_shape_aabb(shaper_t::ShapeID_t sid) {
    fan::graphics::shapes::shape_t* shape = (fan::graphics::shapes::shape_t*)&sid;
    return fan::physics::aabb_t{
      shape->get_position() - shape->get_size(),
      shape->get_position() + shape->get_size()
    };
  }

  static inline void ensure_aabb_storage(culling_t& c, uint32_t nri) {
    if (nri >= c.registry.shapeid_to_aabb.size()) {
      c.registry.shapeid_to_aabb.resize(nri + 1);
      c.registry.aabb_valid.resize(nri + 1, 0);
    }
  }

  static inline const fan::physics::aabb_t& get_aabb_cached(culling_t& c, shaper_t::ShapeID_t sid) {
    ensure_aabb_storage(c, sid.NRI);
    if (!c.registry.aabb_valid[sid.NRI]) {
      c.registry.shapeid_to_aabb[sid.NRI] = get_shape_aabb(sid);
      c.registry.aabb_valid[sid.NRI] = 1;
    }
    return c.registry.shapeid_to_aabb[sid.NRI];
  }

  static inline void set_aabb_cached(culling_t& c, shaper_t::ShapeID_t sid, const fan::physics::aabb_t& aabb) {
    ensure_aabb_storage(c, sid.NRI);
    c.registry.shapeid_to_aabb[sid.NRI] = aabb;
    c.registry.aabb_valid[sid.NRI] = 1;
  }

  static inline void invalidate_aabb(culling_t& c, shaper_t::ShapeID_t sid) {
    if (sid.NRI < c.registry.aabb_valid.size()) {
      c.registry.aabb_valid[sid.NRI] = 0;
    }
  }

  void static_grid_init(static_grid_t& grid, const fan::vec2& world_min, const fan::vec2& cell_size, const fan::vec2i& grid_size) {
    grid.world_min = world_min;
    grid.cell_size = cell_size;
    grid.grid_size = grid_size;
    grid.cells.clear();
    grid.cells.resize(grid_size.x * grid_size.y);
  }

  void static_grid_build(culling_t& culling) {
    auto& grid = culling.static_grid;

    grid.cells.resize(grid.grid_size.x * grid.grid_size.y);
    for (auto& c : grid.cells) c.shapes.clear();

    for (auto sid : culling.registry.static_shapes) {
      const auto& aabb = get_aabb_cached(culling, sid);

      fan::vec2i min_cell = world_to_cell_clamped(aabb.min, grid.world_min, grid.cell_size, grid.grid_size);
      fan::vec2i max_cell = world_to_cell_clamped(aabb.max, grid.world_min, grid.cell_size, grid.grid_size);

      for (int y = min_cell.y; y <= max_cell.y; ++y) {
        for (int x = min_cell.x; x <= max_cell.x; ++x) {
          grid.cells[y * grid.grid_size.x + x].shapes.push_back(sid);
        }
      }
    }

    culling.view_dirty = true;
  }

  void dynamic_grid_init(dynamic_grid_t& grid, const fan::vec2& world_min, const fan::vec2& cell_size, const fan::vec2i& grid_size) {
    grid.world_min = world_min;
    grid.cell_size = cell_size;
    grid.grid_size = grid_size;
    grid.cells.clear();
    grid.cells.resize(grid_size.x * grid_size.y);
    grid.objects.clear();
    grid.shapeid_to_object_index.clear();
  }

  void dynamic_add(culling_t& culling, shaper_t::ShapeID_t sid) {
    auto& grid = culling.dynamic_grid;

    auto aabb = get_shape_aabb(sid);
    set_aabb_cached(culling, sid, aabb);

    auto center = (aabb.min + aabb.max) * 0.5f;
    fan::vec2i cell = world_to_cell_clamped(center, grid.world_min, grid.cell_size, grid.grid_size);
    int idx = cell_index(cell, grid.grid_size);

    uint32_t id = (uint32_t)grid.objects.size();
    if (sid.NRI >= grid.shapeid_to_object_index.size()) {
      grid.shapeid_to_object_index.resize(sid.NRI + 1, std::numeric_limits<uint32_t>::max());
    }
    grid.shapeid_to_object_index[sid.NRI] = id;

    grid.objects.push_back({ sid, aabb.min, aabb.max, idx });
    grid.cells[idx].push_back(id);

    culling.view_dirty = true;
  }

  void dynamic_update(culling_t& culling, uint32_t id) {
    auto& grid = culling.dynamic_grid;

    auto& o = grid.objects[id];
    auto aabb = get_shape_aabb(o.sid);
    set_aabb_cached(culling, o.sid, aabb);

    auto center = (aabb.min + aabb.max) * 0.5f;
    fan::vec2i new_cell = world_to_cell_clamped(center, grid.world_min, grid.cell_size, grid.grid_size);
    int new_index = cell_index(new_cell, grid.grid_size);

    if (new_index != o.cell) {
      auto& old = grid.cells[o.cell];
      auto it = std::find(old.begin(), old.end(), id);
    #if fan_debug >= 3
      if (it == old.end()) fan::throw_error_impl("dynamic_update: object id not found in old cell");
    #endif
      if (it != old.end()) { *it = old.back(); old.pop_back(); }
      grid.cells[new_index].push_back(id);
      o.cell = new_index;
    }

    o.min = aabb.min;
    o.max = aabb.max;

    culling.view_dirty = true;
  }

  void get_visible(
    culling_t& culling,
    const fan::vec2& view_min,
    const fan::vec2& view_max,
    std::vector<shaper_t::ShapeID_t>& out
  ) {
    out.clear();

    auto& sg = culling.static_grid;
    auto& dg = culling.dynamic_grid;

    fan::vec2i min_cell = world_to_cell_clamped(view_min, sg.world_min, sg.cell_size, sg.grid_size);
    fan::vec2i max_cell = world_to_cell_clamped(view_max, sg.world_min, sg.cell_size, sg.grid_size);

    for (int y = min_cell.y; y <= max_cell.y; ++y) {
      for (int x = min_cell.x; x <= max_cell.x; ++x) {
        int idx = cell_index({ x, y }, sg.grid_size);

        // static
        for (auto sid : sg.cells[idx].shapes) {
          if (sid.NRI >= visible_stamp.size()) visible_stamp.resize(sid.NRI + 1, 0);
          if (visible_stamp[sid.NRI] == frame_id) continue;

          const auto& aabb = get_aabb_cached(culling, sid);
          if (!aabb.intersects(view_min, view_max)) continue;

          visible_stamp[sid.NRI] = frame_id;
          out.push_back(sid);
        }

        // dynamic
        for (auto oid : dg.cells[idx]) {
          auto& o = dg.objects[oid];
          auto sid = o.sid;

          if (sid.NRI >= visible_stamp.size()) visible_stamp.resize(sid.NRI + 1, 0);
          if (visible_stamp[sid.NRI] == frame_id) continue;

          fan::physics::aabb_t aabb{ o.min, o.max };
          if (!aabb.intersects(view_min, view_max)) continue;

          visible_stamp[sid.NRI] = frame_id;
          out.push_back(sid);
        }
      }
    }
  }

  void add_shape(culling_t& culling, shaper_t&, shaper_t::ShapeID_t sid, movement_type_t mobility) {
    if (sid.NRI >= culling.registry.shapeid_to_movement.size()) {
      culling.registry.shapeid_to_movement.resize(sid.NRI + 1, movement_dynamic);
      culling.registry.shapeid_to_dynamic_index.resize(sid.NRI + 1, std::numeric_limits<uint32_t>::max());
    }

    culling.registry.shapeid_to_movement[sid.NRI] = mobility;

    set_aabb_cached(culling, sid, get_shape_aabb(sid));

    if (mobility == movement_static) {
      culling.registry.static_shapes.push_back(sid);
      culling.view_dirty = true;
    }
    else {
      uint32_t idx = (uint32_t)culling.registry.dynamic_shapes.size();
      culling.registry.shapeid_to_dynamic_index[sid.NRI] = idx;
      culling.registry.dynamic_shapes.push_back(sid);
      dynamic_add(culling, sid);
    }
  }

  void remove_shape(culling_t& culling, shaper_t::ShapeID_t sid) {
    if (sid.NRI >= culling.registry.shapeid_to_movement.size()) return;

    auto mobility = culling.registry.shapeid_to_movement[sid.NRI];

    if (mobility == movement_static) {
      auto& vec = culling.registry.static_shapes;
      auto it = std::find(vec.begin(), vec.end(), sid);
      if (it != vec.end()) { *it = vec.back(); vec.pop_back(); }
    }
    else {
      uint32_t idx = culling.registry.shapeid_to_dynamic_index[sid.NRI];
      if (idx != std::numeric_limits<uint32_t>::max() && idx < culling.registry.dynamic_shapes.size()) {
        auto& grid = culling.dynamic_grid;

        if (sid.NRI < grid.shapeid_to_object_index.size()) {
          uint32_t oid = grid.shapeid_to_object_index[sid.NRI];
          if (oid != std::numeric_limits<uint32_t>::max() && oid < grid.objects.size()) {
            auto& obj = grid.objects[oid];
            auto& cell = grid.cells[obj.cell];
            auto it = std::find(cell.begin(), cell.end(), oid);
            if (it != cell.end()) { *it = cell.back(); cell.pop_back(); }

            if (oid != grid.objects.size() - 1) {
              grid.objects[oid] = grid.objects.back();
              auto moved_sid = grid.objects[oid].sid;
              if (moved_sid.NRI < grid.shapeid_to_object_index.size()) {
                grid.shapeid_to_object_index[moved_sid.NRI] = oid;
              }
            }
            grid.objects.pop_back();
            grid.shapeid_to_object_index[sid.NRI] = std::numeric_limits<uint32_t>::max();
          }
        }

        auto& vec = culling.registry.dynamic_shapes;
        if (idx < vec.size()) {
          vec[idx] = vec.back();
          if (!vec.empty() && idx < vec.size()) {
            culling.registry.shapeid_to_dynamic_index[vec[idx].NRI] = idx;
          }
          vec.pop_back();
        }
      }
    }

    culling.registry.shapeid_to_movement[sid.NRI] = movement_dynamic;
    culling.registry.shapeid_to_dynamic_index[sid.NRI] = std::numeric_limits<uint32_t>::max();
    invalidate_aabb(culling, sid);

    culling.view_dirty = true;
  }

  void set_movement_type(culling_t& culling, shaper_t& shaper, shaper_t::ShapeID_t sid, movement_type_t new_mobility) {
    if (sid.NRI >= culling.registry.shapeid_to_movement.size()) {
      add_shape(culling, shaper, sid, new_mobility);
      return;
    }
    auto old_mobility = culling.registry.shapeid_to_movement[sid.NRI];
    if (old_mobility == new_mobility) return;

    remove_shape(culling, sid);
    add_shape(culling, shaper, sid, new_mobility);

    culling.view_dirty = true;
  }

  void update_dynamic(culling_t& culling, shaper_t::ShapeID_t sid) {
    if (sid.NRI >= culling.registry.shapeid_to_dynamic_index.size()) return;

    if (sid.NRI < culling.dynamic_grid.shapeid_to_object_index.size()) {
      uint32_t oid = culling.dynamic_grid.shapeid_to_object_index[sid.NRI];
      if (oid != std::numeric_limits<uint32_t>::max() && oid < culling.dynamic_grid.objects.size()) {
        dynamic_update(culling, oid);
      }
    }
  }

  void cull(culling_t& culling, shaper_t& shaper, const fan::vec2& camera_min, const fan::vec2& camera_max) {
    if (!culling.enabled) {
      culling.current_result.clear();
      culling.cached_view_min = camera_min;
      culling.cached_view_max = camera_max;
      culling.view_dirty = false;
      return;
    }

    if (!culling.view_dirty &&
      camera_min == culling.cached_view_min &&
      camera_max == culling.cached_view_max) {
      return;
    }

    culling.current_result.clear();

    ++frame_id;
    if (frame_id == 0) {
      std::fill(visible_stamp.begin(), visible_stamp.end(), 0);
      frame_id = 1;
    }

    for (uint32_t sti = 0; sti < (uint32_t)fan::graphics::shape_type_t::last; ++sti) {
      auto& m = buckets[sti];
      if (last_bucket_cap[sti] != 0) m.reserve(last_bucket_cap[sti]);
      m.clear();

      auto& keys = bucket_keys[sti];
      keys.clear();
      if (last_bucket_cap[sti] != 0) keys.reserve(last_bucket_cap[sti]);
    }

    std::vector<shaper_t::ShapeID_t> visible_shapes;
    get_visible(culling, camera_min, camera_max, visible_shapes);
    culling.current_result.total_visible = (uint32_t)visible_shapes.size();

    for (auto sid : visible_shapes) {
      auto& sdata = shaper.ShapeList[sid];
      auto sti = sdata.sti;
      auto bmid = sdata.bmid;
      auto blid = sdata.blid;

      uint64_t key = make_block_key(bmid, blid);
      auto& map = buckets[sti];

      auto it = map.find(key);
      if (it == map.end()) {
        block_bucket_t b;
        b.bmid = bmid;
        block_bucket_t::block_range_t br;
        br.blid = blid;
        br.idx.reserve(last_idx_cap[sti] ? std::min<size_t>(last_idx_cap[sti], 256) : 32);
        br.idx.push_back((uint16_t)sdata.ElementIndex);
        b.blocks.push_back(std::move(br));
        map.emplace(key, std::move(b));
        bucket_keys[sti].push_back(key);
      }
      else {
        auto& bucket = it->second;
        bool found_block = false;
        for (auto& br : bucket.blocks) {
          if (br.blid.gint() == blid.gint()) {
            br.idx.push_back((uint16_t)sdata.ElementIndex);
            found_block = true;
            break;
          }
        }
        if (!found_block) {
          block_bucket_t::block_range_t br;
          br.blid = blid;
          br.idx.reserve(32);
          br.idx.push_back((uint16_t)sdata.ElementIndex);
          bucket.blocks.push_back(std::move(br));
        }
      }
    }

    for (uint32_t sti = 0; sti < (uint32_t)fan::graphics::shape_type_t::last; ++sti) {
      auto& keys = bucket_keys[sti];
      if (keys.empty()) continue;

      std::sort(keys.begin(), keys.end());

      auto& out = culling.current_result.draw_lists[sti].ranges;
      out.clear();
      if (last_ranges_cap[sti] != 0) out.reserve(last_ranges_cap[sti]);

      auto& map = buckets[sti];
      auto& st = shaper.ShapeTypes[sti];
      uint32_t max_per_block = st.MaxElementPerBlock();

      for (uint64_t k : keys) {
        auto mit = map.find(k);
        if (mit == map.end()) continue;

        auto& bucket = mit->second;

        std::sort(bucket.blocks.begin(), bucket.blocks.end(), [](auto& a, auto& b) {
          return a.blid.gint() < b.blid.gint();
        });

        struct global_element_t {
          uint32_t global_idx;
          shaper_t::blid_t blid;
          uint16_t local_idx;
        };
        std::vector<global_element_t> all_elements;

        for (auto& br : bucket.blocks) {
          auto& idx = br.idx;
          if (idx.empty()) continue;

          uint32_t block_base = br.blid.gint() * max_per_block;
          for (uint16_t local : idx) {
            all_elements.push_back({ block_base + local, br.blid, local });
          }

          last_idx_cap[sti] = std::max(last_idx_cap[sti], idx.size());
        }

        if (all_elements.empty()) continue;

        std::sort(all_elements.begin(), all_elements.end(), [](const auto& a, const auto& b) {
          return a.global_idx < b.global_idx;
        });

        uint32_t range_start_global = all_elements[0].global_idx;
        shaper_t::blid_t range_blid = all_elements[0].blid;
        uint16_t range_first_local = all_elements[0].local_idx;
        uint32_t range_count = 1;

        for (size_t i = 1; i < all_elements.size(); ++i) {
          auto& elem = all_elements[i];

          if (elem.global_idx == range_start_global + range_count &&
            elem.blid.gint() == range_blid.gint()) {
            range_count++;
          }
          else {
            out.push_back({ bucket.bmid, range_blid, range_first_local, range_count });

            range_start_global = elem.global_idx;
            range_blid = elem.blid;
            range_first_local = elem.local_idx;
            range_count = 1;
          }
        }

        out.push_back({ bucket.bmid, range_blid, range_first_local, range_count });
      }

      last_bucket_cap[sti] = std::max(last_bucket_cap[sti], keys.size());
      last_ranges_cap[sti] = std::max(last_ranges_cap[sti], out.size());
    }

    culling.cached_view_min = camera_min;
    culling.cached_view_max = camera_max;
    culling.view_dirty = false;
  }

  void cull_camera(
    culling_t& culling,
    shaper_t& shaper,
    const fan::graphics::camera_t& camera_nr,
    const fan::vec2& /*viewport_size*/
  ) {
    const fan::graphics::context_camera_t& camera = fan::graphics::ctx()->camera_get(fan::graphics::ctx(), camera_nr);
    fan::vec2 cam_pos = fan::graphics::ctx()->camera_get_position(fan::graphics::ctx(), camera_nr);

    fan::vec2 view_min(camera.coordinates.left, camera.coordinates.top);
    fan::vec2 view_max(camera.coordinates.right, camera.coordinates.bottom);

    view_min += cam_pos;
    view_max += cam_pos;

    // Padding convention as in your working version
    view_min -= culling.padding;
    view_max += culling.padding;

    cull(culling, shaper, view_min, view_max);
  }

  void rebuild_static(culling_t& culling) {
    static_grid_build(culling);
  }

} // namespace fan::graphics::culling
