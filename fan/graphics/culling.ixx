module;

#include <fan/utility.h>

#include <vector>
#include <algorithm>
#include <array>
#include <limits>

export module fan.graphics.culling;

import fan.graphics.shapes.types;
import fan.types.vector;
import fan.physics.types;

export namespace fan::graphics::culling {

  enum movement_type_t : uint8_t {
    movement_static,
    movement_dynamic,
    movement_sleeping
  };

  struct culling_state_t {
    std::vector<shaper_t::ShapeID_t> static_shapes;
    std::vector<shaper_t::ShapeID_t> dynamic_shapes;
    std::vector<uint32_t> shapeid_to_dynamic_index;
    std::vector<movement_type_t> shapeid_to_movement;

    std::vector<fan::physics::aabb_t> shapeid_to_aabb;
    std::vector<uint8_t> aabb_valid;
  };

  struct static_cell_t {
    std::vector<shaper_t::ShapeID_t> shapes;
  };

  struct static_grid_t {
    fan::vec2 world_min;
    fan::vec2 cell_size;
    fan::vec2i grid_size;
    std::vector<static_cell_t> cells;
  };

  struct dynamic_object_t {
    shaper_t::ShapeID_t sid;
    fan::vec2 min;
    fan::vec2 max;
    int cell;
  };

  struct dynamic_grid_t {
    fan::vec2 world_min;
    fan::vec2 cell_size;
    fan::vec2i grid_size;
    std::vector<std::vector<uint32_t>> cells;
    std::vector<dynamic_object_t> objects;
    std::vector<uint32_t> shapeid_to_object_index;
  };

  struct draw_range_t {
    shaper_t::bmid_t bmid;
    shaper_t::blid_t block_id;
    uint32_t first_instance;
    uint32_t count;
  };

  struct draw_list_t {
    void reserve(size_t count);
    void clear();
    std::vector<draw_range_t> ranges;
  };

  struct result_t {
    void clear();
    std::array<draw_list_t, fan::graphics::shape_type_t::last> draw_lists;
    uint32_t total_visible = 0;
    uint32_t total_culled = 0;
  };

  struct culling_t {
    culling_state_t registry;
    static_grid_t static_grid;
    dynamic_grid_t dynamic_grid;
    result_t current_result;

    fan::vec2 cached_view_min;
    fan::vec2 cached_view_max;
    fan::vec2 padding = 0;

    bool enabled = false;
    bool view_dirty = true;
  };

  constexpr int cell_index(const fan::vec2i& c, const fan::vec2i& grid_size) {
    return c.y * grid_size.x + c.x;
  }

  constexpr fan::vec2i world_to_cell_clamped(
    const fan::vec2& p,
    const fan::vec2& world_min,
    const fan::vec2& cell_size,
    const fan::vec2i& grid_size
  ) {
    fan::vec2i c = fan::vec2i((p - world_min) / cell_size);
    c.x = std::max(0, std::min(grid_size.x - 1, c.x));
    c.y = std::max(0, std::min(grid_size.y - 1, c.y));
    return c;
  }

  fan::physics::aabb_t get_shape_aabb(shaper_t::ShapeID_t sid);

  void static_grid_init(static_grid_t& grid, const fan::vec2& world_min, const fan::vec2& cell_size, const fan::vec2i& grid_size);
  void static_grid_build(culling_t& vs);

  void dynamic_grid_init(dynamic_grid_t& grid, const fan::vec2& world_min, const fan::vec2& cell_size, const fan::vec2i& grid_size);
  void dynamic_add(culling_t& vs, shaper_t::ShapeID_t sid);
  void dynamic_update(culling_t& vs, uint32_t id);

  void get_visible(culling_t& vs, const fan::vec2& view_min, const fan::vec2& view_max, std::vector<shaper_t::ShapeID_t>& out);

  void add_shape(culling_t& vs, shaper_t& shaper, shaper_t::ShapeID_t sid, movement_type_t mobility = movement_dynamic);
  void remove_shape(culling_t& vs, shaper_t::ShapeID_t sid);
  void set_movement_type(culling_t& vs, shaper_t& shaper, shaper_t::ShapeID_t sid, movement_type_t new_mobility);
  void update_dynamic(culling_t& vs, shaper_t::ShapeID_t sid);

  void cull(culling_t& vs, shaper_t& shaper, const fan::vec2& camera_min, const fan::vec2& camera_max);
  void cull_camera(culling_t& vs, shaper_t& shaper, const fan::graphics::camera_t& camera_nr, const fan::vec2& viewport_size);

  void rebuild_static(culling_t& vs);
}