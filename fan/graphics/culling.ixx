module;
#include <vector>
#include <unordered_map>
#include <array>
#include <limits>
#include <cstdint>

export module fan.graphics.culling;

import fan.graphics.common_context;
import fan.graphics.shapes.types;
import fan.types.vector;
import fan.physics.types;

export namespace fan::graphics::culling {

  enum movement_type_t : uint8_t {
    movement_static,
    movement_dynamic,
    movement_sleeping
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
    std::vector<uint32_t> shapeid_to_object;
  };

  struct culling_registry_t {
    std::vector<shaper_t::ShapeID_t> static_shapes;
    std::vector<uint32_t> shapeid_to_dynamic;
    std::vector<movement_type_t> shapeid_to_movement;
    std::vector<fan::physics::aabb_t> aabb_cache;
  };

  struct per_camera_state_t {
    std::vector<uint8_t> visible;
    fan::vec2 cached_view_min;
    fan::vec2 cached_view_max;
    bool view_dirty = true;
  };

  struct culling_t {
    std::unordered_map<uint32_t, per_camera_state_t> camera_states;
    fan::vec2 padding = 0;
    bool enabled = false;
    uint32_t current_visible = 0;
    uint32_t current_total = 0;

    static_grid_t static_grid;
    dynamic_grid_t dynamic_grid;
    culling_registry_t registry;
  };

  constexpr int cell_index(const fan::vec2i& c, const fan::vec2i& grid_size);
  constexpr fan::vec2i world_to_cell_clamped(const fan::vec2& p, const fan::vec2& world_min, const fan::vec2& cell_size, const fan::vec2i& grid_size);
  fan::physics::aabb_t get_shape_aabb(shaper_t::ShapeID_t sid);
  void static_grid_init(static_grid_t& grid, const fan::vec2& world_min, const fan::vec2& cell_size, const fan::vec2i& grid_size);
  void dynamic_grid_init(dynamic_grid_t& grid, const fan::vec2& world_min, const fan::vec2& cell_size, const fan::vec2i& grid_size);
  void add_shape(culling_t& culling, shaper_t::ShapeID_t sid, movement_type_t movement);
  void remove_shape(culling_t& culling, shaper_t::ShapeID_t sid);
  movement_type_t get_movement(const culling_t& c, shaper_t::ShapeID_t sid);
  void update_dynamic(culling_t& culling, shaper_t::ShapeID_t sid);
  void cull(culling_t& culling, shaper_t& shaper, const fan::vec2& view_min, const fan::vec2& view_max, const fan::graphics::camera_t& camera_nr);
  void cull_camera(culling_t& culling, shaper_t& shaper, const fan::graphics::camera_t& camera_nr);
  void rebuild_static(culling_t& culling);
  void set_enabled(culling_t& culling, bool flag);
}