module;

#if defined(FAN_2D)
  #include <vector>
  #include <unordered_map>
  #include <limits>
  #include <cstdint>
#endif

export module fan.graphics.culling;

#if defined(FAN_2D)

import fan.graphics.common_context;
import fan.graphics.shapes.types;
import fan.graphics.spatial;
import fan.types.vector;
import fan.physics.types;

export namespace fan::graphics::culling {

  enum movement_type_t : uint8_t {
    movement_static = fan::graphics::spatial::movement_static,
    movement_dynamic = fan::graphics::spatial::movement_dynamic,
    movement_sleeping
  };

  struct per_camera_state_t {
    std::unordered_map<uint32_t, uint8_t> visible;
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

    fan::graphics::spatial::static_grid_t<shaper_t::ShapeID_t> static_grid;
    fan::graphics::spatial::dynamic_grid_t<shaper_t::ShapeID_t> dynamic_grid;
    fan::graphics::spatial::registry_t<shaper_t::ShapeID_t> registry;
  };

  fan::physics::aabb_t get_shape_aabb(shaper_t::ShapeID_t sid);
  void update_shape_vram_if_camera_matches(shaper_t::ShapeID_t sid, const fan::graphics::camera_t& culling_camera, bool push);

  void static_grid_init(fan::graphics::spatial::static_grid_t<shaper_t::ShapeID_t>& grid, const fan::vec2& world_min, const fan::vec2& cell_size, const fan::vec2i& grid_size);
  void dynamic_grid_init(fan::graphics::spatial::dynamic_grid_t<shaper_t::ShapeID_t>& grid, const fan::vec2& world_min, const fan::vec2& cell_size, const fan::vec2i& grid_size);

  void add_shape(culling_t& culling, shaper_t::ShapeID_t sid, movement_type_t movement);
  void remove_shape(culling_t& culling, shaper_t::ShapeID_t sid);
  movement_type_t get_movement(const culling_t& c, shaper_t::ShapeID_t sid);
  void update_dynamic(culling_t& culling, shaper_t::ShapeID_t sid);
  void cull(culling_t& culling, shaper_t& shaper, const fan::vec2& view_min, const fan::vec2& view_max, const fan::graphics::camera_t& camera_nr);
  void cull_camera(culling_t& culling, shaper_t& shaper, const fan::graphics::camera_t& camera_nr);
  void rebuild_static(culling_t& culling);
  void set_enabled(culling_t& culling, bool flag);
}

#endif