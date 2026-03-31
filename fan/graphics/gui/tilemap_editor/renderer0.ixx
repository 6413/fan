module;

#if defined(FAN_2D)

#include <fan/utility.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <functional>
#include <cstdlib>
#include <variant>
#include <source_location>
#include <algorithm>

#endif

export module fan.graphics.gui.tilemap_editor.renderer;

#if defined(FAN_2D)

#if defined(FAN_PHYSICS_2D)

import fan.graphics.gui.tilemap_editor.loader;

import fan.types;
import fan.types.color;
import fan.types.vector;
import fan.utility;
import fan.print;
import fan.graphics.common_context;
import fan.graphics.shapes;
import fan.graphics.physics_shapes;
import fan.graphics;
import fan.physics.b2_integration;

export struct tilemap_renderer_t : tilemap_loader_t {

  std::unordered_map<std::string, std::function<void(tile_draw_data_t&, fte_t::tile_t&)>> id_callbacks;
  std::unordered_map<std::string, std::function<void(map_list_data_t::physics_entities_t&, compiled_map_t::physics_data_t&)>> sensor_id_callbacks;

  fan::vec2i view_size = 1;
  fan::graphics::render_view_t* render_view = nullptr;

  id_t open_map(compiled_map_t& out_compiled, const char* path, const properties_t& p = {}, const std::source_location& callers_path = std::source_location::current());

  id_t add(compiled_map_t* compiled_map);
  id_t add(compiled_map_t* compiled_map, const properties_t& p);

  void initialize(node_t& node, const fan::vec2& position);
  void initialize_visual(node_t& node, const fan::vec2& position);
  void erase_visual(id_t map_id, const std::string& id);

  fan::physics::entity_t add_sensor_rectangle(id_t map_id, const std::string& id);
  fan::physics::entity_t add_sensor_circle(id_t map_id, const std::string& id);

  f32_t get_depth_from_y(id_t map_id, const fan::vec2& position);

  struct userdata_t {
    int key;
    int key_state;
  };

  void add_tile(node_t& node, fte_t::tile_t& j, int x, int y, std::uint32_t depth);
  void clear_visual(node_t& node);

  void clear(id_t& id);
  void clear(node_t& node);
  void erase(id_t& id);

  void remove_visual(id_t id, const std::string& str_id, const fan::vec2& position);

  struct shape_depths_t {
    static constexpr int max_layer_depth = 0xFAAA - 2;
  };

  static constexpr int max_layer_depth = 128;

  void adjust_view(fan::vec2i& src);
  void update(id_t id, const fan::vec2& position_);
  void erase_physics_entity(id_t map_id, const std::string& id);

  tile_draw_data_t* get_shape_by_id(id_t map_id, const std::string& id);
  fan::graphics::shape_t* get_light_by_id(tilemap_renderer_t::id_t map_id, const std::string& id);
};
#undef tilemap_renderer
#endif

#endif