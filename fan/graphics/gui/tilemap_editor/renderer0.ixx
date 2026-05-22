module;

#if defined(FAN_2D)
#include <coroutine>
#endif

export module fan.graphics.tilemap_editor.renderer;

import std;

#if defined(FAN_2D)

#if defined(FAN_PHYSICS_2D)

export import fan.graphics.tilemap_editor.loader;

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

export namespace fan::graphics {
  struct tilemap_renderer_t : fan::graphics::tilemap_loader_t {

    std::unordered_map<std::string, std::function<void(tile_draw_data_t&, fte_t::tile_t&)>> id_callbacks;
    std::unordered_map<std::string, std::function<void(map_list_data_t::physics_entities_t&, compiled_map_t::physics_data_t&)>> sensor_id_callbacks;

    fan::vec2i view_size = 1;
    fan::graphics::render_view_t* render_view = nullptr;

    id_t open_map(compiled_map_t& out_compiled, const char* path, const properties_t& p = {}, const std::source_location& callers_path = std::source_location::current());
    // compiles map also
    id_t open_map(std::string_view file_name, const properties_t& p = {}, const std::source_location& callers_path = std::source_location::current());
    id_t open_map(std::string_view map_name, std::string_view file_name, const properties_t& p = {}, const std::source_location& callers_path = std::source_location::current());
    // closes map, but keeps the compiled map cached
    void close_map(id_t& id);

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

    f32_t get_dynamic_depth(id_t map_id, const fan::vec2& position, f32_t body_height);

    // could be something else than auto?
    void setup_view(id_t id, auto& body, auto& camera, f32_t zoom) {
      body.set_physics_position(get_spawn_position(id));
      camera.set_zoom(zoom);
      fan::vec2 map_size = get_map_size(id);
      fan::vec2 tile_size = get_tile_size(id);
      fan::vec2 map_center = get_map_node(id).position + (map_size * tile_size) - tile_size;
      camera.set_center(map_center);
    }

    void iterate_tiles(id_t map_id, auto cb) {
      auto& node = get_map_node(map_id);
      for (const auto& row : node.compiled_map->compiled_shapes) {
        for (const auto& col : row) {
          for (const auto& tile : col) {
            cb(tile);
          }
        }
      }
    }
  };

  struct tilemap_instance_t {
    tilemap_instance_t() = default;
    tilemap_instance_t(tilemap_renderer_t& r, std::string_view path, const tilemap_renderer_t::properties_t& p = {},
                       const std::source_location& loc = std::source_location::current());

    tilemap_instance_t(const tilemap_instance_t&) = delete;
    tilemap_instance_t& operator=(const tilemap_instance_t&) = delete;

    tilemap_instance_t(tilemap_instance_t&& o) noexcept;
    tilemap_instance_t& operator=(tilemap_instance_t&& o) noexcept;

    ~tilemap_instance_t();

    void update(const fan::vec2& pos);
    void close();

    void build_collisions(
      std::uint8_t bt = fan::physics::body_type_e::static_body,
      fan::physics::shape_properties_t props = {}
    );

    fan::vec2 get_tile_size() const {
      return renderer->get_tile_size(id);
    }

    fan::vec2 get_map_size() const {
      return renderer->get_map_size(id);
    }

    void setup_view(auto& body, auto& ic, f32_t scale) {
      renderer->setup_view(id, body, ic, scale);
    }

    void iterate_marks(std::initializer_list<std::pair<std::string_view, std::function<void(tilemap_loader_t::fte_t::spawn_mark_data_t&)>>> dispatch) {
      renderer->iterate_marks(id, dispatch);
    }

    void iterate_marks(std::function<bool(tilemap_loader_t::fte_t::spawn_mark_data_t&)> cb) {
      renderer->iterate_marks(id, cb);
    }

    tilemap_renderer_t* renderer = nullptr;
    tilemap_renderer_t::id_t id;
    std::vector<fan::physics::entity_t> collisions;
  };
}
#undef tilemap_renderer
#endif

#endif