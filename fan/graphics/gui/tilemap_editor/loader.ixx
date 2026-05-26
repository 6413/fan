module;

#include <cstdint>

#if defined(FAN_2D)
  #if defined(FAN_PHYSICS_2D)
    #include <fan/utility.h>
  #endif
#endif

export module fan.graphics.tilemap_editor.loader;

import std;

#if defined(FAN_2D)

#if defined(FAN_PHYSICS_2D)
import fan.types;
import fan.types.vector;
import fan.types.color;
import fan.texture_pack.tp0;
import fan.graphics.common_context;
import fan.graphics.physics_shapes;
import fan.graphics.shapes;
import fan.io.file;
import fan.physics.types;
import fan.physics.b2_integration;
import fan.window.input;
import fan.print.error;
import fan.memory;

export namespace fan::graphics {
  struct tilemap_loader_t {
    struct fte_t {
    #include "common2.h"

      struct spawn_mark_data_t {
        fan::vec3 position;
        fan::vec2 size;
        fte_t::mesh_property_t type;
        std::string id;
        fan::color color;
      };
    };

    struct vec2i_hasher {
      std::size_t operator()(const fan::vec2i& k) const {
        std::hash<int> hasher;
        std::size_t hash_value = 17;
        hash_value = hash_value * 31 + hasher(k.x);
        hash_value = hash_value * 31 + hasher(k.y);
        return hash_value;
      }
    };

    struct vec3i_hasher {
      std::size_t operator()(const fan::vec3i& k) const {
        std::hash<int> hasher;
        std::size_t hash_value = 17;
        hash_value = hash_value * 31 + hasher(k.x);
        hash_value = hash_value * 31 + hasher(k.y);
        hash_value = hash_value * 31 + hasher(k.z);
        return hash_value;
      }
    };

    using tile_t = fte_t::tile_t;

    struct compiled_map_t {
      fan::vec2i map_size = 0;
      fan::vec2i tile_size = 0;
    #if tilemap_renderer == 0
      std::vector<std::vector<std::vector<fte_t::tile_t>>> compiled_shapes;
      struct physics_data_t {
        fan::vec3 position;
        fan::vec2 size;
        fte_t::physics_shapes_t physics_shapes;
      };
      std::vector<physics_data_t> physics_shapes;
      fan::graphics::lighting_t lighting = fan::graphics::get_lighting();
      std::unordered_map<std::string, std::vector<fte_t::tile_t*>> id_lookup;
      std::vector<fte_t::spawn_mark_data_t> spawn_marks;
    #elif tilemap_renderer == 1
      std::unordered_map<fan::vec2i, fan::mp_t<current_version_t::shapes_t>, vec2i_hasher> compiled_shapes;
    #endif
    };

    struct tile_draw_data_t : fan::graphics::shape_t {
      using fan::graphics::shape_t::shape_t;
      std::string id;
    };

    struct light_with_id_t {
      std::string id;
      fan::graphics::shape_t shape;
    };

    struct map_list_data_t {
      compiled_map_t* compiled_map;
      std::unordered_map<fan::vec3i, tile_draw_data_t, vec3i_hasher> rendered_tiles;

      std::vector<light_with_id_t> lights;
      std::unordered_map<std::string, tile_draw_data_t*> id_to_shape;
      struct physics_entities_t {
        std::variant<
          fan::graphics::physics::rectangle_t,
          fan::graphics::physics::circle_t
        > visual;
        std::string id;
      };
      std::vector<physics_entities_t> physics_entities;
      fan::vec3 position = 0;
      fan::vec2 size = 1;
      fan::vec2i prev_render = 10000000;
      std::function<f32_t(const fte_t::tile_t&, const fan::vec2&, const fan::vec2&, f32_t)> depth_fn = nullptr;
    };

    struct map_list_impl_t;
    map_list_impl_t* impl = nullptr;
    std::unordered_map<std::string, compiled_map_t> compiled_maps;

    using id_t = std::uint16_t;
    using node_t = map_list_data_t;

    tilemap_loader_t();
    ~tilemap_loader_t();

    id_t new_map_node();
    void delete_map_node(id_t id);
    node_t& get_map_node(id_t id);
    bool is_map_node_invalid(id_t id) const;
    void invalidate_map_node(id_t& id) const;

    void iterate_physics_entities(id_t map_id, auto l) {
      auto& node = get_map_node(map_id);
      for (auto& i : node.physics_entities) {
        bool stop = std::visit([&]<typename T>(T & entity_visual) -> bool {
          return l(i, entity_visual);
        }, i.visual);
        if (stop) break;
      }
    }

    void iterate_visual(id_t map_id, std::function<bool(fte_t::tile_t&)> cb) {
      auto& node = get_map_node(map_id);
      auto& shapes = node.compiled_map->compiled_shapes;
      for (auto& i : shapes) {
        for (auto& j : i) {
          for (auto& k : j) {
            if (cb(k)) return;
          }
        }
      }
    }

    void iterate_marks(id_t map_id, std::function<bool(fte_t::spawn_mark_data_t&)> cb) {
      auto& node = get_map_node(map_id);
      for (auto& mark : node.compiled_map->spawn_marks) {
        if (cb(mark)) break;
      }
    }

    void iterate_marks(id_t map_id, std::initializer_list<std::pair<std::string_view, std::function<void(fte_t::spawn_mark_data_t&)>>> dispatch) {
      iterate_marks(map_id, [&](fte_t::spawn_mark_data_t& mark) -> bool {
        for (auto& [key, fn] : dispatch) {
          if (mark.id == key) { fn(mark); break; }
        }
        return false;
      });
    }

    fan::physics::body_id_t get_physics_body(id_t map_id, const std::string& id) {
      fan::physics::body_id_t body;
      iterate_physics_entities(map_id, [&]<typename T>(auto& entity, T & entity_visual) -> bool {
        if (entity.id == id) {
          body = entity_visual;
          return true;
        }
        return false;
      });
      return body;
    }

    std::vector<fan::physics::body_id_t> get_physics_bodies(id_t map_id, const std::string& id) {
      std::vector<fan::physics::body_id_t> bodies;
      iterate_physics_entities(map_id, [&]<typename T>(auto& entity, T & entity_visual) -> bool {
        if (entity.id == id) {
          bodies.emplace_back(entity_visual);
        }
        return false;
      });
      return bodies;
    }

    bool get_body(id_t map_id, const std::string& id, fte_t::tile_t& tile) {
      auto& node = get_map_node(map_id);
      {
        auto& shapes = node.compiled_map->physics_shapes;
        for (auto& shape : shapes) {
          if (shape.physics_shapes.id == id) {
            tile.position = shape.position;
            tile.size = shape.size;
            return true;
          }
        }
      }
      {
        auto& compiled_map = *node.compiled_map;
        auto it = compiled_map.id_lookup.find(id);
        if (it != compiled_map.id_lookup.end() && !it->second.empty()) {
          tile = *it->second[0];
          return true;
        }
      }
      return false;
    }

    bool get_bodies(id_t map_id, const std::string& id, std::vector<fte_t::tile_t>& tiles) {
      auto& node = get_map_node(map_id);
      {
        auto& shapes = node.compiled_map->physics_shapes;
        for (auto& shape : shapes) {
          if (shape.physics_shapes.id == id) {
            fte_t::tile_t tile;
            tile.position = shape.position;
            tile.size = shape.size;
            tiles.emplace_back(std::move(tile));
          }
        }
      }
      {
        auto& compiled_map = *node.compiled_map;
        auto it = compiled_map.id_lookup.find(id);
        if (it != compiled_map.id_lookup.end()) {
          for (auto* t : it->second) {
            tiles.emplace_back(*t);
          }
        }
      }
      return tiles.size();
    }

    bool get_visual_bodies(id_t map_id, const std::string& id, std::vector<fte_t::tile_t>* tiles) {
      auto& node = get_map_node(map_id);
      auto& compiled_map = *node.compiled_map;
      auto it = compiled_map.id_lookup.find(id);
      if (it != compiled_map.id_lookup.end()) {
        for (auto* t : it->second) {
          tiles->emplace_back(*t);
        }
      }
      return tiles->size();
    }

    fan::vec3 get_position(id_t map_id, const std::string& id) {
      fte_t::tile_t tile;
      if (get_body(map_id, id, tile)) {
        return tile.position;
      }
      fan::throw_error("failed to find id");
      return {};
    }

    fan::vec3 get_spawn(id_t map_id, const std::string_view id = "") {
      auto& node = get_map_node(map_id);
      auto& marks = node.compiled_map->spawn_marks;

      for (auto& mark : marks) {
        if ((id.size() ? (mark.id != id) : (mark.type != fte_t::mesh_property_t::player_spawn))) {
          continue;
        }
        return mark.position;
      }
      fan::throw_error("spawn position not found: " + std::string(id));
      return {};
    }

    fan::vec3 get_enemy_spawn(id_t map_id, const std::string_view id = "") {
      auto& node = get_map_node(map_id);
      auto& marks = node.compiled_map->spawn_marks;

      for (auto& mark : marks) {
        if ((id.size() ? (mark.id != id) : (mark.type != fte_t::mesh_property_t::enemy_spawn))) {
          continue;
        }
        return mark.position;
      }
      fan::throw_error("spawn position not found: " + std::string(id));
      return {};
    }

    std::vector<fan::vec3> get_all_spawn_positions(id_t map_id, fte_t::mesh_property_t type) {
      std::vector<fan::vec3> positions;
      auto& node = get_map_node(map_id);
      auto& marks = node.compiled_map->spawn_marks;

      for (auto& mark : marks) {
        if (mark.type == type) {
          positions.push_back(mark.position);
        }
      }
      return positions;
    }

    void open() {}

    compiled_map_t* compile(const std::string& filename,const std::source_location& callers_path = std::source_location::current());
    compiled_map_t* compile(const std::string& filename, const std::string& name, const std::source_location& callers_path = std::source_location::current());
    compiled_map_t* get_compiled(const std::string& name);

    fan::vec2 convert_to_grid(const fan::vec2& p, const node_t& node) {
      return ((p / node.size) / (node.compiled_map->tile_size)).floor();
    }

    fan::vec2 get_tile_size(id_t id) {
      const auto& node = get_map_node(id);
      return node.compiled_map->tile_size;
    }

    fan::vec2 get_map_size(id_t id) {
      const auto& node = get_map_node(id);
      return node.compiled_map->map_size;
    }

    std::size_t count(id_t id, const std::string& str_id) {
      auto& map = *get_map_node(id).compiled_map;
      std::size_t c = 0;

      if (auto it = map.id_lookup.find(str_id); it != map.id_lookup.end()) {
        c += it->second.size();
      }
      for (auto& s : map.physics_shapes) {
        c += s.physics_shapes.id == str_id;
      }
      for (auto& m : map.spawn_marks) {
        c += m.id == str_id;
      }

      return c;
    }

    struct properties_t {
      fan::vec3 position = 0;
      fan::vec2 size = 512;
      fan::vec3 offset = 0;
      fan::vec2 scale = 1;
      std::uint8_t collision_body_type = fan::physics::body_type_e::static_body;
      fan::physics::shape_properties_t collision_props;
      bool build_collisions = false;
      f32_t default_friction = -1.f;
      fan::graphics::render_view_t* render_view = nullptr;
      std::function<f32_t(const fte_t::tile_t&, const fan::vec2&, const fan::vec2&, f32_t)> depth_fn = nullptr;
    };

    static inline auto default_depth_fn = [](const tilemap_loader_t::fte_t::tile_t& t, const fan::vec2& world_pos, const fan::vec2& world_size, f32_t tile_size_y) -> f32_t {
      return (t.position.z -  (0xFAAA) / 2.f) + (world_pos.y / tile_size_y) + (0xFAAA - 2) / 2.f;
    };

    using physics_entities_t = map_list_data_t::physics_entities_t;
    using physics_data_t = compiled_map_t::physics_data_t;
  };
}

#endif
#endif  