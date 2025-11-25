module;

#if defined(fan_physics) && defined(fan_gui)


#include <fan/utility.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <variant>
#include <source_location>
#include <sstream>
#include <fstream>

#endif

export module fan.graphics.gui.tilemap_editor.loader;

#if defined(fan_physics) && defined(fan_gui)
import fan.print;
import fan.utility;
import fan.graphics;
import fan.graphics.physics_shapes;
import fan.io.file;
import fan.types.json;
import fan.physics.types;
import fan.physics.b2_integration;

export struct fte_loader_t {

  struct fte_t {
    #include "common2.h"
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
    #elif tilemap_renderer == 1
    std::unordered_map<fan::vec2i, fan::mp_t<current_version_t::shapes_t>, vec2i_hasher> compiled_shapes;
    #endif
  };

  struct tile_draw_data_t : fan::graphics::shape_t {
    using fan::graphics::shape_t::shape_t;
    std::string id;
  };

  #include <fan/fan_bll_preset.h>

  struct map_list_data_t{
    compiled_map_t* compiled_map;
    std::unordered_map<fan::vec3i, tile_draw_data_t, vec3i_hasher> rendered_tiles;
    std::unordered_map<std::string, tile_draw_data_t*> id_to_shape;
    struct physics_entities_t {
      std::variant<
        fan::graphics::physics::rectangle_t,
        fan::graphics::physics::circle_t
      >visual;
      std::string id;
    };
    std::vector<physics_entities_t> physics_entities;
    fan::vec3 position = 0;
    fan::vec2 size = 1;
    fan::vec2i prev_render = 10000000;
  };

  #define BLL_set_prefix map_list
  #define BLL_set_type_node uint16_t
  #define bcontainer_set_StoreFormat 1
  #define BLL_set_NodeDataType map_list_data_t
  #define BLL_set_Link 1
  #define BLL_set_AreWeInsideStruct 1
  #include <fan/fan_bll_preset.h>
protected:
  #include <BLL/BLL.h>
public:

  using id_t = map_list_NodeReference_t;
  using node_t = map_list_NodeData_t;

  map_list_t map_list;

  void iterate_physics_entities(id_t map_id, auto l) {
    auto& node = map_list[map_id];
    for (auto& i : node.physics_entities) {
      bool stop = std::visit([&]<typename T>(T & entity) -> bool {
        return l(i, entity);
      }, i.visual);
      if (stop) {
        break;
      }
    }
  }

  fan::physics::body_id_t get_physics_body(id_t map_id, const std::string& id) {
    fan::physics::body_id_t body;
    iterate_physics_entities(map_id,
      [&]<typename T>(auto& entity, T & entity_visual) -> bool {
      if (entity.id == id) {
        body = entity_visual;
        return true; // quit iterating
      }
      return false;
    });
    return body;
  }

  bool get_body(id_t map_id, const std::string& id, fte_t::tile_t& tile) {
    auto& node = map_list[map_id];
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
      auto& shapes = node.compiled_map->compiled_shapes;
      for (auto& i : shapes) {
        for (auto& j : i) {
          for (auto& k : j) {
            if (id == k.id) {
              tile = k;
              return true;
            }
          }
        }
      }
    }
    return false;
  }

  fan::vec3 get_position(id_t map_id, const std::string& id) {
    fte_t::tile_t tile;
    if (get_body(map_id, id, tile)) {
      return tile.position;
    }
    fan::throw_error("failed to find id");
    return {};
  }

  void open() {

  }

  compiled_map_t compile(const std::string& filename, const std::source_location& callers_path = std::source_location::current()) {
#if defined (fan_json)
    std::string out;
    fan::io::file::read(fan::io::file::find_relative_path(filename, callers_path), &out);
    fan::json json = fan::json::parse(out);
    if (json["version"] != 1) {
      fan::throw_error("version mismatch");
    }

    compiled_map_t compiled_map;

    compiled_map.lighting.ambient = json["lighting.ambient"];
    compiled_map.map_size = json["map_size"];
    compiled_map.tile_size = json["tile_size"];
    if (json.contains("gravity")) {
      fan::physics::gphysics->set_gravity(json["gravity"]);
    }

    compiled_map.compiled_shapes.resize(compiled_map.map_size.y);
    for (auto& i : compiled_map.compiled_shapes) {
      i.resize(compiled_map.map_size.x);
    }

    fan::graphics::shape_deserialize_t it;
    fan::graphics::shape_t shape;
    while (it.iterate(json["tiles"], &shape)) {
      auto& shape_json = *(it.data.it - 1);
      if (shape_json.contains("mesh_property") && shape_json["mesh_property"] == fte_t::mesh_property_t::physics_shape) {
        compiled_map.physics_shapes.resize(compiled_map.physics_shapes.size() + 1);
        compiled_map_t::physics_data_t& physics_element = compiled_map.physics_shapes.back();

        physics_element.position = shape.get_position();
        physics_element.size = shape.get_size();
        fte_t::physics_shapes_t defaults;
        physics_element.physics_shapes.id = shape_json.value("id", defaults.id);
        const fan::json& physics_shape_data = shape_json["physics_shape_data"];
        physics_element.physics_shapes.type = physics_shape_data.value("type", defaults.type);
        physics_element.physics_shapes.body_type = physics_shape_data.value("body_type", defaults.body_type);
        physics_element.physics_shapes.draw = physics_shape_data.value("draw", defaults.draw);
        physics_element.physics_shapes.shape_properties.friction = physics_shape_data.value("friction", defaults.shape_properties.friction);
        physics_element.physics_shapes.shape_properties.density = physics_shape_data.value("density", defaults.shape_properties.density);
        physics_element.physics_shapes.shape_properties.fixed_rotation = physics_shape_data.value("fixed_rotation", defaults.shape_properties.fixed_rotation);
        physics_element.physics_shapes.shape_properties.presolve_events = physics_shape_data.value("presolve_events", defaults.shape_properties.presolve_events);
        physics_element.physics_shapes.shape_properties.is_sensor = physics_shape_data.value("is_sensor", defaults.shape_properties.is_sensor);
       
        continue;
      }
      fte_t::tile_t tile;
      fan::vec2i gp = shape.get_position();
      gp /= compiled_map.tile_size * 2;
      //gp += compiled_map.map_size / 2;
      tile.position = shape.get_position();
      tile.size = shape.get_size();
      tile.angle = shape.get_angle();
      tile.color = shape.get_color();
      if (shape.get_shape_type() == fan::graphics::shape_type_t::sprite) {
        tile.texture_pack_unique_id = ((fan::graphics::shapes::sprite_t::ri_t*)shape.GetData(fan::graphics::g_shapes->shaper))->texture_pack_unique_id;
      }
      else if (shape.get_shape_type() == fan::graphics::shape_type_t::unlit_sprite) {
        tile.texture_pack_unique_id = ((fan::graphics::shapes::unlit_sprite_t::ri_t*)shape.GetData(fan::graphics::g_shapes->shaper))->texture_pack_unique_id;
      }
      // NOTE physics shapes are skipped, they are stored in compiled_map.physics_shapes
      tile.mesh_property = (fte_t::mesh_property_t)shape_json.value("mesh_property", fte_t::tile_t().mesh_property);
      tile.id = shape_json.value("id", fte_t::tile_t().id);
      tile.action = shape_json.value("action", fte_t::actions_e::none);
      tile.key = shape_json.value("key", fan::key_invalid);
      tile.key_state = shape_json.value("key_state", (int)fan::keyboard_state::press);
      tile.flags = shape.get_flags();
      compiled_map.compiled_shapes[gp.y][gp.x].push_back(tile);
    }//
    return compiled_map;
#else
    fan::throw_error("fan_json not enabled");
    __unreachable();
#endif
  }

  fan::vec2 convert_to_grid(const fan::vec2& p, const node_t& node) {
    return ((p / node.size) / (node.compiled_map->tile_size)).floor();
  }

  fan::vec2 get_tile_size(id_t id) {
    const auto& node = map_list[id];
    return node.compiled_map->tile_size;
  }
  // returns map's tile count
  fan::vec2 get_map_size(id_t id) {
    const auto& node = map_list[id];
    return node.compiled_map->map_size;
  }

  struct properties_t {
    fan::vec3 position = 0;
    fan::vec2 size = 1;
    fan::vec3 offset = 0;
    fan::vec2 scale = 1;
    fan::graphics::render_view_t* render_view = nullptr;
  };

  using physics_entities_t = map_list_data_t::physics_entities_t;
  using physics_data_t = compiled_map_t::physics_data_t;
};
#endif