#include <variant>
#include <fan/io/file.h>
#include <fan/io/json_impl.h>

struct fte_loader_t {

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

  struct compiled_map_t {
    fan::vec2i map_size;
    fan::vec2i tile_size;
    #if tilemap_renderer == 0
    std::vector<std::vector<std::vector<fte_t::tile_t>>> compiled_shapes;
    #elif tilemap_renderer == 1
    std::unordered_map<fan::vec2i, fan::mp_t<current_version_t::shapes_t>, vec2i_hasher> compiled_shapes;
    #endif
  };

  using tile_draw_data_t = std::variant<loco_t::shape_t,
    fan::graphics::collider_hidden_t, 
    fan::graphics::collider_sensor_t>;
    
  #include <fan/fan_bll_preset.h>

  struct map_list_data_t{
    compiled_map_t* compiled_map;
    std::unordered_map<fan::vec3i, tile_draw_data_t, vec3i_hasher> tiles;
  };

  #define BLL_set_prefix map_list
  #define BLL_set_type_node uint16_t
  #define BLL_set_StoreFormat 1
  #define BLL_set_NodeDataType map_list_data_t
  #define BLL_set_Link 1
  #define BLL_set_AreWeInsideStruct 1
protected:
  #include <BLL/BLL.h>
public:

  using id_t = map_list_NodeReference_t;
  using node_t = map_list_NodeData_t;

  map_list_t map_list;

  void open(loco_t::texturepack_t* tp) {
    texturepack = tp;
  }

  compiled_map_t compile(const std::string& filename) {

    std::string out;
    fan::io::file::read(filename, &out);
    fan::json json = fan::json::parse(out);
    if (json["version"] != 1) {
      fan::throw_error("version mismatch");
    }

    compiled_map_t compiled_map;

    compiled_map.map_size = json["map_size"];
    compiled_map.tile_size = json["tile_size"];

    compiled_map.compiled_shapes.resize(compiled_map.map_size.y);
    for (auto& i : compiled_map.compiled_shapes) {
      i.resize(compiled_map.map_size.x);
    }

    fan::graphics::shape_deserialize_t it;
    loco_t::shape_t shape;
    while (it.iterate(json["tiles"], &shape)) {
      auto shape_json = *(it.data.it - 1);
      fte_t::tile_t tile;
      fan::vec2i gp = shape.get_position();
      convert_draw_to_grid(compiled_map.tile_size, gp);
      gp /= compiled_map.tile_size * compiled_map.map_size;
      //gp += compiled_map.map_size / 2;
      tile.position = shape.get_position();
      tile.size = shape.get_size();
      tile.angle = shape.get_angle();
      tile.color = shape.get_color();
      tile.image_name = shape_json["image_name"];
      tile.mesh_property = (fte_t::mesh_property_t)shape_json["mesh_property"];
      tile.id = shape_json["id"];
      tile.action = shape_json.value("action", fte_t::actions_e::none);
      tile.key = shape_json.value("key", fan::key_invalid);
      tile.key_state = shape_json.value("key_state", (int)fan::keyboard_state::press);
      tile.flags = shape.get_flags();
      compiled_map.compiled_shapes[gp.y][gp.x].push_back(tile);
    }//
    return compiled_map;
  }

  struct properties_t {
    fan::vec3 position = 0;
    fan::vec2 size = 1;
    fan::vec3 offset = 0;
    fan::graphics::camera_t* camera = nullptr;
  };

  static void convert_draw_to_grid(fan::vec2i tile_size, fan::vec2i& p) {

  }

 /* id_t add(compiled_map_t* compiled_map) {
    add(compiled_map, properties_t());
  }*/

  //id_t add(compiled_map_t* compiled_map, const properties_t& p) {
  //  auto it = map_list.NewNodeLast();
  //  auto& node = map_list[it];
  //  node.compiled_map = compiled_map;
  //  fan::vec2 origin = 0;//-fan::vec2(compiled_map->map_size * compiled_map->tile_size / 2) * p.size;
  //  for (auto& i : compiled_map->compiled_shapes) {
  //    for (auto x : i) {
  //      for (auto& j : x.tile.layers) {
  //        p.object_add_cb(j);

  //        switch (j.mesh_property) {
  //          case fte_t::mesh_property_t::none: {
  //            // set map origin point to 0
  //            node.tiles.push_back(fan::graphics::sprite_t{ {
  //                .position = fan::vec3(origin + *(fan::vec2*)&p.position + fan::vec2(j.position) * p.size, j.position.z + p.position.z),
  //                .size = j.size * p.size,
  //                .angle = j.angle,
  //                .color = j.color
  //            } });
  //            loco_t::texturepack_t::ti_t ti;
  //            if (texturepack->qti(j.image_hash, &ti)) {
  //              fan::throw_error("failed to load image from .fte - corrupted save file");
  //            }
  //            gloco->sprite.load_tp(
  //              map_list[it].tiles.back(),
  //              &ti
  //            );
  //            break;
  //          }
  //          case fte_t::mesh_property_t::collider: {
  //            node.collider_hidden.push_back(
  //              fan::graphics::collider_hidden_t(
  //                *(fan::vec2*)&p.position + fan::vec2(j.position) * p.size,
  //                j.size * p.size
  //              )
  //            );
  //            break;
  //          }
  //          case fte_t::mesh_property_t::sensor: {
  //            node.collider_sensor.push_back(
  //              fan::graphics::collider_sensor_t(
  //                *(fan::vec2*)&p.position + fan::vec2(j.position) * p.size,
  //                j.size * p.size
  //              )
  //            );
  //            break;
  //          }
  //          case fte_t::mesh_property_t::light: {
  //            node.tiles.push_back(fan::graphics::light_t{ {
  //              .position = fan::vec3(origin + *(fan::vec2*)&p.position + fan::vec2(j.position) * p.size, j.position.z + p.position.z),
  //              .size = j.size * p.size,
  //              .color = j.color
  //            } });
  //            break;
  //          }
  //        }
  //      }
  //    }
  //  }
  //  return it;
  //}

  loco_t::texturepack_t* texturepack;
};