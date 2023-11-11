struct ftme_loader_t {

  #define tilemap_editor_loader
  #include "common.h"

  struct compiled_map_t {
    fan::vec2ui map_size;
    fan::vec2ui tile_size;
    std::vector<fan::mp_t<current_version_t::shapes_t>> compiled_shapes;
  };

  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix map_list
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeData \
    compiled_map_t* compiled_map; \
    std::vector<loco_t::shape_t> tiles; \
    std::vector<fan::graphics::collider_static_t> collider_static;
  #define BLL_set_Link 1
  #define BLL_set_AreWeInsideStruct 1
protected:
  #include _FAN_PATH(BLL/BLL.h)
public:

  using id_t = map_list_NodeReference_t;

  map_list_t map_list;

  void open(loco_t::texturepack_t* tp) {
    texturepack = tp;
  }

  compiled_map_t compile(const fan::string& filename) {
    compiled_map_t compiled_map;

    fan::vec2ui& map_size = compiled_map.map_size;
    fan::vec2ui& tile_size = compiled_map.tile_size;

    #define tilemap_editor_loader
    #include "loader_versions/1.h"

    return compiled_map;
  }

  struct properties_t {
    fan::vec3 position = 0;
    fan::vec2 size = 1;
  };

  id_t add(compiled_map_t* compiled_map) {
    add(compiled_map, properties_t());
  }

  id_t add(compiled_map_t* compiled_map, const properties_t& p) {
    auto it = map_list.NewNodeLast();
    auto& node = map_list[it];
    node.compiled_map = compiled_map;
    for (auto& i : compiled_map->compiled_shapes) {
      // set map origin point to 0
      fan::vec2 origin = -fan::vec2(compiled_map->map_size * compiled_map->tile_size / 2) * p.size;
      node.tiles.push_back(fan::graphics::sprite_t{{
          .position = fan::vec3(origin + *(fan::vec2*)&p.position + i.tile.position * p.size, p.position.z),
          .size = compiled_map->tile_size * p.size
      }});
      loco_t::texturepack_t::ti_t ti;
      if (texturepack->qti(i.tile.image_hash, &ti)) {
        fan::throw_error("failed to load image from .ftme - corrupted save file");
      }
      gloco->shapes.sprite.load_tp(
        map_list[it].tiles.back(),
        &ti
      );
      // todo fix
      if (i.tile.mesh_property != 0) {
        node.collider_static.push_back(fan::graphics::collider_static_t{
          fan::graphics::sprite_t{{
              .position = node.tiles.back().get_position(),
              .size = node.tiles.back().get_size(),
              .color = fan::color(0, 0, 0, 0)
            }}
        });
      }
    }
    return it;
  }

  loco_t::texturepack_t* texturepack;
};