#include "loader.h"

struct fte_renderer_t : fte_loader_t {

  fan::vec3 position = 0;
  fan::vec2 view_size = 1;
  fan::vec2 size = 1;

  fan::vec2i prev_render = 0;

  id_t add(compiled_map_t* compiled_map) {
    add(compiled_map, properties_t());
  }

  id_t add(compiled_map_t* compiled_map, const properties_t& p) {
    auto it = map_list.NewNodeLast();
    auto& node = map_list[it];
    node.compiled_map = compiled_map;
    return it;
  }
  void set_position(id_t id, const fan::vec3& position_) {
    auto& node = map_list[id];
    fan::vec3 offset = position - position_;
    position = position_;
    for (auto& i : node.tiles) {
      i.set_position(i.get_position() + offset);
    }
    for (auto& i : node.collider_sensor) {
      i.set_position(i.get_position() + offset);
    }
    for (auto& i : node.collider_hidden) {
      i.set_position(i.get_position() + offset);
    }
  }

  void add_tile(node_t& node, fte_t::tile_t& j) {
    switch (j.mesh_property) {
      case fte_t::mesh_property_t::none: {
        node.tiles.push_back(fan::graphics::sprite_t{ {
            .position = position + fan::vec3(fan::vec2(j.position) * size, j.position.z),
            .size = j.size * size,
            .angle = j.angle,
            .color = j.color
        } });
        loco_t::texturepack_t::ti_t ti;
        if (texturepack->qti(j.image_hash, &ti)) {
          fan::throw_error("failed to load image from .fte - corrupted save file");
        }
        gloco->shapes.sprite.load_tp(
          node.tiles.back(),
          &ti
        );
        break;
      }
      case fte_t::mesh_property_t::collider: {
        node.collider_hidden.push_back(
          fan::graphics::collider_hidden_t(
            *(fan::vec2*)&position + fan::vec2(j.position) * size,
            j.size * size
          )
        );
        break;
      }
      case fte_t::mesh_property_t::sensor: {
        node.collider_sensor.push_back(
          fan::graphics::collider_sensor_t(
            *(fan::vec2*)&position + fan::vec2(j.position) * size,
            j.size * size
          )
        );
        break;
      }
      case fte_t::mesh_property_t::light: {
        node.tiles.push_back(fan::graphics::light_t{ {
          .position = position + fan::vec3(fan::vec2(j.position) * size, j.position.z),
          .size = j.size * size,
          .color = j.color
        } });
        break;
      }
    }
  }

  void clear(node_t& node) {
    node.tiles.clear();
    for (auto& i : node.collider_sensor) {
      i.close();
    }
    node.collider_sensor.clear();
    for (auto& i : node.collider_hidden) {
      i.close();
    }

    node.collider_hidden.clear();
  }

  void update(id_t id, const fan::vec2& position_, const fan::vec2& view_size_) {
    auto& node = map_list[id];

    if (prev_render == fan::vec2i(position_ / node.compiled_map->tile_size)) {
      return;
    }
    prev_render = position_ / node.compiled_map->tile_size;

    clear(node);
    //position = position_;
    
    view_size = view_size_ / node.compiled_map->tile_size;
    fan::vec2i src = (position_ / node.compiled_map->tile_size + node.compiled_map->map_size) / 2;
    src.x -= view_size.x / 2;
    src.y -= view_size.y / 2;

    // starting from -1 0
    src.y += 2;
    //position -= (node.compiled_map->tile_size * node.compiled_map->map_size) / 2;
    auto& map_tiles = node.compiled_map->compiled_shapes;
    // doesnt calculate optimal layer count
    for (int y = 0; y < view_size.y; ++y) {
      for (int x = 0; x < view_size.x; ++x) {
        fan::vec2i grid_pos = src + fan::vec2i(x, y);
        if (grid_pos.y < map_tiles.size() && grid_pos.x < map_tiles[grid_pos.y].size()) {
          if (!map_tiles[grid_pos.y][grid_pos.x].tile.layers.empty()) {
            for (auto& j : map_tiles[grid_pos.y][grid_pos.x].tile.layers) {
              add_tile(node, j);
            }
          }
        }
      }
    }
  }

  // hard update - only call in view resize
  void update(id_t id, const properties_t& p) {
    position = p.position;
    auto& node = map_list[id];
    clear(node);
    position = p.position;//-fan::vec2(compiled_map->map_size * compiled_map->tile_size / 2) * p.size;
    size = p.size;
    for (auto& i : node.compiled_map->compiled_shapes) {
      for (auto& x : i) {
        for (auto& j : x.tile.layers) {
          p.object_add_cb(j);
          add_tile(node, j);
        }
      }
    }
  }

private:
  fte_loader_t::add;
};