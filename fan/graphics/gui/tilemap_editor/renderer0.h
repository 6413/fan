#define tilemap_renderer 0
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
   
    view_size = p.size / node.compiled_map->tile_size / 2;
    prev_render = (p.position / node.compiled_map->tile_size / 2).floor();

    initialize(node, p.position);

    return it;
  }

  void initialize(node_t& node, const fan::vec2& position) {

    clear(node);

    resize_map(node);

    fan::vec2i src = (position / node.compiled_map->tile_size).floor();
    src += node.compiled_map->map_size;
    src /= 2;
    src.x -= view_size.x / 2;
    src.y -= view_size.y / 2;

    auto& map_tiles = node.compiled_map->compiled_shapes;

    for (int y = 0; y < view_size.y; ++y) {
      for (int x = 0; x < view_size.x; ++x) {
        fan::vec2i grid_pos = src + fan::vec2i(x, y);
        if (!(grid_pos.y < map_tiles.size() && grid_pos.x < map_tiles[grid_pos.y].size())) {
          continue;
        }
        if (map_tiles[grid_pos.y][grid_pos.x].tile.layers.empty()) {
          continue;
        }
        int depth = 0;
        for (auto& j : map_tiles[grid_pos.y][grid_pos.x].tile.layers) {
          add_tile(node, j, x, y, depth++);
        }
      }
    }
  }

  void add_tile(node_t& node, fte_t::tile_t& j, uint32_t x, uint32_t y, uint32_t depth) {
    switch (j.mesh_property) {
      case fte_t::mesh_property_t::none: {
        node.tiles[y][x][depth] = fan::graphics::sprite_t{{
            .position = position + fan::vec3(fan::vec2(j.position) * size, j.position.z),
            .size = j.size * size,
            .angle = j.angle,
            .color = j.color
        } };
        loco_t::texturepack_t::ti_t ti;
        if (texturepack->qti(j.image_hash, &ti)) {
          fan::throw_error("failed to load image from .fte - corrupted save file");
        }
        gloco->shapes.sprite.load_tp(
          std::get<loco_t::shape_t>(node.tiles[y][x][depth]),
          &ti
        );
        break;
      }
      case fte_t::mesh_property_t::light: {
        node.tiles[y][x][depth] = fan::graphics::light_t{ {
          .position = position + fan::vec3(fan::vec2(j.position) * size, j.position.z),
          .size = j.size * size,
          .color = j.color
        } };
        break;
      }
      case fte_t::mesh_property_t::collider: {
        node.tiles[y][x][depth] =
          fan::graphics::collider_hidden_t(
            *(fan::vec2*)&position + fan::vec2(j.position) * size,
            j.size * size
          )
        ;
        break;
      }
      case fte_t::mesh_property_t::sensor: {
        node.tiles[y][x][depth] =
          fan::graphics::collider_sensor_t(
            *(fan::vec2*)&position + fan::vec2(j.position) * size,
            j.size * size
          )
        ;
        break;
      }
    }
  }

  void clear(node_t& node) {
    for (auto& i : node.tiles) {
      for (auto& j : i) {
        for (auto& k : j) {
          std::visit([]<typename T>(T & v) {
            if constexpr (fan::same_as_any<T,
              fan::graphics::collider_hidden_t,
              fan::graphics::collider_sensor_t>) {
              v.close();
            }
          }, k);
        }
      }
    }
    node.tiles.clear();
  }

  static constexpr int max_layer_depth = 128;

  void resize_map(node_t& node) {
    node.tiles.resize(view_size.y);
    for (auto& j : node.tiles) {
      j.resize(view_size.x);
      for (auto& k : j) {
        k.resize(max_layer_depth);
      }
    }
  }

  void update(id_t id, const fan::vec2& position_) {
    auto& node = map_list[id];
    if (prev_render == (position_ / node.compiled_map->tile_size / 2).floor()) {
      return;
    }
    fan::vec2i old_render = prev_render;
    auto& map_tiles = node.compiled_map->compiled_shapes;

    prev_render = (position_ / node.compiled_map->tile_size / 2).floor();
    fan::vec2i offset = prev_render - old_render;

    //if (offset.x > 1 || offset.y > 1)
    fan::print(offset);

    if (offset.x > view_size.x || offset.y > view_size.y) {
      initialize(node, position_);
      return;
    }

    fan::vec2 src = (position_ / node.compiled_map->tile_size).floor();
    src += node.compiled_map->map_size;
    src /= 2;
    src.x -= view_size.x / 2;
    src.y -= view_size.y / 2;
    //src = src.floor();

   
   
    if (offset.x) {
      for (int k = 0; k < std::abs(offset.x); ++k) {
        f32_t x = 0;
        if (offset.x < 0) {
          for (int i = 0; i < view_size.y; ++i) {
            node.tiles[i].erase(node.tiles[i].end() - 1);
            node.tiles[i].push_front({});
            node.tiles[i].front().resize(max_layer_depth);
          }
        }
        else {
          for (int i = 0; i < view_size.y; ++i) {
            node.tiles[i].erase(node.tiles[i].begin());
            node.tiles[i].push_back({});
            node.tiles[i].back().resize(max_layer_depth);
          }
          x = node.tiles[0].size() - 1;
        }

        for (int i = 0; i < view_size.y; ++i) {
          fan::vec2i grid_pos = src + fan::vec2i(x - offset.x + k * fan::math::sgn(offset.x) + fan::math::sgn(offset.x), i);
          if (!(grid_pos.y < map_tiles.size() && grid_pos.x < map_tiles[grid_pos.y].size())) {
            continue;
          }
          if (map_tiles[grid_pos.y][grid_pos.x].tile.layers.empty()) {
            continue;
          }
          //fan::print(grid_pos);
          add_tile(node, map_tiles[grid_pos.y][grid_pos.x].tile.layers[0], x, i, 0);
        }
        //fan::print("\n");
      }
    }
    if (offset.y) {
      for (int k = 0; k < std::abs(offset.y); ++k) {
        int index = (offset.y < 0) ?
          (node.tiles.erase(node.tiles.end() - 1), node.tiles.push_front({}), 0) :
          (node.tiles.erase(node.tiles.begin()), node.tiles.push_back({}), node.tiles.size() - 1);

        node.tiles[index].resize(view_size.x);
        for (auto& i : node.tiles[index]) {
          i.resize(max_layer_depth);
        }

        f32_t y = (offset.y < 0) ? 0 : node.tiles.size() - 1;

        for (int i = 0; i < view_size.x; ++i) {
          fan::vec2i grid_pos = src + fan::vec2i(i, y - offset.y + k * fan::math::sgn(offset.y) + fan::math::sgn(offset.y));
          if (!(grid_pos.y < map_tiles.size() && grid_pos.x < map_tiles[grid_pos.y].size())) {
            continue;
          }
          if (map_tiles[grid_pos.y][grid_pos.x].tile.layers.empty()) {
            continue;
          }
          // fan::print(grid_pos);
          add_tile(node, map_tiles[grid_pos.y][grid_pos.x].tile.layers[0], i, y, 0);
        }
        // fan::print("\n");
      }
    }
  }

private:
 // fte_loader_t::add;
};

#undef tilemap_renderer