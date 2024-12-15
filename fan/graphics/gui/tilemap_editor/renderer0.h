#define tilemap_renderer 0
#include "loader.h"


struct fte_renderer_t : fte_loader_t {

  std::unordered_map<std::string, fan::function_t<void(tile_draw_data_t&, fte_t::tile_t&)>> id_callbacks;

  fan::vec3 position = 0;
  fan::vec2 size = 1;
  fan::vec2i view_size = 1;
  fan::graphics::camera_t* camera = nullptr;

  fan::vec2i prev_render = 0;

  id_t add(compiled_map_t* compiled_map) {
    return add(compiled_map, properties_t());
  }

  id_t add(compiled_map_t* compiled_map, const properties_t& p) {

    if (p.camera == nullptr) {
      camera = &gloco->orthographic_camera;
    }
    else {
      camera = p.camera;
    }

    auto it = map_list.NewNodeLast();
    auto& node = map_list[it];
    node.compiled_map = compiled_map;
   
    view_size = p.size / node.compiled_map->tile_size / 2;
    prev_render = (p.position / node.compiled_map->tile_size / 2).floor();

    position = p.offset;
    initialize(node, p.position);


    return it;
  }

  void initialize(node_t& node, const fan::vec2& position) {

    clear(node);

    fan::vec2i src = (position / node.compiled_map->tile_size).floor();
    //src.x -= view_size.x / 2;
    //src.y -= view_size.y / 2;

    auto& map_tiles = node.compiled_map->compiled_shapes;

    for (int y = 0; y < view_size.y; ++y) {
      for (int x = 0; x < view_size.x; ++x) {
        fan::vec2i grid_pos = src + fan::vec2i(x, y);
        if (!(grid_pos.y < (int64_t)map_tiles.size() && grid_pos.x < (int64_t)map_tiles[grid_pos.y].size())) {
          continue;
        }
        if (grid_pos.y >= (int64_t)map_tiles.size() || grid_pos.x >= (int64_t)map_tiles[grid_pos.y].size()) {
          continue;
        }
        if (grid_pos.x < 0 || grid_pos.y < 0) {
          continue;
        }
        if (map_tiles[grid_pos.y][grid_pos.x].empty()) {
          continue;
        }
        int depth = 0;
        for (auto& j : map_tiles[grid_pos.y][grid_pos.x]) {
          add_tile(node, j, src.x + x, src.y + y, depth++);
        }
      }
    }
  }

  struct userdata_t {
    int key;
    int key_state;
  };

  void add_tile(node_t& node, fte_t::tile_t& j, int x, int y, uint32_t depth) {
    int additional_depth = y + node.compiled_map->map_size.y / 2;
    switch (j.mesh_property) {
      case fte_t::mesh_property_t::none: {
        node.tiles[fan::vec3i(x, y, depth)] = fan::graphics::sprite_t{ {
            .camera = camera,
            .position = position + fan::vec3(fan::vec2(j.position) * size, additional_depth + j.position.z),
            .size = j.size * size,
            .angle = j.angle,
            .color = j.color,
            .parallax_factor = 0,
            .flags = j.flags
        } };
        loco_t::texturepack_t::ti_t ti;
        if (texturepack->qti(j.image_name, &ti)) {
          fan::throw_error("failed to load image from .fte - corrupted save file");
        }
        std::get<loco_t::shape_t>(node.tiles[fan::vec3i(x, y, depth)]).load_tp(
          &ti
        );
        break;
      }
      case fte_t::mesh_property_t::light: {
        node.tiles[fan::vec3i(x, y, depth)] = fan::graphics::light_t{ {
          .camera = camera,
          .position = position + fan::vec3(fan::vec2(j.position) * size, additional_depth + j.position.z),
          .size = j.size * size,
          .color = j.color
        } };
        break;
      }
      case fte_t::mesh_property_t::collider: {
        node.tiles[fan::vec3i(x, y, depth)] =
          fan::graphics::collider_hidden_t(
            *(fan::vec2*)&position + fan::vec2(j.position) * size,
            j.size * size
          )
        ;
        break;
      }
      case fte_t::mesh_property_t::sensor: {
        userdata_t userdata;
        userdata.key = j.key;
        userdata.key_state = j.key_state;
        node.tiles[fan::vec3i(x, y, depth)] =
          fan::graphics::collider_sensor_t(
            *(fan::vec2*)&position + fan::vec2(j.position) * size,
            j.size * size,
            userdata
          )
        ;
        break;
      }
      default: {
        fan::throw_error("unimplemented switch");
      }
    }
    std::visit([&]<typename T>(T & v) {
      if constexpr (!std::is_same_v<fan::graphics::collider_sensor_t, T> &&
        !std::is_same_v<fan::graphics::collider_hidden_t, T>) {
       // fan::print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", v.get_parallax_factor());
        //v.set_camera(camera->camera);
      }
      //if constexpr (fan_has_function(T, set_camera (v))) {

      //}
    }, node.tiles[fan::vec3i(x, y, depth)]);

    auto found = id_callbacks.find(j.id);
    if (found != id_callbacks.end()) {
      found->second(node.tiles[fan::vec3i(x, y, depth)], j);
    }
  }

  void clear(node_t& node) {
    for (auto& i : node.tiles) {
      std::visit([]<typename T>(T & v) {
        if constexpr (fan::same_as_any<T,
          fan::graphics::collider_hidden_t,
          fan::graphics::collider_sensor_t>) {
          v.close();
        }
      }, i.second);
    }
    node.tiles.clear();
  }

  struct shape_depths_t {
    static constexpr int max_layer_depth = 0xFAAA - 2;
  };

  static constexpr int max_layer_depth = 128;

  void update(id_t id, const fan::vec2& position_) {
    auto& node = map_list[id];
    if (prev_render == (position_ / node.compiled_map->tile_size / 2).floor()) {
      return;
    }
    fan::vec2i old_render = prev_render;
    auto& map_tiles = node.compiled_map->compiled_shapes;

    prev_render = (position_ / node.compiled_map->tile_size / 2).floor();
    fan::vec2i offset = prev_render - old_render;

    if (offset.x > view_size.x || offset.y > view_size.y) {
      initialize(node, position_);
      return;
    }
    auto convert_to_grid = [&node, this] (fan::vec2i& src) {
      src -= view_size;
      src /= 2;
      /*src.x += .x / 2;
      src.y += view_size.y / 2;*/
      src = (src).floor();
    };

    fan::vec2i prev_src = old_render;
    convert_to_grid(prev_src);
    fan::vec2i src = (position_ / node.compiled_map->tile_size).floor();
    convert_to_grid(src);

    fan::print(src);

    fan::vec3i src_vec3 = prev_src;

    for (int off = 0; off < std::abs(offset.y); ++off) {
      for (int y = 0; y < view_size.x; ++y) {
        // HARDCODED
        for (int depth = 0; depth < 10; ++depth) {
          fan::vec3 erase_at = src_vec3 + fan::vec3i(
            y,
            (offset.y < 0 ? view_size.y - off - 1 : off),
            depth);
          std::visit([]<typename T>(T& v) {
            if constexpr (fan::same_as_any<T,
              fan::graphics::collider_hidden_t,
              fan::graphics::collider_sensor_t>) {
              fan::print("closing collider");
              v.close();
            }
          }, node.tiles[erase_at]);
          node.tiles.erase(erase_at);
        }
        fan::vec2i grid_pos = src;
        if (offset.y > 0) {
          grid_pos += fan::vec2i(y, view_size.y - 1 - off);
        }
        else {
          grid_pos += fan::vec2i(y, off);
        }
        if (!(grid_pos.y < (int64_t)map_tiles.size() && grid_pos.x < (int64_t)map_tiles[grid_pos.y].size())) {
          continue;
        }
        if (grid_pos.x < 0 || grid_pos.y < 0) {
          continue;
        }
        if (grid_pos.y >= (int64_t)map_tiles.size() || grid_pos.x >= (int64_t)map_tiles[grid_pos.y].size()) {
          continue;
        }
        if (map_tiles[grid_pos.y][grid_pos.x].empty()) {
          continue;
        }
        int depth = 0;
        for (auto& j : map_tiles[grid_pos.y][grid_pos.x]) {
          add_tile(node, j, grid_pos.x, grid_pos.y, depth++);
        }
      }
    }
    for (int off = 0; off < std::abs(offset.x); ++off) {
      for (int x = 0; x < view_size.y; ++x) {
        for (int depth = 0; depth < 10; ++depth) {
          fan::vec3 erase_at = src_vec3 + fan::vec3i(
            (offset.x < 0 ? view_size.x - off - 1 : off),
            x,
            depth);
          std::visit([]<typename T>(T & v) {
            if constexpr (fan::same_as_any<T,
              fan::graphics::collider_hidden_t,
              fan::graphics::collider_sensor_t>) {
              fan::print("closing collider");
              v.close();
            }
          }, node.tiles[erase_at]);
          node.tiles.erase(erase_at);
        }
        fan::vec2i grid_pos = src;
        if (offset.x > 0) {
          grid_pos += fan::vec2i(view_size.x - 1 - off, x);
        }
        else {
          grid_pos += fan::vec2i(off, x);
        }
        if (!(grid_pos.y < (int64_t)map_tiles.size() && grid_pos.x < (int64_t)map_tiles[grid_pos.y].size())) {
          continue;
        }
        if (grid_pos.x < 0 || grid_pos.y < 0) {
          continue;
        }
        if (grid_pos.y >= (int64_t)map_tiles.size() || grid_pos.x >= (int64_t)map_tiles[grid_pos.y].size()) {
          continue;
        }
        if (map_tiles[grid_pos.y][grid_pos.x].empty()) {
          continue;
        }
        int depth = 0;
        for (auto& j : map_tiles[grid_pos.y][grid_pos.x]) {
          add_tile(node, j, grid_pos.x, grid_pos.y, depth++);
        }
      }
    }
  }

private:
 // fte_loader_t::add;
};

#undef tilemap_renderer