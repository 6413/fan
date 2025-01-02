#define tilemap_renderer 0
#include "loader.h"


struct fte_renderer_t : fte_loader_t {

  std::unordered_map<std::string, fan::function_t<void(tile_draw_data_t&, fte_t::tile_t&)>> id_callbacks;
  std::unordered_map<std::string, fan::function_t<void(map_list_data_t::physics_entities_t&, compiled_map_t::physics_data_t&)>> sensor_id_callbacks;

  fan::vec2i view_size = 1;
  fan::graphics::camera_t* camera = nullptr;

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
   
    view_size = p.size * 2;
    node.prev_render = convert_to_grid(p.position, node);
    node.size = p.scale;

    node.position = p.offset;
    initialize(node, p.position);


    return it;
  }

  void initialize(id_t& node_id, const fan::vec2& position) {
    initialize(map_list[node_id], position);
  }
  void initialize(node_t& node, const fan::vec2& position) {

    clear(node);

    fan::vec2i src = convert_to_grid(position, node);
    adjust_view(src);

    auto& map_tiles = node.compiled_map->compiled_shapes;

    for (int y = 0; y < view_size.y; ++y) {
      for (int x = 0; x < view_size.x; ++x) {
        fan::vec2i grid_pos = src + fan::vec2i(x, y);
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
    for (compiled_map_t::physics_data_t& pd : node.compiled_map->physics_shapes) {
      switch (pd.physics_shapes.type) {
      case fte_t::physics_shapes_t::type_e::box: {
        node.physics_entities.push_back({
          .visual = fan::graphics::physics_shapes::rectangle_t{{
              .camera = camera,
              .position = node.position + pd.position * node.size,
              .size = pd.size * node.size,
              .color = pd.physics_shapes.draw ? fan::color::hex(0x6e8d6eff) : fan::colors::transparent,
              .outline_color = (pd.physics_shapes.draw ? fan::color::hex(0x6e8d6eff) : fan::colors::transparent) * 2,
              .blending = true,
              .body_type = pd.physics_shapes.body_type,
              .shape_properties = pd.physics_shapes.shape_properties,
            }}
        });
        //pd.physics_shapes.
        //std::get<fan::graphics::physics_shapes::rectangle_t>(node.physics_entities.back().visual).body_id
        break;
      }
      case fte_t::physics_shapes_t::type_e::circle: {
        node.physics_entities.push_back({
          .visual = fan::graphics::physics_shapes::circle_t{{
              .camera = camera,
              .position = node.position + pd.position * node.size,
              .radius = pd.size.max() * node.size.x == 1 ? node.size.y : node.size.x,
              .color = pd.physics_shapes.draw ? fan::color::hex(0x6e8d6eff) : fan::colors::transparent,
              .blending = true,
              .body_type = pd.physics_shapes.body_type,
              .shape_properties = pd.physics_shapes.shape_properties
            }}
        });
        break;
      }
      }
      auto found = sensor_id_callbacks.find(pd.physics_shapes.id);
      if (found != sensor_id_callbacks.end()) {
        found->second(node.physics_entities.back(), pd);
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
            .position = node.position + fan::vec3(fan::vec2(j.position) * node.size, additional_depth + j.position.z),
            .size = j.size * node.size,
            .angle = j.angle,
            .color = j.color,
            .parallax_factor = 0,
            .flags = j.flags
        } };
        loco_t::texturepack_t::ti_t ti;
        if (texturepack->qti(j.image_name, &ti)) {
          fan::throw_error("failed to load image from .fte - corrupted save file");
        }
        node.tiles[fan::vec3i(x, y, depth)].load_tp(
          &ti
        );
        break;
      }
      case fte_t::mesh_property_t::light: {
        node.tiles[fan::vec3i(x, y, depth)] = fan::graphics::light_t{ {
          .camera = camera,
          .position = node.position + fan::vec3(fan::vec2(j.position) * node.size, additional_depth + j.position.z),
          .size = j.size * node.size,
          .color = j.color
        } };
        break;
      }
      default: {
        fan::throw_error("unimplemented switch");
      }
    }

    auto found = id_callbacks.find(j.id);
    if (found != id_callbacks.end()) {
      found->second(node.tiles[fan::vec3i(x, y, depth)], j);
    }
  }

  void clear(id_t& id) {
    clear(map_list[id]);
  }
  void clear(node_t& node) {
    node.tiles.clear();
    for (auto& j : node.physics_entities) {
      std::visit([](auto& obj){obj.body_id.destroy();}, j.visual);
    }
    node.physics_entities.clear();
  }

  void erase(id_t& id) {
    clear(id);
    map_list.unlrec(id);
  }

  struct shape_depths_t {
    static constexpr int max_layer_depth = 0xFAAA - 2;
  };

  static constexpr int max_layer_depth = 128;

  fan::vec2 convert_to_grid(const fan::vec2& p, const node_t& node) {
    return ((p / node.size) / (node.compiled_map->map_size)).floor();
  }
  void adjust_view(fan::vec2i& src) {
    src /= 2;
    src.x -= view_size.x / 2;
    src.y -= view_size.y / 2;
    src += 1;
  };

  void update(id_t id, const fan::vec2& position_) {
    auto& node = map_list[id];
    if (node.prev_render == convert_to_grid(position_, node)) {
      return;
    }
    fan::vec2i old_render = node.prev_render;
    auto& map_tiles = node.compiled_map->compiled_shapes;

    node.prev_render = convert_to_grid(position_, node);
    fan::vec2i offset = node.prev_render - old_render;

    if (offset.x > view_size.x || offset.y > view_size.y) {
      initialize(node, position_);
      return;
    }

    fan::vec2i prev_src = old_render;
    adjust_view(prev_src);
    fan::vec2i src = convert_to_grid(position_, node);
    adjust_view(src);

    fan::vec3i src_vec3 = prev_src;

    for (int off = 0; off < std::abs(offset.y); ++off) {
      for (int y = 0; y < view_size.x; ++y) {
        // HARDCODED
        for (int depth = 0; depth < 10; ++depth) {
          fan::vec3 erase_at = src_vec3 + fan::vec3i(
            y,
            (offset.y < 0 ? view_size.y - off - 1 : off),
            depth);
          node.tiles.erase(erase_at);
        }
        fan::vec2i grid_pos = src;
        if (offset.y > 0) {
          grid_pos += fan::vec2i(y, view_size.y - 1 - off);
        }
        else {
          grid_pos += fan::vec2i(y, off);
        }
        if (grid_pos.x < 0 || grid_pos.y < 0) {
          continue;
        }
        if (grid_pos.y >= (int64_t)map_tiles.size() || grid_pos.x >= (int64_t)map_tiles[grid_pos.y].size()) {
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
          node.tiles.erase(erase_at);
        }
        fan::vec2i grid_pos = src;
        if (offset.x > 0) {
          grid_pos += fan::vec2i(view_size.x - 1 - off, x);
        }
        else {
          grid_pos += fan::vec2i(off, x);
        }
        if (grid_pos.x < 0 || grid_pos.y < 0) {
          continue;
        }
        if (grid_pos.y >= (int64_t)map_tiles.size() || grid_pos.x >= (int64_t)map_tiles[grid_pos.y].size()) {
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