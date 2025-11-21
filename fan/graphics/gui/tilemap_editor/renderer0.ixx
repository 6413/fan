module;

export module fan.graphics.gui.tilemap_editor.renderer;

#if defined(fan_physics) && defined(fan_gui)

export import fan.graphics.gui.tilemap_editor.loader;

import std;

import fan.utility;
import fan.print;
import fan.graphics;
import fan.graphics.physics_shapes;
import fan.physics.b2_integration;
//import fan;

export struct fte_renderer_t : fte_loader_t {

  std::unordered_map<std::string, std::function<void(tile_draw_data_t&, fte_t::tile_t&)>> id_callbacks;
  std::unordered_map<std::string, std::function<void(map_list_data_t::physics_entities_t&, compiled_map_t::physics_data_t&)>> sensor_id_callbacks;

  fan::vec2i view_size = 1;
  fan::graphics::render_view_t* render_view = nullptr;

  id_t add(compiled_map_t* compiled_map) {
    return add(compiled_map, properties_t());
  }

  id_t add(compiled_map_t* compiled_map, const properties_t& p) {
    if (p.render_view == nullptr) {
      render_view = &fan::graphics::get_orthographic_render_view();
    }
    else {
      render_view = p.render_view;
    }

    auto it = map_list.NewNodeLast();
    auto& node = map_list[it];
    clear(node);
    node.compiled_map = compiled_map;
   
    view_size = p.size * 2;
    node.prev_render = convert_to_grid(p.position, node);
    node.size = p.scale;

    node.position = p.offset;
    initialize(node, p.position);

    return it;
  }

  //void initialize(id_t& node_id, const fan::vec2& position) {
  //  initialize(map_list[node_id], position);
  //}
  void initialize(node_t& node, const fan::vec2& position) {
    initialize_visual(node, position);

    for (compiled_map_t::physics_data_t& pd : node.compiled_map->physics_shapes) {
      switch (pd.physics_shapes.type) {
      case fte_t::physics_shapes_t::type_e::box: {
        node.physics_entities.push_back({
          .visual = fan::graphics::physics::rectangle_t{{
              .render_view = render_view,
              .position = node.position + pd.position * node.size,
              .size = pd.size * node.size,
              .color = pd.physics_shapes.draw ? fan::color::from_rgba(0x6e8d6eff) : fan::colors::transparent,
              .outline_color = (pd.physics_shapes.draw ? fan::color::from_rgba(0x6e8d6eff) : fan::colors::transparent) * 2,
              .blending = true,
              .body_type = pd.physics_shapes.body_type,
              .shape_properties = pd.physics_shapes.shape_properties,
            }}
        });
        //pd.physics_shapes.
        //std::get<fan::graphics::physics::rectangle_t>(node.physics_entities.back().visual).body_id
        break;
      }
      case fte_t::physics_shapes_t::type_e::circle: {
        node.physics_entities.push_back({
          .visual = fan::graphics::physics::circle_t{{
              .render_view = render_view,
              .position = node.position + pd.position * node.size,
              .radius = (pd.size.max() * node.size.x == node.compiled_map->tile_size.x ? pd.size.y * node.size.y : pd.size.x  *  node.size.x),
              .color = pd.physics_shapes.draw ? fan::color::from_rgba(0x6e8d6eff) : fan::colors::transparent,
              .blending = true,
              .body_type = pd.physics_shapes.body_type,
              .shape_properties = pd.physics_shapes.shape_properties
            }}
        });
        break;
      }
      }
      node.physics_entities.back().id = pd.physics_shapes.id;
      auto found = sensor_id_callbacks.find(pd.physics_shapes.id);
      if (found != sensor_id_callbacks.end()) {
        found->second(node.physics_entities.back(), pd);
      }
    }
  }

  void initialize_visual(node_t& node, const fan::vec2& position) {
    clear_visual(node);

    fan::vec2i src = convert_to_grid(position, node);
    adjust_view(src);

    auto& map_tiles = node.compiled_map->compiled_shapes;

    for (int y = 0; y < view_size.y; ++y) {
      for (int x = 0; x < view_size.x; ++x) {
        fan::vec2i grid_pos = src + fan::vec2i(x, y);
        if (grid_pos.x < 0 || grid_pos.y < 0) {
          continue;
        }
        if (grid_pos.y >= (std::int64_t)map_tiles.size() || grid_pos.x >= (std::int64_t)map_tiles[grid_pos.y].size()) {
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

  void erase_id(id_t map_id, const std::string& id) {
    auto& node = map_list[map_id];
    auto& shapes = node.compiled_map->compiled_shapes;
    for (auto& i : shapes) {
      for (auto& j : i) {
        for (int k = 0; k < j.size(); ++k) {
          if (id == j[k].id) {
            j.erase(j.begin() + k);
            tile_t t;
            auto f = get_body(map_id, id, t);
            node.prev_render = 100000;
            initialize(node, node.position);
            return;
          }
        }
      }
    }
  }

  fan::physics::entity_t add_sensor_rectangle(id_t map_id, const std::string& id) {
    fan::physics::entity_t sensor_id;
    tile_t tile;
    if (get_body(map_id, id, tile)) {
      sensor_id = fan::physics::create_sensor_rectangle(
        tile.position,
        tile.size / 2.f
      );
    }
    return sensor_id;
  }
  fan::physics::entity_t add_sensor_circle(id_t map_id, const std::string& id) {
    fan::physics::entity_t sensor_id;
    tile_t tile;
    if (get_body(map_id, id, tile)) {
      sensor_id = fan::physics::create_sensor_circle(
        tile.position,
        tile.size.x / 2.f
      );
    }
    return sensor_id;
  }
  f32_t get_depth_from_y(id_t map_id, const fan::vec2& position) {
    auto& node = map_list[map_id];
    return fan::graphics::get_depth_from_y(position, node.compiled_map->tile_size.y * 2.f);
  }

  struct userdata_t {
    int key;
    int key_state;
  };

  void add_tile(node_t& node, fte_t::tile_t& j, int x, int y, std::uint32_t depth) {

    // temporary
    {
      if (fan::get_hash(j.id.c_str()) == fan::get_hash("##tile_world_dirt")) {
        fan::physics::gphysics->create_box(node.position + fan::vec2(j.position) * node.size, j.size * node.size);
      }
    }

    fan::vec3i tile_key(x, y, depth);
    int additional_depth = y + node.compiled_map->tile_size.y / 2;
    switch (j.mesh_property) {
      case fte_t::mesh_property_t::none: {
        node.rendered_tiles[tile_key] = fan::graphics::sprite_t{ {
          .render_view = render_view,
          .position = node.position + fan::vec3(fan::vec2(j.position) * node.size, additional_depth + j.position.z),
          .size = j.size * node.size,
          .angle = j.angle,
          .color = j.color,
          .parallax_factor = 0,
          .blending = true,
          .flags = j.flags,
          .texture_pack_unique_id = j.texture_pack_unique_id
        } };
        switch (fan::get_hash(j.id.c_str())) {
        case fan::get_hash("##tile_world_background"): {
          node.rendered_tiles[tile_key].set_image(fan::graphics::tile_world_images::background);
          break;
        }
        case fan::get_hash("##tile_world_dirt"): {
          node.rendered_tiles[tile_key].set_image(fan::graphics::tile_world_images::dirt);
          break;
        }
        }
        break;
      }
      case fte_t::mesh_property_t::light: {
        node.rendered_tiles[tile_key] = fan::graphics::light_t{ {
          .render_view = render_view,
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
    if (!j.id.empty()) {
      node.rendered_tiles[tile_key].id = j.id;
      node.id_to_shape[j.id] = &node.rendered_tiles[tile_key];
    }

    auto found = id_callbacks.find(j.id);
    if (found != id_callbacks.end()) {
      found->second(node.rendered_tiles[tile_key], j);
    }
  }

  void clear_visual(node_t& node) {
    node.rendered_tiles.clear();
    node.id_to_shape.clear();
  }
  void clear(id_t& id) {
    clear(map_list[id]);
  }
  void clear(node_t& node) {
    clear_visual(node);
    node.physics_entities.clear();
  }

  void erase(id_t& id) {
    clear(id);
    map_list.unlrec(id);
    id.sic();
  }

  struct shape_depths_t {
    static constexpr int max_layer_depth = 0xFAAA - 2;
  };

  static constexpr int max_layer_depth = 128;

  void adjust_view(fan::vec2i& src) {
    src /= 2;
    src.x -= view_size.x / 2;
    src.y -= view_size.y / 2;
    src += 1;
  };

  void update(id_t id, const fan::vec2& position_) {
    if (id.iic()) {
      return;
    }
    auto& node = map_list[id];
    fan::vec2i new_grid_pos = convert_to_grid(position_, node);

    if (node.prev_render == new_grid_pos) {
      return;
    }

    fan::vec2i offset = new_grid_pos - node.prev_render;
    auto& map_tiles = node.compiled_map->compiled_shapes;

    // reinitialize big movement like teleport or big delta
    if (std::abs(offset.x) >= view_size.x / 2 || std::abs(offset.y) >= view_size.y / 2) {
      initialize_visual(node, position_);
      node.prev_render = new_grid_pos;
      return;
    }

    fan::vec2i old_src = node.prev_render;
    adjust_view(old_src);
    fan::vec2i new_src = new_grid_pos;
    adjust_view(new_src);

    node.prev_render = new_grid_pos;

    fan::vec2i old_min = old_src;
    fan::vec2i old_max = old_src + view_size;
    fan::vec2i new_min = new_src;
    fan::vec2i new_max = new_src + view_size;

    auto it = node.rendered_tiles.begin();
    while (it != node.rendered_tiles.end()) {
      fan::vec3i tile_pos = it->first;

      if (tile_pos.x < new_min.x || tile_pos.x >= new_max.x ||
        tile_pos.y < new_min.y || tile_pos.y >= new_max.y) {
        if (!it->second.id.empty()) {
          node.id_to_shape.erase(it->second.id);
        }
        it = node.rendered_tiles.erase(it);
      }
      else {
        ++it;
      }
    }

    for (int y = new_min.y; y < new_max.y; ++y) {
      for (int x = new_min.x; x < new_max.x; ++x) {
        if (x >= old_min.x && x < old_max.x && y >= old_min.y && y < old_max.y) {
          continue;
        }

        // Check bounds
        if (x < 0 || y < 0) {
          continue;
        }
        if (y >= (std::int64_t)map_tiles.size() || x >= (std::int64_t)map_tiles[y].size()) {
          continue;
        }
        if (map_tiles[y][x].empty()) {
          continue;
        }

        int depth = 0;
        for (auto& tile_data : map_tiles[y][x]) {
          add_tile(node, tile_data, x, y, depth++);
        }
      }
    }
  }

  // dont hold the pointer
  tile_draw_data_t* get_shape_by_id(id_t map_id, const std::string& id) {
    auto& node = map_list[map_id];
    auto it = node.id_to_shape.find(id);
    return (it != node.id_to_shape.end()) ? it->second : nullptr;
  }

private:
 // fte_loader_t::add;
};

#undef tilemap_renderer
#endif