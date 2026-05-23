module;

#if defined(FAN_2D)
#endif
#define MEASURE_LOAD_TIMES 0
#if MEASURE_LOAD_TIMES
#define TIMER_START(name) fan::time::timer timer_##name{true}; fan::print_impl(#name " start")
#define TIMER_END(name) fan::print_impl(#name " end", timer_##name.millis())
#else
#define TIMER_START(name)
#define TIMER_END(name)
#endif

module fan.graphics.tilemap_editor.renderer;

import std;

#if defined(FAN_2D)
#if defined(FAN_PHYSICS_2D)

import fan.io.file;

namespace fan::graphics {
  tilemap_renderer_t::id_t tilemap_renderer_t::open_map(compiled_map_t& out_compiled, const char* path, const properties_t& p, const std::source_location& callers_path) {
    out_compiled = *compile(path, path, callers_path);
    return add(&out_compiled, p);
  }

  tilemap_renderer_t::id_t tilemap_renderer_t::open_map(std::string_view file_name, const properties_t& p, const std::source_location& callers_path) {
    std::string smap_name(file_name);
    compiled_map_t* compiled = compile(fan::io::file::strip_extension(smap_name), smap_name, callers_path);
    return add(compiled, p);
  }

  tilemap_renderer_t::id_t tilemap_renderer_t::open_map(std::string_view map_name, std::string_view file_name, const properties_t& p, const std::source_location& callers_path) {
    std::string smap_name(file_name);
    compiled_map_t* compiled = compile(std::string(map_name), smap_name, callers_path);
    return add(compiled, p);
  }

  void tilemap_renderer_t::close_map(id_t& id) {
    clear(id);
  }

  tilemap_renderer_t::id_t tilemap_renderer_t::add(compiled_map_t* compiled_map) {
    return add(compiled_map, properties_t());
  }

  tilemap_renderer_t::id_t tilemap_renderer_t::add(compiled_map_t* compiled_map, const properties_t& p) {
    if (p.render_view == nullptr) {
      render_view = &fan::graphics::get_orthographic_render_view();
    }
    else {
      render_view = p.render_view;
    }

    auto it = new_map_node();
    auto& node = get_map_node(it);

    clear(node);
    node.compiled_map = compiled_map;
    node.depth_fn = p.depth_fn;

    view_size = p.size * 2;
    view_size = view_size.clamp(fan::vec2i(0), node.compiled_map->map_size * 4);
    node.prev_render = convert_to_grid(p.position, node);
    node.size = p.scale;

    node.position = p.offset;
    initialize(node, p.position);

    return it;
  }

  void tilemap_renderer_t::initialize(node_t& node, const fan::vec2& position) {
    TIMER_START(total_initialize);

    node.physics_entities.clear();
    node.lights.clear();

    auto& compiled_map = *node.compiled_map;
    std::size_t light_count = 0;
    std::size_t physics_count = compiled_map.physics_shapes.size();

    TIMER_START(light_count_scan);
    for (int y = 0; y < compiled_map.map_size.y; ++y) {
      for (int x = 0; x < compiled_map.map_size.x; ++x) {
        for (auto& j : compiled_map.compiled_shapes[y][x]) {
          if (j.mesh_property == fte_t::mesh_property_t::light) {
            ++light_count;
          }
        }
      }
    }
    TIMER_END(light_count_scan);

    node.lights.reserve(light_count);
    node.physics_entities.reserve(physics_count);

    TIMER_START(initialize_visual_block);
    initialize_visual(node, position);
    TIMER_END(initialize_visual_block);

    TIMER_START(add_lights_block);
    for (int y = 0; y < compiled_map.map_size.y; ++y) {
      for (int x = 0; x < compiled_map.map_size.x; ++x) {
        for (auto& j : compiled_map.compiled_shapes[y][x]) {
          if (j.mesh_property != fte_t::mesh_property_t::light) continue;

          light_with_id_t light;
          light.id = j.id;
          light.shape = fan::graphics::light_t {{
              .render_view = render_view,
              .position = node.position + fan::vec3(fan::vec2(j.position) * node.size, y + compiled_map.tile_size.y / 2 + j.position.z),
              .size = j.size * node.size,
              .color = j.color,
              .flags = j.flags
          }};
          node.lights.emplace_back(std::move(light));
        }
      }
    }
    TIMER_END(add_lights_block);

    TIMER_START(add_physics_block);
    for (compiled_map_t::physics_data_t& pd : compiled_map.physics_shapes) {
      switch (pd.physics_shapes.type) {
      case fte_t::physics_shapes_t::type_e::box:
      {
        node.physics_entities.push_back({
            .visual = fan::graphics::physics::rectangle_t{{
                .render_view = render_view,
                .position = node.position + pd.position * node.size,
                .size = pd.size * node.size,
                .color = pd.physics_shapes.draw ? fan::color::from_rgba(0x6e8d6eff) : fan::colors::transparent,
                .outline_color = (pd.physics_shapes.draw ? fan::color::from_rgba(0x6e8d6eff) : fan::colors::transparent) * 2,
                .blending = true,
                .body_type = pd.physics_shapes.body_type,
                .shape_properties = pd.physics_shapes.shape_properties
            }}
          });
        break;
      }
      case fte_t::physics_shapes_t::type_e::circle:
      {
        node.physics_entities.push_back({
            .visual = fan::graphics::physics::circle_t{{
                .render_view = render_view,
                .position = node.position + pd.position * node.size,
                .radius = (pd.size.max() * node.size.x == compiled_map.tile_size.x ? pd.size.y * node.size.y : pd.size.x * node.size.x),
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
    TIMER_END(add_physics_block);

    TIMER_END(total_initialize);
  }


  void tilemap_renderer_t::initialize_visual(node_t& node, const fan::vec2& position) {
    TIMER_START(total_initialize_visual);

    clear_visual(node);

    fan::vec2i src = convert_to_grid(position, node);
    adjust_view(src);

    auto& map_tiles = node.compiled_map->compiled_shapes;

    std::size_t tile_estimate = 0;
    TIMER_START(tile_estimate_scan);
    for (int y = 0; y < view_size.y; ++y) {
      for (int x = 0; x < view_size.x; ++x) {
        fan::vec2i grid_pos = src + fan::vec2i(x, y);
        if (grid_pos.x < 0 || grid_pos.y < 0) continue;
        if (grid_pos.y >= (std::int64_t)map_tiles.size() ||
          grid_pos.x >= (std::int64_t)map_tiles[grid_pos.y].size()) continue;
        tile_estimate += map_tiles[grid_pos.y][grid_pos.x].size();
      }
    }
    TIMER_END(tile_estimate_scan);

    node.rendered_tiles.reserve(tile_estimate);

    TIMER_START(tile_add_loop);
    for (int y = 0; y < view_size.y; ++y) {
      for (int x = 0; x < view_size.x; ++x) {
        fan::vec2i grid_pos = src + fan::vec2i(x, y);
        if (grid_pos.x < 0 || grid_pos.y < 0) continue;
        if (grid_pos.y >= (std::int64_t)map_tiles.size() ||
          grid_pos.x >= (std::int64_t)map_tiles[grid_pos.y].size()) continue;
        if (map_tiles[grid_pos.y][grid_pos.x].empty()) continue;

        int depth = 0;
        for (auto& tile_data : map_tiles[grid_pos.y][grid_pos.x]) {
          add_tile(node, tile_data, grid_pos.x, grid_pos.y, depth++);
        }
      }
    }
    TIMER_END(tile_add_loop);

    TIMER_END(total_initialize_visual);
  }

  void tilemap_renderer_t::erase_visual(id_t map_id, const std::string& id) {
    auto& node = get_map_node(map_id);
    auto& compiled_map = *node.compiled_map;

    auto lookup_it = compiled_map.id_lookup.find(id);
    if (lookup_it == compiled_map.id_lookup.end() || lookup_it->second.empty()) {
      return;
    }

    fte_t::tile_t* tile_ptr = lookup_it->second[0];
    fan::vec3 pos = tile_ptr->position;

    fan::vec2i gp;
    gp.x = pos.x / (compiled_map.tile_size.x * 2);
    gp.y = pos.y / (compiled_map.tile_size.y * 2);

    if (gp.x < 0 || gp.y < 0 ||
      gp.y >= compiled_map.map_size.y ||
      gp.x >= compiled_map.map_size.x) {
      return;
    }

    auto& cell = compiled_map.compiled_shapes[gp.y][gp.x];

    cell.erase(
      std::remove_if(cell.begin(), cell.end(), [&](const fte_t::tile_t& t) {
      return t.id == id;
    }),
      cell.end()
    );

    lookup_it->second.erase(
      std::remove_if(lookup_it->second.begin(), lookup_it->second.end(), [&](fte_t::tile_t* t) {
      return t->id == id;
    }),
      lookup_it->second.end()
    );

    node.prev_render = 100000;
    initialize(node, node.position);
  }

  fan::physics::entity_t tilemap_renderer_t::add_sensor_rectangle(id_t map_id, const std::string& id) {
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

  fan::physics::entity_t tilemap_renderer_t::add_sensor_circle(id_t map_id, const std::string& id) {
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

  f32_t tilemap_renderer_t::get_depth_from_y(id_t map_id, const fan::vec2& position) {
    auto& node = get_map_node(map_id);
    return fan::graphics::get_depth_from_y(position, node.compiled_map->tile_size.y * 2.f);
  }

  void tilemap_renderer_t::add_tile(node_t& node, fte_t::tile_t& j, int x, int y, std::uint32_t depth) {
    {
      if (fan::get_hash(j.id.c_str()) == fan::get_hash("##tile_world_dirt")) {
        fan::physics::gphysics()->create_box(node.position + fan::vec2(j.position) * node.size, j.size * node.size);
      }
    }

    fan::vec3i tile_key(x, y, depth);
    if (node.rendered_tiles.count(tile_key)) {
      fan::print_impl("WARNING: Tile already exists at", x, y, depth);
      fan::print_impl("  Old image:", node.rendered_tiles[tile_key].get_image().NRI);
      fan::print_impl("  New would be:", j.texture_pack_unique_id.id);
    }
    int additional_depth = 0;
  
    fan::vec2 world_pos = node.position.xy() + fan::vec2(j.position) * node.size;
    fan::vec2 world_size = j.size * node.size;

    f32_t z = j.position.z;
    if (node.depth_fn) {
      z = node.depth_fn(j, world_pos, world_size, node.compiled_map->tile_size.y * 2.f);
    }

    switch (j.mesh_property) {
    case fte_t::mesh_property_t::none:
    {
      node.rendered_tiles[tile_key] = fan::graphics::sprite_t {{
          .render_view = render_view,
          .position = node.position + fan::vec3(fan::vec2(j.position) * node.size, additional_depth + z),
          .size = j.size * node.size,
          .angle = j.angle,
          .color = j.color,
          .parallax_factor = 0,
          .blending = true,
          .flags = j.flags,
          .texture_pack_unique_id = j.texture_pack_unique_id
        }};
      switch (fan::get_hash(j.id.c_str())) {
      case fan::get_hash("##tile_world_background"):
      {
        node.rendered_tiles[tile_key].set_image(fan::graphics::tile_world_images.background);
        break;
      }
      case fan::get_hash("##tile_world_dirt"):
      {
        node.rendered_tiles[tile_key].set_image(fan::graphics::tile_world_images.dirt);
        break;
      }
      }
      break;
    }
    default:
    {
      return;
    }
    }
    node.rendered_tiles[tile_key].set_static();
    if (!j.id.empty()) {
      node.rendered_tiles[tile_key].id = j.id;
      node.id_to_shape[j.id] = &node.rendered_tiles[tile_key];
    }

    auto found = id_callbacks.find(j.id);
    if (found != id_callbacks.end()) {
      found->second(node.rendered_tiles[tile_key], j);
    }
  }

  void tilemap_renderer_t::clear_visual(node_t& node) {
    node.rendered_tiles.clear();
    node.id_to_shape.clear();
  }

  void tilemap_renderer_t::clear(id_t& id) {
    clear(get_map_node(id));
  }

  void tilemap_renderer_t::clear(node_t& node) {
    clear_visual(node);

    for (auto& e : node.physics_entities) {
      std::visit([](auto& v) {
        v.destroy();
      }, e.visual);
    }
    node.physics_entities.clear();
    node.lights.clear();
  }

  void tilemap_renderer_t::erase(id_t& id) {
    clear(id);
    delete_map_node(id);
    invalidate_map_node(id);
  }

  void tilemap_renderer_t::remove_visual(id_t id, const std::string& str_id, const fan::vec2& position) {
    auto& node = get_map_node(id);
    auto& compiled_map = *node.compiled_map;

    fan::vec2i gp;
    gp.x = position.x / (compiled_map.tile_size.x * 2);
    gp.y = position.y / (compiled_map.tile_size.y * 2);

    if (gp.x < 0 || gp.y < 0 ||
      gp.y >= compiled_map.map_size.y ||
      gp.x >= compiled_map.map_size.x) {
      return;
    }

    auto& cell = compiled_map.compiled_shapes[gp.y][gp.x];
    std::string removed_id;
    bool found_tile = false;

    cell.erase(
      std::remove_if(cell.begin(), cell.end(), [&](const fte_t::tile_t& t) {
      if (t.position == position && str_id == t.id) {
        removed_id = t.id;
        found_tile = true;
        return true;
      }
      return false;
    }),
      cell.end()
    );

    if (!found_tile) {
      return;
    }

    if (!removed_id.empty()) {
      auto lookup_it = compiled_map.id_lookup.find(removed_id);
      if (lookup_it != compiled_map.id_lookup.end()) {
        lookup_it->second.erase(
          std::remove_if(lookup_it->second.begin(), lookup_it->second.end(), [&](fte_t::tile_t* t) {
          return t->position == position && t->id == str_id;
        }),
          lookup_it->second.end()
        );
      }
    }

    auto rendered_it = node.rendered_tiles.begin();
    while (rendered_it != node.rendered_tiles.end()) {
      auto& shape = rendered_it->second;

      if (shape.id == removed_id) {
        fan::vec3 shape_pos = shape.get_position();

        if (std::abs(shape_pos.x - position.x) < 1.0f &&
          std::abs(shape_pos.y - position.y) < 1.0f) {

          if (!shape.id.empty()) {
            node.id_to_shape.erase(shape.id);
          }

          rendered_it = node.rendered_tiles.erase(rendered_it);
          return;
        }
      }
      ++rendered_it;
    }
  }

  void tilemap_renderer_t::adjust_view(fan::vec2i& src) {
    src /= 2;
    src.x -= view_size.x / 2;
    src.y -= view_size.y / 2;
    src += 1;
  };

  void tilemap_renderer_t::update(id_t id, const fan::vec2& position_) {
    if (is_map_node_invalid(id)) {
      return;
    }
    auto& node = get_map_node(id);
    fan::vec2i new_grid_pos = convert_to_grid(position_, node);

    if (node.prev_render == new_grid_pos) {
      return;
    }

    fan::vec2i offset = new_grid_pos - node.prev_render;
    auto& map_tiles = node.compiled_map->compiled_shapes;

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

  void tilemap_renderer_t::erase_physics_entity(id_t map_id, const std::string& id) {
    if (is_map_node_invalid(map_id)) {
      return;
    }

    auto& node = get_map_node(map_id);

    for (auto it = node.physics_entities.begin(); it != node.physics_entities.end();) {
      if (it->id == id) {
        std::visit([&](auto& v) {
          v.destroy();
        }, it->visual);
        it = node.physics_entities.erase(it);
      }
      else {
        ++it;
      }
    }
  }

  tilemap_renderer_t::tile_draw_data_t* tilemap_renderer_t::get_shape_by_id(id_t map_id, const std::string& id) {
    auto& node = get_map_node(map_id);
    auto it = node.id_to_shape.find(id);
    return (it != node.id_to_shape.end()) ? it->second : nullptr;
  }

  fan::graphics::shape_t* tilemap_renderer_t::get_light_by_id(tilemap_renderer_t::id_t map_id, const std::string& id) {
    auto& node = get_map_node(map_id);
    for (auto& light : node.lights) {
      if (light.id == id) {
        return &light.shape;
      }
    }
    return nullptr;
  }

  f32_t tilemap_renderer_t::get_dynamic_depth(id_t map_id, const fan::vec2& position, f32_t body_height) {
    auto& node = get_map_node(map_id);
    f32_t tile_size_y = node.compiled_map->tile_size.y * 2.f;
    return fan::graphics::get_player_depth_from_y(position, body_height, tile_size_y);
  }


  tilemap_instance_t::tilemap_instance_t(
    tilemap_renderer_t& r, std::string_view path,
    const tilemap_renderer_t::properties_t& p,
    const std::source_location& loc) : renderer(&r), id(r.open_map(path, p, loc)) {
    if (p.build_collisions) {
      build_collisions(p.collision_body_type, p.collision_props);
    }
  }

  tilemap_instance_t::tilemap_instance_t(tilemap_instance_t&& o) noexcept
    : renderer(o.renderer), id(o.id), collisions(std::move(o.collisions)), collision_datas(std::move(o.collision_datas)) {
    o.renderer = nullptr;
  }

  tilemap_instance_t& tilemap_instance_t::operator=(tilemap_instance_t&& o) noexcept {
    if (this != &o) {
      close();
      renderer = o.renderer;
      id = o.id;
      collisions = std::move(o.collisions);
      collision_datas = std::move(o.collision_datas);
      o.renderer = nullptr;
    }
    return *this;
  }

  tilemap_instance_t::~tilemap_instance_t() { close(); }

  void tilemap_instance_t::update(const fan::vec2& pos) {
    if (renderer) renderer->update(id, pos);
  }

  void tilemap_instance_t::close() {
    for (auto& c : collisions) { c.destroy(); }
    collisions.clear();
    collision_datas.clear();
    if (renderer) { renderer->close_map(id); renderer = nullptr; }
  }

  void tilemap_instance_t::build_collisions(std::uint8_t bt, fan::physics::shape_properties_t props) {
    props.contact_events = true;
    renderer->iterate_tiles(id, [&](const auto& tile) {
      auto body = fan::physics::gphysics()->create_box(tile.position, tile.size, 0, bt, props);
      collision_datas.push_back({
        .cell = fan::vec2i(tile.position.x, tile.position.y),
        .type = tile.mesh_property,
        .id = tile.id
      });
      body.set_user_data(&collision_datas.back());
      collisions.emplace_back(body);
    });
  }
}
#endif
#endif