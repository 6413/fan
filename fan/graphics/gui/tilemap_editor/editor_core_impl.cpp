module;

#if defined (FAN_WINDOW)

#include <cstddef>

#if defined(FAN_2D)
  #include <fan/utility.h>
#endif

#endif

module fan.graphics.gui.tilemap_editor.core;

#if defined (FAN_WINDOW)

#if defined(FAN_2D)
#if defined(FAN_GUI) && defined(FAN_PHYSICS_2D)

import std;

import fan.types;
import fan.window;
import fan.window.input;
import fan.graphics.common_context;
import fan.graphics.gui.types;
import fan.graphics.gui;
import fan.types.color;
import fan.types.vector;
import fan.types.json;
import fan.math;
import fan.print.error;
import fan.file_dialog;
import fan.io.file;
import fan.graphics;
import fan.random;
import fan.physics.types;
import fan.physics.b2_integration;
import fan.math.intersection;
import fan.graphics.physics_shapes;
import fan.graphics.shapes;
import fan.graphics.algorithm.raycast_grid;
import fan.texture_pack.tp0;
import fan.graphics.gui.base;
import fan.graphics.gui.text_logger;

using namespace fan::graphics;

void fte_t::open(const properties_t& properties) {
    image_load_properties_t lp;
    lp.visual_output = image_sampler_address_mode_e::repeat;
    lp.min_filter = image_filter_e::nearest;
    lp.mag_filter = image_filter_e::nearest;

    if (properties.camera == nullptr) {
      render_view = &get_orthographic_render_view();
    }
    else {
      render_view = properties.camera;
    }

    mouse_move_handle = get_window().add_mouse_move_callback([this](const fan::window_t::mouse_move_data_t& d) {
      if (viewport_settings.move) {
        fan::vec2 move_off = (d.position - viewport_settings.offset) / camera_get_zoom(render_view->camera);
        camera_set_position(render_view->camera, viewport_settings.pos - move_off);
      }

      fan::vec2i p;
      if (window_relative_to_grid(d.position, &p)) {
        grid_visualize.highlight_hover.set_position(fan::vec2(p) + tile_size * brush.offset);
        grid_visualize.highlight_hover.set_size(tile_size * brush.tile_size * brush.size);
        grid_visualize.highlight_hover.set_color(fan::color(1, 1, 1, 0.6));
      }
      else {
        grid_visualize.highlight_hover.set_color(fan::colors::transparent);
      }
    });

    buttons_handle = get_window().add_buttons_callback([this](const fan::window_t::buttons_data_t& d) {
      if (d.button == fan::mouse_left && d.state == fan::mouse_state::release) {
        prev_grid_position = -999999;
      }

      if (!editor_settings.hovered && d.state != fan::mouse_state::release) {
        return;
      }

      switch (d.button) {
        case fan::mouse_middle: {
          viewport_settings.move = (bool)d.state;
          viewport_settings.offset = fan::window::get_mouse_position();
          viewport_settings.pos = camera_get_position(render_view->camera);
          break;
        }
        case fan::mouse_scroll_up: {
          if (get_window().key_pressed(fan::key_left_control)) {
            brush.depth += 1;
            brush.depth = std::min((int)brush.depth, shape_depths_t::max_layer_depth);
          }
          else if (get_window().key_pressed(fan::key_left_shift)) {
            brush.size += 1;
            grid_visualize.highlight_hover.set_size(tile_size * brush.size);
          }
          else {
            camera_set_zoom(render_view->camera, camera_get_zoom(render_view->camera) * scroll_speed);
            fan::vec2 pos = (get_window().get_mouse_position() - viewport_settings.window_related_mouse_pos);
            pos /= get_window().get_size();
            pos *= viewport_settings.size / 2;
          }
          return;
        }
        case fan::mouse_scroll_down: {
          if (get_window().key_pressed(fan::key_left_control)) {
            brush.depth -= 1;
            brush.depth = std::max((std::uint32_t)brush.depth, (std::uint32_t)1);
          }
          else if (get_window().key_pressed(fan::key_left_shift)) {
            brush.size = (brush.size - 1).max(fan::vec2i(1));
            grid_visualize.highlight_hover.set_size(tile_size * brush.size);
          }
          else {
            camera_set_zoom(render_view->camera, camera_get_zoom(render_view->camera) / scroll_speed);
          }
          return;
        }
      }
    });

    keys_handle = get_window().add_keys_callback([this](const fan::window_t::keys_data_t& d) {
      if (d.state != fan::keyboard_state::press || gui::is_any_item_active()) {
        return;
      }
      switch (d.key) {
        case fan::key_r: {
          brush.angle.z = std::fmod(brush.angle.z + fan::math::pi / 2, fan::math::pi * 2);
          break;
        }
        case fan::key_delete: {
          if (get_window().key_pressed(fan::key_left_control)) {
            reset_map();
          }
          break;
        }
      }
    });

    viewport_settings.size = 0;
    transparent_texture = create_transparent_texture();

    grid_visualize.background = sprite_t{{
      .render_view = render_view,
      .position = fan::vec3(tile_size * 2 * map_size / 2 - tile_size, 0),
      .size = 0,
      .image = transparent_texture,
    }};

    lp.min_filter = image_filter_e::nearest;
    lp.mag_filter = image_filter_e::nearest;

    grid_visualize.highlight_color = image_load("images/highlight_hover.webp", lp);
    grid_visualize.collider_color = image_create(fan::color(0, 0.5, 0, 0.5));
    grid_visualize.light_color = image_load("images/lightbulb.webp", lp);

    grid_visualize.highlight_hover = unlit_sprite_t{{
      .render_view = render_view,
      .position = fan::vec3(viewport_settings.pos, shape_depths_t::cursor_highlight_depth),
      .size = tile_size,
      .image = grid_visualize.highlight_color,
      .blending = true
    }};

    grid_visualize.highlight_selected = unlit_sprite_t{{
      .render_view = render_view,
      .position = fan::vec3(viewport_settings.pos, shape_depths_t::cursor_highlight_depth - 1),
      .size = 0,
      .color = fan::color(2, 2, 2, 1),
      .image = grid_visualize.highlight_color,
      .blending = true
    }};

    fan::vec2 p = 0;
    p = ((p - tile_size) / tile_size).floor() * tile_size;
    grid_visualize.highlight_hover.set_position(p);

    camera_set_position(render_view->camera, viewport_settings.pos + tile_size * 2.f * map_size / 2.f);

    shapes::grid_t::properties_t gp;
    gp.viewport = render_view->viewport;
    gp.camera = render_view->camera;
    gp.position = fan::vec3(map_size * (tile_size * 2.f) / 2.f - tile_size, shape_depths_t::cursor_highlight_depth - 1);
    gp.size = 0;
    gp.color = fan::colors::black.set_alpha(0.4);
    grid_visualize.grid = gp;
    resize_map();

    visual_line = line_t{{
      .render_view = render_view,
      .src = fan::vec3(0, 0, shape_depths_t::cursor_highlight_depth + 1),
      .dst = fan::vec2(400),
      .color = fan::colors::white
    }};

    content_browser.init("");
    content_browser.current_view_mode = gui::content_browser_t::view_mode_large_thumbnails;
  }

bool fte_t::handle_tile_push(fan::vec2i& position, int& pj, int& pi) {
    if (apply_jitter(position) || !is_in_constraints(position, pj, pi)) {
      return true;
    }

    f32_t inital_x = position.x;
    fan::vec2i grid_position = position;
    convert_draw_to_grid(grid_position);
    brush.line_src = snap_to_tile_center(get_mouse_position(render_view->camera, render_view->viewport));
    grid_position /= (tile_size * 2);

    if (brush.type == brush_t::type_e::player_spawn || 
        brush.type == brush_t::type_e::enemy_spawn || 
        brush.type == brush_t::type_e::mark) {

      auto& marks = spawn_marks[brush.depth];
      fan::vec3 pos = fan::vec3(position, brush.depth);

      for (auto& mark : marks) {
        if (mark.position == pos) return false;
      }

      spawn_mark_t new_mark;
      new_mark.position = pos;
      new_mark.size = tile_size * brush.tile_size;
      new_mark.id = brush.id;
      new_mark.color = brush.color;

      if (brush.type == brush_t::type_e::player_spawn) new_mark.type = mesh_property_t::player_spawn;
      else if (brush.type == brush_t::type_e::enemy_spawn) new_mark.type = mesh_property_t::enemy_spawn;
      else new_mark.type = mesh_property_t::mark;

      marks.push_back(new_mark);
      visual_shapes[pos].shape = make_sprite(pos, new_mark.size, fan::color(1), render_view, get_marker_image(new_mark.type));
      return false;
    }

    auto& layers = map_tiles[grid_position].layers;
    visual_layers[brush.depth].positions[grid_position] = 1;
    std::uint32_t idx = find_layer_shape(layers);

    if (idx == invalid && (brush.type == brush_t::type_e::light)) {
      layers.resize(layers.size() + 1);
      layers.back().tile.position = fan::vec3(position, brush.depth);
      layers.back().tile.size = tile_size * brush.tile_size;

      fan::vec3 pos = fan::vec3(position, brush.depth);
      layers.back().tile.id = brush.id;
      layers.back().shape = light_t{{
        .render_view = render_view,
        .position = pos,
        .size = tile_size * brush.tile_size,
        .color = brush.dynamics_color == brush_t::dynamics_e::randomize ? fan::random::color() : brush.color,
        .flags = brush.flags
      }};
      layers.back().tile.mesh_property = mesh_property_t::light;
      visual_shapes[pos].shape = make_sprite(
        fan::vec3(fan::vec2(pos), pos.z + 1),
        tile_size, fan::color(1), render_view, grid_visualize.light_color
      );
      
      current_tile.position = position;
      current_tile.layer = layers.data();
      current_tile.layer_index = layers.size() - 1;
      return false;
    }
    else if (brush.type == brush_t::type_e::light) {
      if (idx != invalid || idx < layers.size()) {
        auto& layer = layers[idx];
        layer.tile.id = brush.id;
        layer.tile.size = tile_size;

        auto& shape = layer.shape;
        fan::vec3 pos = shape.get_position();

        layer.shape = light_t{ {
          .render_view = render_view,
          .position = pos,
          .size = shape.get_size(),
          .color = shape.get_color(),
          .flags = shape.get_flags(),
          .angle = shape.get_angle(),
        } };
        layer.tile.mesh_property = mesh_property_t::light;

        visual_shapes[pos].shape = make_sprite(
          fan::vec3(fan::vec2(pos), pos.z + 1),
          tile_size, fan::color(1), render_view, grid_visualize.light_color
        );
      }
      return false;
    }

    for (auto& i : current_tile_images) {
      for (auto& tile : i) {
        grid_position = position / (tile_size * 2);

        if (!is_in_constraints(position)) {
          position.x += tile_size.x * 2;
          continue;
        }

        if (idx == invalid) {
          visual_layers[brush.depth].positions[grid_position] = 1;
          auto& layers = map_tiles[grid_position].layers;
          layers.resize(layers.size() + 1);
          layers.back().tile.size = tile_size * brush.tile_size;
          layers.back().tile.position = fan::vec3(position, brush.depth);
          layers.back().tile.id = brush.id;
          layers.back().tile.mesh_property = mesh_property_t::none;

          if (brush.type != brush_t::type_e::light) {
            layers.back().shape = sprite_t{{
              .render_view = render_view,
              .position = fan::vec3(position, brush.depth),
              .size = tile_size * brush.tile_size,
              .angle = brush.dynamics_angle == brush_t::dynamics_e::randomize ? fan::vec3(0, 0, get_snapped_angle()) : brush.angle,
              .color = brush.dynamics_color == brush_t::dynamics_e::randomize ? fan::random::color() : brush.color,
              .blending = true
            }};
          }

          if (brush.type == brush_t::type_e::texture) {
            if (layers.back().shape.set_tp(&tile.ti)) {
              gui::print("failed to load image");
            }
          }
          
          current_tile.position = position;
          current_tile.layer = layers.data();
          current_tile.layer_index = layers.size() - 1;
        }
        else {
          auto found = map_tiles.find(grid_position);
          if (found != map_tiles.end()) {
            auto& layers = found->second.layers;
            idx = find_layer_shape(layers);

            if (idx != invalid || idx < layers.size()) {
              auto& layer = layers[idx];
              if (brush.dynamics_angle == brush_t::dynamics_e::original && layer.shape.get_angle() != brush.angle) {
                layer.shape.set_angle(brush.angle);
              }

              layer.tile.size = tile_size * brush.tile_size;
              layer.shape.set_size(tile_size * brush.tile_size);
              layer.shape.set_color(brush.color);
              layer.tile.id = brush.id;

              if (brush.type == brush_t::type_e::texture) {
                layer.shape = sprite_t{{
                  .render_view = render_view,
                  .position = layer.shape.get_position(),
                  .size = layer.shape.get_size(),
                  .angle = layer.shape.get_angle(),
                  .color = layer.shape.get_color()
                }};

                if (layer.shape.set_tp(&tile.ti)) gui::print("failed to load image");
                layer.tile.mesh_property = mesh_property_t::none;
              }
            }
          }
        }
        position.x += tile_size.x * 2;
      }
      position.x = inital_x;
      position.y += tile_size.y * 2;
    }

    return false;
  }

bool fte_t::handle_tile_erase(fan::vec2i& position, int& j, int& i) {
    if (!is_in_constraints(position, j, i)) return true;

    fan::vec2i grid_position = position;
    convert_draw_to_grid(grid_position);
    grid_position /= tile_size * 2;

    auto found_marks = spawn_marks.find(brush.depth);
    if (found_marks != spawn_marks.end()) {
      for (auto it = found_marks->second.begin(); it != found_marks->second.end(); ++it) {
        if (fan::math::d2::aabb_point_inside(position, it->position, it->size)) {
          auto visual_found = visual_shapes.find(it->position);
          if (visual_found != visual_shapes.end()) {
            visual_shapes.erase(visual_found);
          }
          found_marks->second.erase(it);
          prev_grid_position = -999999;
          return false;
        }
      }
    }

    auto found = physics_shapes.find(brush.depth);
    if (found != physics_shapes.end()) {
      for (auto it = found->second.begin(); it != found->second.end(); ++it) {
        if (fan::math::d2::aabb_point_inside(position, it->visual.get_position(), it->visual.get_size())) {
          found->second.erase(it);
          return false;
        }
      }
    }

    if (apply_jitter(position)) return true;
    
    auto found_tile = map_tiles.find(grid_position);
    if (found_tile != map_tiles.end()) {
      auto& layers = found_tile->second.layers;
      std::uint32_t idx = find_layer_shape(layers);

      if (idx != invalid || idx < layers.size()) {
        if (layers[idx].tile.mesh_property == mesh_property_t::light) {
          fan::vec3 erase_position = layers[idx].shape.get_position();
          auto found_visual = visual_shapes.find(erase_position);
          if (found_visual != visual_shapes.end()) visual_shapes.erase(found_visual);
        }
        layers.erase(layers.begin() + idx);
        auto visual_found = visual_layers.find(brush.depth);
        if (visual_found != visual_layers.end()) {
          visual_found->second.positions.erase(found_tile->first);
          if (visual_found->second.positions.empty()) {
            visual_layers.erase(visual_found);
          }
        }
      }
      if (found_tile->second.layers.empty()) {
        map_tiles.erase(found_tile->first);
      }
    }
    invalidate_selection();
    prev_grid_position = -999999;
    return false;
  }

void fte_t::fout(std::string filename) {
    if (!filename.ends_with(".fte") && !filename.ends_with(".json")) {
      filename += ".fte";
    }
    bool is_temp = filename.find("temp") != std::string::npos;

  #if defined(FAN_JSON)
    previous_filename = filename;
    fan::json ostr;

    ostr["version"] = 1;
    ostr["map_size"] = map_size;
    ostr["tile_size"] = tile_size;
    ostr["camera_position"] = camera_get_position(render_view->camera);
    ostr["camera_zoom"] = camera_get_zoom(render_view->camera);
    ostr["lighting.ambient"] = get_lighting().ambient;
    ostr["gravity"] = fan::physics::gphysics()->get_gravity();

    fan::json jtps = fan::json::array();
    jtps.reserve(texture_packs.size());
    for (auto* tp : texture_packs) {
      std::filesystem::path tp_path = std::filesystem::absolute(tp->file_path);
      std::filesystem::path file_dir = std::filesystem::absolute(filename).parent_path();
      jtps.push_back(std::filesystem::relative(tp_path, file_dir).generic_string());
    }
    ostr["texture_packs"] = jtps;

    std::size_t total_tiles = 0;
    for (auto& [depth, vec] : physics_shapes) {
      total_tiles += vec.size();
    }
    for (auto& [depth, marks] : spawn_marks) {
      total_tiles += marks.size();
    }

    std::unordered_map<std::uint16_t, std::unordered_map<fan::vec2i, fte_t::shapes_t::global_t::layer_t*>> depth_tiles;
    depth_tiles.reserve(visual_layers.size());

    for (auto& [gp, cell] : map_tiles) {
      for (auto& layer : cell.layers) {
        depth_tiles[layer.tile.position.z][gp] = &layer;
      }
    }

    for (auto& [depth, tilemap] : depth_tiles) {
      total_tiles += tilemap.size() / 2 + 1;
    }

    fan::json tiles = fan::json::array();
    tiles.reserve(total_tiles);

    static const fte_t::tile_t defaults = fte_t::tile_t();

    for (auto& [depth, tilemap] : depth_tiles) {
      std::unordered_set<fan::vec2i> visited;
      visited.reserve(tilemap.size());

      std::vector<std::pair<fan::vec2i, fte_t::shapes_t::global_t::layer_t*>> sorted_tiles;
      sorted_tiles.reserve(tilemap.size());
      for (auto& pair : tilemap) {
        sorted_tiles.push_back(pair);
      }

      std::sort(sorted_tiles.begin(), sorted_tiles.end(), 
        [](const auto& a, const auto& b) {
        return a.first.y < b.first.y || (a.first.y == b.first.y && a.first.x < b.first.x);
      });

      for (auto& [gp, base] : sorted_tiles) {
        if (visited.contains(gp)) {
          continue;
        }

        int max_x = gp.x;
        {
          fan::vec2i test_pos = gp;
          while (true) {
            test_pos.x = max_x + 1;
            if (visited.contains(test_pos)) break;

            auto it = tilemap.find(test_pos);
            if (it == tilemap.end() || !same_visual(*base, *it->second)) {
              break;
            }
            max_x++;
          }
        }

        int max_y = gp.y;
        {
          bool can_extend = true;
          while (can_extend) {
            int test_y = max_y + 1;
            for (int x = gp.x; x <= max_x; ++x) {
              fan::vec2i p(x, test_y);
              if (visited.contains(p)) {
                can_extend = false;
                break;
              }
              auto it = tilemap.find(p);
              if (it == tilemap.end() || !same_visual(*base, *it->second)) {
                can_extend = false;
                break;
              }
            }
            if (can_extend) {
              max_y = test_y;
            }
          }
        }

        std::vector<fan::vec2i> to_visit;
        int rect_size = (max_x - gp.x + 1) * (max_y - gp.y + 1);
        to_visit.reserve(rect_size);
        for (int y = gp.y; y <= max_y; ++y) {
          for (int x = gp.x; x <= max_x; ++x) {
            to_visit.emplace_back(x, y);
          }
        }
        visited.insert(to_visit.begin(), to_visit.end());

        fan::json tile_json;
        shape_serialize(base->shape, &tile_json);

        if (base->tile.mesh_property != defaults.mesh_property) {
          tile_json["mesh_property"] = static_cast<std::uint32_t>(base->tile.mesh_property);
        }
        if (!base->tile.id.empty() && base->tile.id != defaults.id) {
          tile_json["id"] = base->tile.id;
        }
        if (base->tile.action != defaults.action) {
          tile_json["action"] = base->tile.action;
        }
        if (base->tile.key != defaults.key) {
          tile_json["key"] = base->tile.key;
        }
        if (base->tile.key_state != defaults.key_state) {
          tile_json["key_state"] = base->tile.key_state;
        }
        if (!base->tile.object_names.empty() && base->tile.object_names != defaults.object_names) {
          tile_json["object_names"] = base->tile.object_names;
        }

        int count_x = max_x - gp.x + 1;
        int count_y = max_y - gp.y + 1;

        if (count_x > 1 || count_y > 1) {
          fan::json inst;
          inst["count_x"] = count_x;
          inst["count_y"] = count_y;
          inst["delta_x"] = fan::vec3((f32_t)tile_size.x * 2.f, 0.f, 0.f);
          inst["delta_y"] = fan::vec3(0.f, (f32_t)tile_size.y * 2.f, 0.f);
          tile_json["instance"] = inst;
        }

        tiles.push_back(tile_json);
      }
    }

    {
      std::unordered_set<fan::vec3> seen;
      seen.reserve(physics_shapes.size() * 10);

      static const fan::physics::shape_properties_t default_props;

      for (auto& [depth, vec] : physics_shapes) {
        for (auto& j : vec) {
          if (!seen.insert(j.visual.get_position()).second) {
            continue;
          }

          fan::json tile;
          shape_serialize(j.visual, &tile);
          tile["mesh_property"] = static_cast<std::uint32_t>(mesh_property_t::physics_shape);

          if (!j.id.empty()) {
            tile["id"] = j.id;
          }

          fan::json ps;
          if (j.type != 0) ps["type"] = j.type;
          if (j.body_type != 0) ps["body_type"] = j.body_type;
          if (j.draw != false) ps["draw"] = j.draw;
          if (j.shape_properties.friction != default_props.friction) {
            ps["friction"] = j.shape_properties.friction;
          }
          if (j.shape_properties.density != default_props.density) {
            ps["density"] = j.shape_properties.density;
          }
          if (j.shape_properties.fixed_rotation != default_props.fixed_rotation) {
            ps["fixed_rotation"] = j.shape_properties.fixed_rotation;
          }
          if (j.shape_properties.presolve_events != default_props.presolve_events) {
            ps["presolve_events"] = j.shape_properties.presolve_events;
          }
          if (j.shape_properties.is_sensor != default_props.is_sensor) {
            ps["is_sensor"] = j.shape_properties.is_sensor;
          }

          if (!ps.empty()) {
            tile["physics_shape_data"] = ps;
          }

          tiles.push_back(tile);
        }
      }
    }

    for (auto& [depth, marks] : spawn_marks) {
      for (auto& m : marks) {
        fan::json tile;
        tile["shape"] = "sprite";
        
        tile["position"] = m.position;
        tile["size"] = m.size;
        tile["color"] = m.color;
        
        tile["mesh_property"] = static_cast<std::uint32_t>(m.type);
        if (!m.id.empty()) {
          tile["id"] = m.id;
        }
        tiles.push_back(tile);
      }
    }

    {
      fan::json j = fan::json::array();
      j.reserve(visual_layers.size());

      for (auto& [depth, layer] : visual_layers) {
        fan::json l;
        l["layer_name"] = layer.text;
        l["depth"] = depth;
        j.push_back(l);
      }
      ostr["layer_info"] = j;
    }

    ostr["tiles"] = tiles;

    std::string json_str = ostr.dump(2);

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
      fan::throw_error("Failed to open file for writing: " + filename);
      return;
    }

    file.write(json_str.c_str(), json_str.size());
    file.close();

    if (!is_temp) {
      gui::print_success("File saved to " + std::filesystem::absolute(filename).generic_string());
    }
  #else
    fan::throw_error("FAN_JSON not enabled");
  #endif
  }

void fte_t::fin(const std::string& filename, const std::source_location& callers_path) {
  #if defined(FAN_JSON)

    std::ifstream file(fan::io::file::find_relative_path(filename, callers_path), std::ios::binary | std::ios::ate);
    if (!file) {
      fan::throw_error("Failed to open file: " + filename);
      return;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string buffer;
    buffer.resize(size);

    if (!file.read(&buffer[0], size)) {
      fan::throw_error("Failed to read file: " + filename);
      return;
    }
    file.close();

    fan::json json = fan::json::parse(buffer);
    buffer.clear(); 
    buffer.shrink_to_fit();

    if (json["version"] != 1) {
      fan::throw_error("version mismatch");
    }

    if (json.contains("texture_packs")) {
      std::vector<std::string> tp_paths = json["texture_packs"];
      std::filesystem::path file_dir = std::filesystem::absolute(
        fan::io::file::find_relative_path(filename, callers_path)
      ).parent_path();
      for (auto& path : tp_paths) {
        std::string resolved = (file_dir / path).lexically_normal().generic_string();
        open_texture_pack(resolved);
      }
    }
    else if (texture_packs.empty() || texture_packs[0]->size() == 0) {
      gui::print("open valid texturepack");
      return;
    }

    invalidate_selection();
    previous_filename = filename;

    map_size = json["map_size"];
    tile_size = json["tile_size"];

    if (json.contains("camera_position")) {
      camera_set_position(render_view->camera, json["camera_position"]);
    }
    if (json.contains("camera_zoom")) {
      camera_set_zoom(render_view->camera, json["camera_zoom"]);
    }
    if (json.contains("gravity")) {
      fan::physics::gphysics()->set_gravity(json["gravity"]);
    }
    get_lighting().set_target(json["lighting.ambient"]);

    map_tiles.clear();
    visual_layers.clear();
    visual_shapes.clear();
    physics_shapes.clear();
    spawn_marks.clear();

    std::size_t estimated_tiles = map_size.x * map_size.y / 4;
    map_tiles.reserve(estimated_tiles);

    resize_map();

    static image_t player_marker_image = image_create(fan::color(0, 1, 0, 0.5));
    static image_t enemy_marker_image = image_create(fan::color(1, 0, 0, 0.5));
    static image_t mark_marker_image = image_create(fan::color(1, 1, 0, 0.5));

    static const fte_t::physics_shapes_t physics_defaults;
    static const fte_t::tile_t tile_defaults = fte_t::tile_t();

    auto expand_instance = [&](const fan::json& shape_json, auto emit_one) {
      if (!shape_json.contains("instance")) {
        emit_one(fan::vec3(0, 0, 0));
        return;
      }

      const auto& inst = shape_json["instance"];

      if (inst.contains("count_x")) {
        int cx = inst.value("count_x", 1);
        int cy = inst.value("count_y", 1);

        fan::vec3 dx = inst.value("delta_x", fan::vec3(0, 0, 0));
        fan::vec3 dy = inst.value("delta_y", fan::vec3(0, 0, 0));

        for (int y = 0; y < cy; y++) {
          for (int x = 0; x < cx; x++) {
            emit_one(dx * (f32_t)x + dy * (f32_t)y);
          }
        }
        return;
      }

      int count = inst.value("count", 1);
      fan::vec3 delta = inst.value("delta", fan::vec3(0, 0, 0));
      for (int i = 0; i < count; i++) {
        emit_one(delta * (f32_t)i);
      }
    };

    shape_deserialize_t it;
    shape_t base_shape;

    const auto& tiles_array = json["tiles"];

    while (it.iterate(tiles_array, &base_shape)) {
      const auto shape_json = *(it.data.it - 1);

      expand_instance(shape_json, [&](const fan::vec3& offs) {
        shape_t shape = base_shape;
        shape.set_position(shape.get_position() + offs);

        if (shape_json.contains("mesh_property")) {
          auto mesh_prop = static_cast<fte_t::mesh_property_t>(shape_json["mesh_property"].get<std::uint32_t>());

          if (mesh_prop == fte_t::mesh_property_t::physics_shape) {
            std::uint16_t depth = shape.get_position().z;
            auto& physics_shape = physics_shapes[depth];
            physics_shape.emplace_back();
            auto& physics_element = physics_shape.back();

            shape.set_camera(render_view->camera);
            shape.set_viewport(render_view->viewport);
            shape.set_image(grid_visualize.collider_color);

            if (shape_json.contains("physics_shape_data")) {
              physics_element.id = shape_json.value("id", physics_defaults.id);
              const fan::json& physics_shape_data = shape_json["physics_shape_data"];
              physics_element.type = physics_shape_data.value("type", physics_defaults.type);
              physics_element.body_type = physics_shape_data.value("body_type", physics_defaults.body_type);
              physics_element.draw = physics_shape_data.value("draw", physics_defaults.draw);
              physics_element.shape_properties.friction = physics_shape_data.value("friction", physics_defaults.shape_properties.friction);
              physics_element.shape_properties.density = physics_shape_data.value("density", physics_defaults.shape_properties.density);
              physics_element.shape_properties.fixed_rotation = physics_shape_data.value("fixed_rotation", physics_defaults.shape_properties.fixed_rotation);
              physics_element.shape_properties.presolve_events = physics_shape_data.value("presolve_events", physics_defaults.shape_properties.presolve_events);
              physics_element.shape_properties.is_sensor = physics_shape_data.value("is_sensor", physics_defaults.shape_properties.is_sensor);
            }

            physics_element.visual = std::move(shape);
            return;
          }
          else if (mesh_prop == mesh_property_t::player_spawn ||
            mesh_prop == mesh_property_t::enemy_spawn ||
            mesh_prop == mesh_property_t::mark) {

            spawn_mark_t mark;
            mark.position = shape_json.contains("position") ? shape_json["position"].get<fan::vec3>() : fan::vec3(0);
            mark.position += offs;
            mark.size = shape_json.contains("size") ? shape_json["size"].get<fan::vec2>() : fan::vec2(0);
            
            if (shape_json.contains("color")) {
              auto col_array = shape_json["color"];
              mark.color = fan::color(col_array[0], col_array[1], col_array[2], col_array[3]);
            } else {
              mark.color = fan::colors::white;
            }

            mark.type = mesh_prop;
            mark.id = shape_json.value("id", "");

            std::uint16_t depth = mark.position.z;
            spawn_marks[depth].push_back(std::move(mark));
            auto& inserted_mark = spawn_marks[depth].back();

            image_t marker_image;
            if (mesh_prop == mesh_property_t::player_spawn) {
              marker_image = player_marker_image;
            }
            else if (mesh_prop == mesh_property_t::enemy_spawn) {
              marker_image = enemy_marker_image;
            }
            else {
              marker_image = mark_marker_image;
            }

            visual_shapes[inserted_mark.position].shape = 
              make_sprite(inserted_mark.position, inserted_mark.size, 
                fan::color(1), render_view, marker_image);
            return;
          }
        }

        fan::vec2i gp = shape.get_position();
        std::uint16_t depth = shape.get_position().z;

        convert_draw_to_grid(gp);
        gp /= tile_size * 2;

        visual_layers[depth].positions[gp];

        auto& cell = map_tiles[gp];
        cell.layers.emplace_back();
        auto* layer = &cell.layers.back();

        layer->tile.position = fan::vec3i(gp, depth);
        layer->tile.size = shape.get_size();
        layer->tile.angle = shape.get_angle();
        layer->tile.color = shape.get_color();
        layer->tile.id = shape_json.value("id", tile_defaults.id);
        layer->tile.mesh_property = shape_json.contains("mesh_property") 
            ? static_cast<fte_t::mesh_property_t>(shape_json["mesh_property"].get<std::uint32_t>()) 
            : tile_defaults.mesh_property;

        layer->shape = std::move(shape);

        switch (layer->tile.mesh_property) {
        case fte_t::mesh_property_t::none: {
          layer->shape.set_camera(render_view->camera);
          layer->shape.set_viewport(render_view->viewport);
          break;
        }
        case fte_t::mesh_property_t::light: {
          layer->shape = light_t{{
              .render_view = render_view,
              .position = layer->shape.get_position(),
              .size = layer->tile.size,
              .color = layer->tile.color,
              .flags = layer->shape.get_flags()
            }};
          visual_shapes[layer->shape.get_position()].shape = sprite_t{{
              .render_view = render_view,
              .position = fan::vec3(fan::vec2(layer->shape.get_position()), layer->shape.get_position().z + 1),
              .size = tile_size,
              .image = grid_visualize.light_color,
              .blending = true
            }};
          break;
        }
        default: {
          fan::throw_error("");
        }
        }
      });
    }

    if (json.contains("layer_info")) {
      for (const auto& layer_json : json["layer_info"]) {
        layer_info_t layer_info;
        layer_info.layer_name = layer_json["layer_name"];
        layer_info.depth = layer_json["depth"];
        if (visual_layers.find(layer_info.depth) != visual_layers.end()) {
          visual_layers[layer_info.depth].text = std::move(layer_info.layer_name);
        }
      }
    }
  #else
    fan::throw_error("FAN_JSON not enabled");
  #endif
  }

#endif
#endif
#endif
