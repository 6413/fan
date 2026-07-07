module;

#if defined (FAN_WINDOW)

#include <cstddef>

#if defined(FAN_2D)
  #include <fan/utility.h>
#endif

#endif

export module fan.graphics.gui.tilemap_editor.core;

#if defined (FAN_WINDOW)

import std;

#if defined(FAN_2D)
#if defined(FAN_GUI) && defined(FAN_PHYSICS_2D)

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

export struct fte_t {
  static constexpr int max_id_len = 48;
  static constexpr fan::vec2 default_button_size{100, 30};
  static constexpr fan::vec2 tile_viewer_sprite_size{64, 64};
  static constexpr fan::color highlighted_tile_color = fan::color(0.5, 0.5, 1);
  static constexpr fan::color highlighted_selected_tile_color = fan::color(0.5, 0, 0, 0.1);
  static constexpr f32_t scroll_speed = 1.2;
  static constexpr std::uint32_t invalid = -1;

  struct shape_depths_t {
    static constexpr int max_layer_depth = 0xFAAA - 2;
    static constexpr int cursor_highlight_depth = 0xFAAA - 1;
  };

  #include "common2.h"

  struct shapes_t {
    struct global_t {
      global_t() = default;

      template <typename T>
      global_t(fte_t* root, const T& obj) {
        layers.push_back(obj);
      }

      struct layer_t {
        tile_t tile;
        shape_t shape;
      };
      std::vector<layer_t> layers;
    };
  };

  enum class event_type_e {
    none,
    add,
    remove
  };

  struct sort_by_y_t {
    bool operator()(const fan::vec2i& a, const fan::vec2i& b) const {
      if (a.y == b.y) {
        return a.x < b.x;
      }
      return a.y < b.y;
    }
  };

  struct tile_info_t {
    texture_pack::ti_t ti;
    mesh_property_t mesh_property = mesh_property_t::none;
  };

  struct current_tile_t {
    fan::vec2i position = 0;
    shapes_t::global_t::layer_t* layer = nullptr;
    std::uint32_t layer_index;
  };

  struct visual_layer_t {
    std::unordered_map<fan::vec2i, bool> positions;
    std::string text = "default";
    bool visible = true;
  };

  struct visualize_t {
    shape_t shape;
  };

  struct brush_t {
    enum class mode_e : std::uint8_t { draw, copy };
    mode_e mode = mode_e::draw;
    fan::vec2 line_src = -9999999;
    static constexpr const char* mode_names[] = {"Draw", "Copy"};

    enum class type_e : std::uint8_t {
      texture,
      physics_shape = (std::uint8_t)fte_t::mesh_property_t::physics_shape,
      light,
      player_spawn,
      enemy_spawn,
      mark
    };
    static constexpr const char* type_names[] = {"Texture", "Physics shape", "Light", "Player spawn", "Enemy spawn", "Mark"};
    type_e type = type_e::texture;

    enum class dynamics_e : std::uint8_t { original, randomize };
    static constexpr const char* dynamics_names[] = {"Original", "Randomize"};
    dynamics_e dynamics_angle = dynamics_e::original;
    dynamics_e dynamics_color = dynamics_e::original;

    fan::vec2i size = 1;
    fan::vec2 tile_size = 1;
    fan::vec3 angle = 0;
    f32_t depth = shape_depths_t::max_layer_depth / 2;
    int jitter = 0;
    f32_t jitter_chance = 0.33;
    std::string id;
    fan::color color = fan::color(1);
    fan::vec2 offset = 0;
    std::uint32_t flags = 0;
    std::uint8_t physics_type = physics_shapes_t::type_e::box;
    static constexpr const char* physics_type_names[] = {"Box", "Circle"};
    std::uint8_t physics_body_type = fan::physics::body_type_e::static_body;
    static constexpr const char* physics_body_type_names[] = {"Static", "Kinematic", "Dynamic"};
    bool physics_draw = false;
    fan::physics::shape_properties_t physics_shape_properties;
  };

  struct viewport_settings_t {
    bool move = false;
    fan::vec2 pos = 0;
    fan::vec2 size = 0;
    fan::vec2 offset = 0;
    fan::vec2 window_related_mouse_pos = 0;
  };

  struct editor_settings_t {
    bool hovered = false;
  };

  struct grid_visualize_t {
    shape_t background;
    shape_t highlight_selected;
    shape_t highlight_hover;
    shape_t grid;
    image_t highlight_color;
    image_t collider_color;
    image_t light_color;
    bool render_grid = true;
  };

  struct properties_t {
    render_view_t* camera = nullptr;
  };

  struct layer_info_t {
    std::string layer_name;
    std::uint16_t depth;
  };

  struct spawn_mark_t {
    fan::vec3 position;
    fan::vec2 size;
    mesh_property_t type;
    std::string id;
    fan::color color = fan::colors::white;

    void json_write(fan::json& j) const {
      j["position"] = position;
      j["size"] = size;
      j["color"] = color;
      j["mesh_property"] = type;
      if (!id.empty()) j["id"] = id;
    }
    void json_read(const fan::json& j) {
      position = j.value("position", fan::vec3(0));
      size = j.value("size", fan::vec2(0));
      color = j.value("color", fan::colors::white);
      type = j.value("mesh_property", mesh_property_t::mark);
      id = j.value("id", "");
    }
  };

  std::uint32_t find_layer_shape(const auto& vec, bool top = false) {
    std::uint32_t found = invalid;
    std::int64_t depth = -1;
    for (std::size_t i = 0; i < vec.size(); ++i) {
      if (top) {
        if (vec[i].tile.position.z > depth) {
          depth = vec[i].tile.position.z;
          found = i;
        }
      }
      else if (vec[i].tile.position.z == brush.depth) {
        return i;
      }
    }
    return found;
  }

  void resize_map() {
    grid_visualize.background.set_size(tile_size * map_size);
    grid_visualize.background.set_tc_size(fan::vec2(0.5) * map_size);
    grid_visualize.background.set_position(fan::vec3(tile_size * 2 * map_size / 2 - tile_size, 0));

    grid_visualize.grid.set_position(fan::vec2(map_size * (tile_size * 2.f) / 2.f - tile_size));
    grid_visualize.grid.set_grid_size(map_size / 2.f);

    if (grid_visualize.render_grid) {
      grid_visualize.grid.set_size(map_size * (tile_size * 2.f) / 2.f);
    }
    else {
      grid_visualize.grid.set_size(0);
    }

    fan::vec2 s = grid_visualize.highlight_hover.get_size();
    fan::vec2 sp = fan::vec2(grid_visualize.highlight_hover.get_position());
    fan::vec2 p = tile_size * ((sp / s));
    grid_visualize.highlight_hover.set_position(p);
    grid_visualize.highlight_hover.set_size(tile_size);
    grid_visualize.highlight_selected.set_position(p);

    if (current_tile.layer != nullptr) {
      grid_visualize.highlight_selected.set_size(tile_size);
    }
  }

  void reset_map() {
    physics_shapes.clear();
    visual_layers.clear();
    map_tiles.clear();
    spawn_marks.clear();
    visual_shapes.clear();
    previous_filename.clear();
    invalidate_selection();
    resize_map();
  }

  bool window_relative_to_grid(const fan::vec2& window_relative_position, fan::vec2i* in) {
    auto camera_position = camera_get_position(render_view->camera);
    fan::vec2 p = translate_position(window_relative_position, render_view->viewport, render_view->camera) + camera_position;
    *in = ((p + tile_size) / (tile_size * 2)).floor() * (tile_size * 2);
    return fan::math::d2::aabb_point_inside(*in - map_size * tile_size / 2, map_size / 2 * tile_size - tile_size, map_size * tile_size);
  }

  void convert_draw_to_grid(fan::vec2i& p) {}
  void convert_grid_to_draw(fan::vec2i& p) {}

  void open(const properties_t& properties) {
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

  void close() {}

  bool is_in_constraints(const fan::vec2i& position) {
    if (position.x >= map_size.x * tile_size.x * 2 || position.x < 0) return false;
    if (position.y >= map_size.y * tile_size.y * 2 || position.y < 0) return false;
    return true;
  }

  bool is_in_constraints(fan::vec2i& position, int j, int i) {
    position += -brush.size / 2 * tile_size * 2 + tile_size * 2 * fan::vec2(j, i);
    return is_in_constraints(position);
  }

  f32_t get_snapped_angle() {
    switch (fan::random::value_i64(0, 3)) {
      case 0: return 0;
      case 1: return fan::math::pi * 0.5;
      case 2: return fan::math::pi;
      case 3: return fan::math::pi * 1.5;
      default: return 0;
    }
  }

  fan::vec2 snap_to_tile_center(const fan::vec2& world_pos) {
    fan::vec2i grid_coord = ((world_pos + tile_size) / (tile_size * 2.f)).floor();
    return fan::vec2(grid_coord * (tile_size * 2.f));
  }

  bool apply_jitter(fan::vec2i& position) {
    if (!brush.jitter) return false;
    if (brush.jitter_chance <= fan::random::value_f32(0, 1)) return true;
    position += (fan::random::vec2i(-brush.jitter, brush.jitter) * 2 + 1) * tile_size + tile_size;
    return false;
  }

  sprite_t make_sprite(
    const fan::vec3& pos,
    const fan::vec2& size,
    const fan::color& color,
    render_view_t* rv,
    image_t image)
  {
    return sprite_t{ {
      .render_view = rv,
      .position = pos,
      .size = size,
      .color = color,
      .image = image
    } };
  }

  image_t get_marker_image(mesh_property_t type) {
    switch (type) {
      case mesh_property_t::player_spawn: return image_create(fan::color(0, 1, 0, 0.5));
      case mesh_property_t::enemy_spawn:  return image_create(fan::color(1, 0, 0, 0.5));
      default:                            return image_create(fan::color(1, 1, 0, 0.5));
    }
  }

  bool handle_tile_push(fan::vec2i& position, int& pj, int& pi) {
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

  bool handle_tile_erase(fan::vec2i& position, int& j, int& i) {
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

  bool mouse_to_grid(fan::vec2i& position) {
    if (window_relative_to_grid(fan::window::get_mouse_position(), &position)) {
      convert_draw_to_grid(position);
      position /= tile_size * 2;
      return true;
    }
    return false;
  }

  template <typename func_t>
  static void for_each_rect(const fan::vec2i& src, const fan::vec2i& dst, func_t&& f) {
    int y_start = std::min(src.y, dst.y);
    int y_end = std::max(src.y, dst.y);
    int x_start = std::min(src.x, dst.x);
    int x_end = std::max(src.x, dst.x);

    for (int j = y_start; j <= y_end; ++j) {
      for (int i = x_start; i <= x_end; ++i) {
        f(i, j);
      }
    }
  }

  bool physics_shape_exists(const fan::vec3& position, const fan::vec2& size) {
    auto found = physics_shapes.find(position.z);
    if (found == physics_shapes.end()) return false;
    for (auto& shape : found->second) {
      if (shape.visual.get_position() == position && shape.visual.get_size() == size) return true;
    }
    return false;
  }

  void invalidate_selection() {
    grid_visualize.highlight_selected.set_size(0);
    current_tile.layer = nullptr;
  }

  void handle_tile_action(fan::vec2i& position, auto action) {
    if (!window_relative_to_grid(fan::window::get_mouse_position(), &position)) {
      if (editor_settings.hovered && current_tile.layer != nullptr) invalidate_selection();
      return;
    }

    fan::vec2i grid_position = position;
    convert_draw_to_grid(grid_position);
    grid_position /= tile_size * 2;
    
    if (fan::vec3i(grid_position, brush.depth) == prev_grid_position) return;
    prev_grid_position = fan::vec3i(grid_position, brush.depth);

    for (int i = 0; i < brush.size.y; ++i) {
      for (int j = 0; j < brush.size.x; ++j) {
        if (action(position, j, i)) continue;
      }
    }
  }

  void open_texture_pack(const std::string& path) {
    if (texture_packs.empty()) texture_packs.resize(1);
    texture_packs[0]->open_compiled(path);
    texture_pack_images.clear();
    current_image_indices.clear();
    current_tile_images.clear();
    texture_pack_images.reserve(texture_packs[0]->size());
    texturepack_position_offset = 0;
    texturepack_size = 0;
    texturepack_single_image_size = 0;

    texture_packs[0]->iterate_loaded_images([this](auto& image) {
      tile_info_t ii;
      ii.ti.unique_id = image.unique_id;
      ii.ti.position = image.position;
      ii.ti.size = image.size;
      ii.ti.image = texture_packs[0]->get_pixel_data(image.unique_id).image;

      auto& img_data = fan::graphics::image_get_data(texture_packs[0]->get_pixel_data(image.unique_id).image);
      fan::vec2 size = img_data.size;

      texture_pack_images.push_back(ii);
      texturepack_size = texturepack_size.max(fan::vec2(size));
      texturepack_single_image_size = texturepack_single_image_size.max(fan::vec2(image.size));
    });
    if (texturepack_size.x > 0) {
      original_image_width = texturepack_size.x;
    }
  }

  void apply_brush_settings(const std::string& id, f32_t depth, const fan::vec2& size, const fan::color& color, const fan::vec3& angle = fan::vec3(0)) {
    brush.id = id;
    brush.depth = depth;
    brush.tile_size = size / tile_size;
    brush.color = color;
    brush.angle = angle;
  }

  void handle_pick_tile() {
    fan::vec2i position;
    if (!window_relative_to_grid(fan::window::get_mouse_position(), &position)) return;
    fan::vec2i grid_position = position;
    convert_draw_to_grid(grid_position);
    grid_position /= tile_size * 2;

    for (auto& depth_pair : spawn_marks) {
      for (auto& mark : depth_pair.second) {
        if (fan::math::d2::aabb_point_inside(position, mark.position, mark.size)) {
          current_image_indices.clear();
          current_tile_images.clear();
          apply_brush_settings(mark.id, mark.position.z, mark.size, mark.color);

          if (mark.type == mesh_property_t::player_spawn) brush.type = brush_t::type_e::player_spawn;
          else if (mark.type == mesh_property_t::enemy_spawn) brush.type = brush_t::type_e::enemy_spawn;
          else brush.type = brush_t::type_e::mark;
          return;
        }
      }
    }

    for (auto& depth_pair : physics_shapes) {
      for (auto& physics_shape : depth_pair.second) {
        if (fan::math::d2::aabb_point_inside(position, physics_shape.visual.get_position(), physics_shape.visual.get_size())) {
          current_image_indices.clear();
          current_tile_images.clear();
          auto& visual = physics_shape.visual;
          apply_brush_settings(physics_shape.id, visual.get_position().z, visual.get_size(), visual.get_color());
          brush.type = brush_t::type_e::physics_shape;
          brush.physics_type = physics_shape.type;
          brush.physics_body_type = physics_shape.body_type;
          brush.physics_draw = physics_shape.draw;
          brush.physics_shape_properties = physics_shape.shape_properties;
          return;
        }
      }
    }

    auto found = map_tiles.find(grid_position);
    if (found == map_tiles.end()) return;

    auto& layers = found->second.layers;
    std::uint32_t idx = find_layer_shape(layers);
    if (idx == invalid) idx = find_layer_shape(layers, true);
    if (idx == invalid || idx >= layers.size()) return;

    auto& layer = layers[idx];
    apply_brush_settings(layer.tile.id, layer.tile.position.z, layer.tile.size, layer.shape.get_color(), layer.shape.get_angle());

    std::uint16_t st = layer.shape.get_shape_type();
    if (st == (std::uint16_t)shape_type_t::sprite || st == (std::uint16_t)shape_type_t::unlit_sprite) {
      current_image_indices.clear();
      current_tile_images.clear();
      current_tile_images.resize(1);
      current_tile_images[0].push_back({.ti = layer.shape.get_tp()});
      brush.type = brush_t::type_e::texture;
    }
    else if (layer.tile.mesh_property == mesh_property_t::physics_shape) {
      current_image_indices.clear();
      current_tile_images.clear();
      brush.type = brush_t::type_e::physics_shape;
    }
    else if (layer.tile.mesh_property == mesh_property_t::light) {
      current_image_indices.clear();
      current_tile_images.clear();
      brush.type = brush_t::type_e::light;
    }
  }

  void handle_select_tile() {
    fan::vec2i position;
    if (window_relative_to_grid(fan::window::get_mouse_position(), &position)) {
      fan::vec2i grid_position = position;
      convert_draw_to_grid(grid_position);
      grid_position /= tile_size * 2;
      auto found = map_tiles.find(fan::vec2i(grid_position.x, grid_position.y));
      if (found != map_tiles.end()) {
        auto& layers = found->second.layers;
        std::uint32_t idx = find_layer_shape(layers);
        if ((idx != invalid || idx < brush.depth)) {
          current_tile.position = position;
          current_tile.layer = layers.data();
          current_tile.layer_index = idx;
          grid_visualize.highlight_selected.set_position(fan::vec2(position));
          grid_visualize.highlight_selected.set_size(tile_size);
        }
      }
    }
  }

  bool same_visual(const fte_t::shapes_t::global_t::layer_t& a, const fte_t::shapes_t::global_t::layer_t& b) {
    return
      a.tile.mesh_property == b.tile.mesh_property &&
      a.tile.id == b.tile.id &&
      a.tile.size == b.tile.size &&
      a.shape.get_angle() == b.shape.get_angle() &&
      a.shape.get_color() == b.shape.get_color() &&
      a.shape.get_flags() == b.shape.get_flags() &&
      a.shape.get_tp().unique_id == b.shape.get_tp().unique_id;
  }

  void fout(std::string filename) {
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

  void fin(const std::string& filename, const std::source_location& callers_path = std::source_location::current()) {
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

  #define editor OFFSETLESS(this, fte_t, terrain_generator)

  struct terrain_generator_t {
    terrain_generator_t() { init(); }

    void init() {
      ic.create_default(0.15f);
      tile_world.init();
      fan::vec2 map_size = tile_world.map_size;
      f32_t cell_size = tile_world.cell_size;

      rects.reserve(map_size.x * map_size.y);
      for (int y = 0; y < map_size.y; y++) {
        for (int x = 0; x < map_size.x; x++) {
          rects.push_back(sprite_t{ {
            .render_view = ic,
            .position = fan::vec3(fan::vec2(x, y) * cell_size * 2.f + cell_size, 0xFFFA),
            .size = fan::vec2(cell_size, cell_size),
            .image = tile_world_images.dirt
          } });
        }
      }

      visual_grid = grid_t{ {
        .render_view = ic,
        .position = fan::vec3(fan::vec2(map_size.x, map_size.y) * cell_size, 0xFFFA),
        .size = fan::vec2(map_size.x, map_size.y) * cell_size,
        .grid_size = fan::vec2(map_size.x, map_size.y) / 2.f,
        .color = fan::colors::black.set_alpha(0.4)
      } };

      ic.pan_with_middle_mouse = true;
      ic.set_position(tile_world.map_size * tile_world.cell_size);
      rebuild_colors();
    }

    void rebuild_colors() {
      for (int y = 0; y < tile_world.map_size.y; y++) {
        for (int x = 0; x < tile_world.map_size.x; x++) {
          rects[x + y * tile_world.map_size.x].set_image(
            tile_world.is_solid(x, y) ? tile_world_images.dirt : tile_world_images.background
          );
        }
      }
    }

    void iterate() {
      tile_world.iterate();
      rebuild_colors();
    }

    void reset() {
      tile_world.init_tile_world();
      rebuild_colors();
    }

    void render() {
      gui::set_next_window_bg_alpha(0);
      if (gui::begin("Terrain Generator", nullptr,
        gui::window_flags_no_background | gui::window_flags_no_focus_on_appearing |
        gui::window_flags_override_input)) 
      {
        {
          if (gui::button("Iterate")) iterate();
          if (gui::button("Reset")) reset();
          if (gui::button("Insert to map")) {
            gui::open_popup("Confirm Insert");
          }
        }
        {
          if (gui::is_popup_open("Confirm Insert")) {
            gui::set_next_window_pos(fan::window::get_size() / 2.0f, gui::cond_once, 0.5);
          }

          if (gui::begin_popup_modal("Confirm Insert", gui::window_flags_always_auto_resize)) {
            gui::text("Insert tiles into map at depth " + std::to_string((int)editor->brush.depth - shape_depths_t::max_layer_depth / 2) + "?", fan::colors::yellow);
            gui::text("It might overwrite tiles at the depth level.", fan::colors::yellow);

            if (gui::button("Cancel")) {
              gui::close_current_popup();
            }
            gui::same_line();
            if (gui::button("Confirm")) {
              insert_selected_tiles(editor->brush.depth);
              gui::close_current_popup();
            }
            gui::end_popup();
          }
        }
        {
          fan::vec2 need_init = !prev_viewport_size || prev_viewport_size != gui::get_window_size();
          gui::set_viewport(ic.render_view.viewport);
          if (need_init) ic.update();
          prev_viewport_size = gui::get_window_size();
        }
      }
      else {
        viewport_zero(ic.render_view.viewport);
      }
      gui::end();
    }

    void insert_selected_tiles(int depth) {
      fan::vec2i map_size = tile_world.map_size;
      if (editor->map_size.x < map_size.x || editor->map_size.y < map_size.y) {
        editor->map_size.x = std::max(editor->map_size.x, map_size.x);
        editor->map_size.y = std::max(editor->map_size.y, map_size.y);
        editor->resize_map();
      }

      for (int y = 0; y < map_size.y; ++y) {
        for (int x = 0; x < map_size.x; ++x) {
          fan::vec2i grid_pos(x, y);
          auto& layers = editor->map_tiles[grid_pos].layers;

          std::uint32_t idx = editor->find_layer_shape(layers);
          if (idx == fte_t::invalid) {
            layers.resize(layers.size() + 1);
            idx = layers.size() - 1;
          }

          auto& layer = layers[idx];
          layer.tile.position = fan::vec3(grid_pos * editor->tile_size * 2, depth);
          layer.tile.size = editor->tile_size;
          layer.tile.mesh_property = fte_t::mesh_property_t::none;

          image_t img = rects[x + y * map_size.x].get_image();
          if (img == tile_world_images.dirt) layer.tile.id = "##tile_world_dirt";
          else if (img == tile_world_images.background) layer.tile.id = "##tile_world_background";

          layer.shape = sprite_t{ {
            .render_view = editor->render_view,
            .position = layer.tile.position,
            .size = layer.tile.size,
            .image = img,
            .blending = true
          } };

          editor->visual_layers[depth].positions[grid_pos] = true;
        }
      }
    }

    tile_world_generator_t tile_world;
    std::vector<sprite_t> rects;
    grid_t visual_grid;
    interactive_camera_t ic;
    fan::vec2 prev_viewport_size = 0;
  }terrain_generator;

  #undef editor

  void render(); // implemented in ui module

  std::string file_name = "tilemap_editor.json";
  fan::vec2i map_size{6, 6};
  fan::vec2i tile_size{32, 32};
  current_tile_t current_tile;
  fan::vec2i current_tile_brush_count;
  std::vector<std::vector<tile_info_t>> current_tile_images;
  std::map<fan::vec2i, int, sort_by_y_t> current_image_indices;
  std::unordered_map<fan::vec2i, shapes_t::global_t> map_tiles;
  std::unordered_map<f32_t, std::vector<fte_t::physics_shapes_t>> physics_shapes;
  std::unordered_map<fan::vec3, visualize_t> visual_shapes;
  std::unordered_map<f32_t, std::vector<spawn_mark_t>> spawn_marks;
  std::map<std::uint16_t, visual_layer_t> visual_layers;
  fan::vec2 texturepack_position_offset{};
  fan::vec2 texturepack_size{};
  fan::vec2 texturepack_single_image_size{};
  int original_image_width = 2048;
  std::vector<tile_info_t> texture_pack_images;
  grid_visualize_t grid_visualize;
  brush_t brush;
  viewport_settings_t viewport_settings;
  editor_settings_t editor_settings;
  fan::vec3i prev_grid_position = 999999;
  image_t transparent_texture;
  fan::vec2i copy_buffer_region = 0;
  std::vector<shapes_t::global_t> copy_buffer;
  render_view_t* render_view = nullptr;
  std::function<void(int)> modify_cb = [](int) {};
  std::string previous_filename;
  shape_t visual_line;
  fan::window_t::buttons_handle_t buttons_handle;
  fan::window_t::keys_handle_t keys_handle;
  fan::window_t::mouse_move_handle_t mouse_move_handle;
  std::vector<texture_pack_t*> texture_packs;

  gui::content_browser_t content_browser {false};
};
#endif
#endif

#endif
