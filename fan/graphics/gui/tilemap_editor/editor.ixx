module;

#include <fan/utility.h>
#include <cstring>
#include <functional>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <source_location>
#include <sstream>
#include <fstream>

export module fan.graphics.gui.tilemap_editor.editor;

#if defined(fan_gui) && defined(fan_physics)

import fan.graphics.gui;

import fan.graphics.algorithm.raycast_grid;
import fan.types.color;
import fan.types.vector;
import fan.types.json;
import fan.print;
import fan.file_dialog;
import fan.io.file;
import fan.graphics;
import fan.random;
import fan.physics.b2_integration;
import fan.physics.collision.rectangle;
import fan.graphics.physics_shapes;

export struct fte_t {
  static constexpr int max_id_len = 48;
  static constexpr fan::vec2 default_button_size{100, 30};
  static constexpr fan::vec2 tile_viewer_sprite_size{64, 64};
  static constexpr fan::color highlighted_tile_color = fan::color(0.5, 0.5, 1);
  static constexpr fan::color highlighted_selected_tile_color = fan::color(0.5, 0, 0, 0.1);
  static constexpr f32_t scroll_speed = 1.2;
  static constexpr uint32_t invalid = -1;

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
        fan::graphics::shape_t shape;
      };
      std::vector<layer_t> layers;
    };
  };

  enum class event_type_e {
    none,
    add,
    remove
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

  struct vec3_hasher {
    std::size_t operator()(const fan::vec3& k) const {
      std::hash<f32_t> hasher;
      std::size_t hash_value = 17;
      hash_value = hash_value * 31 + hasher(k.x);
      hash_value = hash_value * 31 + hasher(k.y);
      hash_value = hash_value * 31 + hasher(k.z);
      return hash_value;
    }
  };

  struct sort_by_y_t {
    bool operator()(const fan::vec2i& a, const fan::vec2i& b) const {
      if (a.y == b.y) {
        return a.x < b.x;
      }
      else {
        return a.y < b.y;
      }
    }
  };

  struct tile_info_t {
    fan::graphics::texture_pack::ti_t ti;
    mesh_property_t mesh_property = mesh_property_t::none;
  };

  struct current_tile_t {
    fan::vec2i position = 0;
    shapes_t::global_t::layer_t* layer = nullptr;
    uint32_t layer_index;
  };

  struct visual_layer_t {
    std::unordered_map<fan::vec2i, bool, vec2i_hasher> positions;
    std::string text = "default";
    bool visible = true;
  };

  struct visualize_t {
    fan::graphics::shape_t shape;
  };

  struct brush_t {
    enum class mode_e : uint8_t {
      draw,
      copy
    };
    mode_e mode = mode_e::draw;
    fan::vec2 line_src = -9999999;
    static constexpr const char* mode_names[] = {"Draw", "Copy"};

    enum class type_e : uint8_t {
      texture,
      physics_shape = (uint8_t)fte_t::mesh_property_t::physics_shape,
      light
    };
    static constexpr const char* type_names[] = {"Texture", "Physics shape", "Light"};
    type_e type = type_e::texture;

    enum class dynamics_e : uint8_t {
      original,
      randomize
    };
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
    uint8_t physics_type = physics_shapes_t::type_e::box;
    static constexpr const char* physics_type_names[] = {"Box", "Circle"};
    uint8_t physics_body_type = fan::physics::body_type_e::static_body;
    static constexpr const char* physics_body_type_names[] = {"Static", "Kinematic", "Dynamic"};
    bool physics_draw = false;
    fan::physics::shape_properties_t physics_shape_properties;
  };

  struct viewport_settings_t {
    f32_t zoom = 1;
    bool move = false;
    fan::vec2 pos = 0;
    fan::vec2 size = 0;
    fan::vec2 offset = 0;
    fan::vec2 window_related_mouse_pos = 0;
    fan::vec2 zoom_offset = 0;
  };

  struct editor_settings_t {
    bool hovered = false;
  };

  struct grid_visualize_t {
    fan::graphics::shape_t background;
    fan::graphics::shape_t highlight_selected;
    fan::graphics::shape_t highlight_hover;
    fan::graphics::shape_t grid;
    fan::graphics::image_t highlight_color;
    fan::graphics::image_t collider_color;
    fan::graphics::image_t light_color;
    bool render_grid = true;
  };

  struct properties_t {
    fan::graphics::render_view_t* camera = nullptr;
  };

  struct layer_info_t {
    std::string layer_name;
    uint16_t depth;
  };

  uint32_t find_layer_shape(const auto& vec, bool top = false) {
    uint32_t found = invalid;
    int64_t depth = -1;
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
    resize_map();
  }

  bool window_relative_to_grid(const fan::vec2& window_relative_position, fan::vec2i* in) {
    auto camera_position = fan::graphics::camera_get_position(render_view->camera);
    fan::vec2 p = fan::graphics::translate_position(window_relative_position, render_view->viewport, render_view->camera) + camera_position;
    *in = ((p + tile_size) / (tile_size * 2)).floor() * (tile_size * 2);
    return fan_2d::collision::rectangle::point_inside_no_rotation(*in - map_size * tile_size / 2, map_size / 2 * tile_size - tile_size, map_size * tile_size);
  }

  void convert_draw_to_grid(fan::vec2i& p) {}
  void convert_grid_to_draw(fan::vec2i& p) {}

  void open(const properties_t& properties) {
    fan::graphics::image_load_properties_t lp;
    lp.visual_output = fan::graphics::image_sampler_address_mode::repeat;
    lp.min_filter = fan::graphics::image_filter::nearest;
    lp.mag_filter = fan::graphics::image_filter::nearest;

    if (properties.camera == nullptr) {
      render_view = &fan::graphics::get_orthographic_render_view();
    }
    else {
      render_view = properties.camera;
    }

    mouse_move_handle = fan::graphics::get_window().add_mouse_move_callback([this](const auto& d) {
      if (viewport_settings.move) {
        fan::vec2 move_off = (d.position - viewport_settings.offset) / viewport_settings.zoom;
        fan::graphics::camera_set_position(render_view->camera, viewport_settings.pos - move_off);
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

    buttons_handle = fan::graphics::get_window().add_buttons_callback([this](const auto& d) {
      if (d.button == fan::mouse_left && d.state == fan::mouse_state::release) {
        prev_grid_position = -999999;
      }

      if (!editor_settings.hovered && d.state != fan::mouse_state::release) {
        return;
      }

      f32_t old_zoom = viewport_settings.zoom;

      switch (d.button) {
        case fan::mouse_middle: {
          viewport_settings.move = (bool)d.state;
          viewport_settings.offset = fan::window::get_mouse_position();
          viewport_settings.pos = fan::graphics::camera_get_position(render_view->camera);
          break;
        }
        case fan::mouse_scroll_up: {
          if (fan::graphics::get_window().key_pressed(fan::key_left_control)) {
            brush.depth += 1;
            brush.depth = std::min((int)brush.depth, shape_depths_t::max_layer_depth);
          }
          else if (fan::graphics::get_window().key_pressed(fan::key_left_shift)) {
            brush.size += 1;
            grid_visualize.highlight_hover.set_size(tile_size * brush.size);
          }
          else {
            viewport_settings.zoom *= scroll_speed;
            auto& style = fan::graphics::gui::get_style();
            fan::vec2 pos = (fan::graphics::get_window().get_mouse_position() - viewport_settings.window_related_mouse_pos);
            pos /= fan::graphics::get_window().get_size();
            pos *= viewport_settings.size / 2;
            //viewport_settings.zoom_offset += pos / viewport_settings.zoom;
          }
          return;
        }
        case fan::mouse_scroll_down: {
          if (fan::graphics::get_window().key_pressed(fan::key_left_control)) {
            brush.depth -= 1;
            brush.depth = std::max((uint32_t)brush.depth, (uint32_t)1);
          }
          else if (fan::graphics::get_window().key_pressed(fan::key_left_shift)) {
            brush.size = (brush.size - 1).max(fan::vec2i(1));
            grid_visualize.highlight_hover.set_size(tile_size * brush.size);
          }
          else {
            viewport_settings.zoom /= scroll_speed;
          }
          return;
        }
        default: {
          return;
        }
      }
    });

    keys_handle = fan::graphics::get_window().add_keys_callback([this](const auto& d) {
      if (d.state != fan::keyboard_state::press) {
        return;
      }
      if (fan::graphics::gui::is_any_item_active()) {
        return;
      }

      switch (d.key) {
        case fan::key_r: {
          brush.angle.z = fmod(brush.angle.z + fan::math::pi / 2, fan::math::pi * 2);
          break;
        }
        case fan::key_delete: {
          if (fan::graphics::get_window().key_pressed(fan::key_left_control)) {
            reset_map();
          }
          break;
        }
        case fan::key_e: {
          break;
        }
      }
    });

    viewport_settings.size = 0;
    transparent_texture = fan::graphics::create_transparent_texture();

    grid_visualize.background = fan::graphics::sprite_t{{
      .render_view = render_view,
      .position = fan::vec3(tile_size * 2 * map_size / 2 - tile_size, 0),
      .size = 0,
      .image = transparent_texture,
    }};

    lp.min_filter = fan::graphics::image_filter::nearest;
    lp.mag_filter = fan::graphics::image_filter::nearest;

    grid_visualize.highlight_color = fan::graphics::image_load("images/highlight_hover.webp", lp);
    grid_visualize.collider_color = fan::graphics::image_create(fan::color(0, 0.5, 0, 0.5));
    grid_visualize.light_color = fan::graphics::image_load("images/lightbulb.webp", lp);

    grid_visualize.highlight_hover = fan::graphics::unlit_sprite_t{{
      .render_view = render_view,
      .position = fan::vec3(viewport_settings.pos, shape_depths_t::cursor_highlight_depth),
      .size = tile_size,
      .image = grid_visualize.highlight_color,
      .blending = true
    }};

    grid_visualize.highlight_selected = fan::graphics::unlit_sprite_t{{
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

    fan::graphics::camera_set_position(render_view->camera, viewport_settings.pos + tile_size * 2.f * map_size / 2.f);

    fan::graphics::shapes::grid_t::properties_t gp;
    gp.viewport = render_view->viewport;
    gp.camera = render_view->camera;
    gp.position = fan::vec3(map_size * (tile_size * 2.f) / 2.f - tile_size, shape_depths_t::cursor_highlight_depth - 1);
    gp.size = 0;
    gp.color = fan::colors::black.set_alpha(0.4);
    grid_visualize.grid = gp;
    resize_map();

    visual_line = fan::graphics::line_t{{
      .render_view = render_view,
      .src = fan::vec3(0, 0, shape_depths_t::cursor_highlight_depth + 1),
      .dst = fan::vec2(400),
      .color = fan::colors::white
    }};
  }

  void close() {}

  bool is_in_constraints(const fan::vec2i& position) {
    if (position.x >= map_size.x * tile_size.x * 2 || position.x < 0) {
      return false;
    }
    if (position.y >= map_size.y * tile_size.y * 2 || position.y < 0) {
      return false;
    }
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
      default: {
        fan::throw_error("");
        return 0;
      }
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

  fan::graphics::sprite_t make_sprite(
    const fan::vec3& pos,
    const fan::vec2& size,
    const fan::color& color,
    fan::graphics::render_view_t* rv,
    fan::graphics::image_t image)
  {
    return fan::graphics::sprite_t{ {
      .render_view = rv,
      .position = pos,
      .size = size,
      .color = color,
      .image = image
    } };
  }



  bool handle_tile_push(fan::vec2i& position, int& pj, int& pi) {
    if (apply_jitter(position)) {
      return true;
    }

    if (!is_in_constraints(position, pj, pi)) {
      return true;
    }

    f32_t inital_x = position.x;
    fan::vec2i grid_position = position;
    convert_draw_to_grid(grid_position);
    brush.line_src = snap_to_tile_center(fan::graphics::get_mouse_position(render_view->camera, render_view->viewport));
    grid_position /= (tile_size * 2);
    auto& layers = map_tiles[grid_position].layers;
    visual_layers[brush.depth].positions[grid_position] = 1;
    uint32_t idx = find_layer_shape(layers);

    if (idx == invalid && (brush.type == brush_t::type_e::light)) {
      layers.resize(layers.size() + 1);
      layers.back().tile.position = fan::vec3(position, brush.depth);
      layers.back().tile.size = tile_size * brush.tile_size;

      switch (brush.type) {
        case brush_t::type_e::light: {
          fan::vec3 pos = fan::vec3(position, brush.depth);
          auto& shape = layers.back().shape;
          layers.back().tile.id = brush.id;
          layers.back().shape = fan::graphics::light_t{{
            .render_view = render_view,
            .position = pos,
            .size = tile_size * brush.tile_size,
            .color = brush.dynamics_color == brush_t::dynamics_e::randomize ? fan::random::color() : brush.color
          }};
          layers.back().tile.mesh_property = mesh_property_t::light;
          visual_shapes[pos].shape = make_sprite(
            fan::vec3(fan::vec2(pos), pos.z + 1),
            tile_size,
            fan::color(1),
            render_view,
            grid_visualize.light_color
          );
          break;
        }
        default: {
          break;
        }
      }
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

        switch (brush.type) {
        case brush_t::type_e::light: {
          auto& shape = layer.shape;
          fan::vec3 pos = shape.get_position();

          layer.shape = fan::graphics::light_t{ {
            .render_view = render_view,
            .position = pos,
            .size = shape.get_size(),
            .color = shape.get_color(),
            .angle = shape.get_angle(),
          } };
          layer.tile.mesh_property = mesh_property_t::light;

          visual_shapes[pos].shape = make_sprite(
            fan::vec3(fan::vec2(pos), pos.z + 1),
            tile_size,
            fan::color(1),
            render_view,
            grid_visualize.light_color
          );
          break;
        }
        default: {
          break;
        }
        }

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
            layers.back().shape = fan::graphics::sprite_t{{
              .render_view = render_view,
              .position = fan::vec3(position, brush.depth),
              .size = tile_size * brush.tile_size,
              .angle = brush.dynamics_angle == brush_t::dynamics_e::randomize ? fan::vec3(0, 0, get_snapped_angle()) : brush.angle,
              .color = brush.dynamics_color == brush_t::dynamics_e::randomize ? fan::random::color() : brush.color,
              .blending = true
            }};
          }

          switch (brush.type) {
            case brush_t::type_e::texture: {
              if (layers.back().shape.set_tp(&tile.ti)) {
                fan::print("failed to load image");
              }
              break;
            }
            default: {
              break;
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

              switch (brush.dynamics_angle) {
                case brush_t::dynamics_e::original: {
                  if (layer.shape.get_angle() != brush.angle) {
                    layer.shape.set_angle(brush.angle);
                  }
                  break;
                }
                default: {
                  break;
                }
              }

              layer.tile.size = tile_size * brush.tile_size;
              layer.shape.set_size(tile_size * brush.tile_size);
              layer.shape.set_color(brush.color);
              layer.tile.id = brush.id;

              switch (brush.type) {
                case brush_t::type_e::texture: {
                  layer.shape = fan::graphics::sprite_t{{
                    .render_view = render_view,
                    .position = layer.shape.get_position(),
                    .size = layer.shape.get_size(),
                    .angle = layer.shape.get_angle(),
                    .color = layer.shape.get_color()
                  }};

                  if (layer.shape.set_tp(&tile.ti)) {
                    fan::print("failed to load image");
                  }
                  layer.tile.mesh_property = mesh_property_t::none;
                  break;
                }
                default: {
                  break;
                }
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
    if (!is_in_constraints(position, j, i)) {
      return true;
    }

    fan::vec2i grid_position = position;
    convert_draw_to_grid(grid_position);
    grid_position /= tile_size * 2;

    auto found = physics_shapes.find(brush.depth);
    if (found != physics_shapes.end()) {
      for (auto it = found->second.begin(); it != found->second.end(); ++it) {
        if (fan_2d::collision::rectangle::point_inside_no_rotation(position, it->visual.get_position(), it->visual.get_size())) {
          it = found->second.erase(it);
          return false;
        }
      }
    }

    if (apply_jitter(position)) {
      return true;
    }
    
    auto found_tile = map_tiles.find(grid_position);
    if (found_tile != map_tiles.end()) {
      auto& layers = found_tile->second.layers;
      uint32_t idx = find_layer_shape(layers);

      if (idx != invalid || idx < layers.size()) {
        switch (layers[idx].tile.mesh_property) {
          case mesh_property_t::light: {
            fan::vec3 erase_position = layers[idx].shape.get_position();
            auto found_visual = visual_shapes.find(erase_position);
            if (found_visual != visual_shapes.end()) {
              visual_shapes.erase(found_visual);
            }
            break;
          }
          default: {
            break;
          }
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


  void handle_tile_brush() {
    if (!editor_settings.hovered) {
      return;
    }

    static std::vector<fan::graphics::shape_t> select;
    static fan::vec2i copy_src;
    static fan::vec2i copy_dst;

    bool is_mouse_left_clicked = fan::window::is_mouse_clicked();
    bool is_mouse_left_down = fan::window::is_mouse_down();
    bool is_mouse_left_released = fan::window::is_mouse_released();
    bool is_mouse_right_clicked = fan::window::is_mouse_clicked(fan::mouse_right);
    bool is_mouse_right_down = fan::window::is_mouse_down(fan::mouse_right);

    static auto handle_select_tiles = [this] {
      select.clear();
      fan::vec2i mouse_grid_pos;
      if (mouse_to_grid(mouse_grid_pos)) {
        fan::vec2i src = copy_src;
        fan::vec2i dst = mouse_grid_pos;
        copy_dst = dst;
        int stepx = (src.x <= dst.x) ? 1 : -1;
        int stepy = (src.y <= dst.y) ? 1 : -1;
        for_each_rect(src, dst, [&](int i, int j) {
          select.push_back(fan::graphics::unlit_sprite_t{ {
            .render_view = render_view,
            .position = fan::vec3(fan::vec2(i, j) * tile_size * 2, shape_depths_t::cursor_highlight_depth),
            .size = tile_size,
            .image = grid_visualize.highlight_color,
            .blending = true
          } });
        });
      }
    };

    switch (brush.mode) {
      case brush_t::mode_e::draw: {
        switch (brush.type) {
          case brush_t::type_e::physics_shape: {
            if (is_mouse_left_clicked) {
              select.clear();
              fan::vec2i mouse_grid_pos;
              if (mouse_to_grid(mouse_grid_pos)) {
                copy_src = mouse_grid_pos;
              }
            }
            if (is_mouse_left_down) {
              handle_select_tiles();
            }
            if (is_mouse_left_released) {
              select.clear();
              fan::vec2i mouse_grid_pos;
              if (mouse_to_grid(mouse_grid_pos)) {
                auto& layers = physics_shapes[brush.depth];
                layers.push_back({
                  .visual = fan::graphics::sprite_t{{
                    .render_view = render_view,
                    .position = fan::vec3(((fan::vec2(copy_src) + (fan::vec2(copy_dst) - fan::vec2(copy_src)) / 2) + brush.offset/2) * tile_size * 2, brush.depth),
                    .size = (((copy_dst - copy_src) + fan::vec2(1, 1)) * fan::vec2(tile_size)).abs() * brush.tile_size,
                    .image = grid_visualize.collider_color,
                    .blending = true
                  }},
                  .type = brush.physics_type,
                  .body_type = brush.physics_body_type,
                  .draw = brush.physics_draw,
                  .shape_properties = brush.physics_shape_properties,
                  .id = brush.id
                });
              }
            }
            if (!(is_mouse_right_clicked || is_mouse_right_down)) {
              return;
            }
            break;
          }
          default: {
            break;
          }
        }

        fan::vec2i position;
        bool is_ctrl_pressed = fan::graphics::get_window().key_pressed(fan::key_left_control);
        bool is_shift_pressed = fan::graphics::get_window().key_pressed(fan::key_left_shift);

        if (is_mouse_left_down && !is_ctrl_pressed && 
          !is_shift_pressed && !fan::window::is_key_down(fan::key_t) && !fan::window::is_key_down(fan::key_5)) {
          handle_tile_action(position, [this](auto...args) {
            return handle_tile_push(args...);
          });
        }

        if (is_mouse_right_down) {
          handle_tile_action(position, [this](auto...args) {
            return handle_tile_erase(args...);
          });
        }
        break;
      }
      case brush_t::mode_e::copy: {
        if (is_mouse_left_clicked) {
          select.clear();
          copy_buffer.clear();
          fan::vec2i mouse_grid_pos;
          if (mouse_to_grid(mouse_grid_pos)) {
            copy_src = mouse_grid_pos;
          }
        }
        if (is_mouse_left_down) {
          handle_select_tiles();
        }
        if (is_mouse_left_released) {
          select.clear();
          fan::vec2i mouse_grid_pos;
          if (mouse_to_grid(mouse_grid_pos)) {
            fan::vec2 src = copy_src;
            fan::vec2 dst = copy_dst;
            int stepx = (src.x <= dst.x) ? 1 : -1;
            int stepy = (src.y <= dst.y) ? 1 : -1;
            copy_buffer_region.x = std::max(1.f, std::abs((dst.x + stepx) - src.x));
            copy_buffer_region.y = std::max(1.f, std::abs((dst.y + stepy) - src.y));

            if (src == dst) {
              auto found = map_tiles.find(copy_src);
              if (found != map_tiles.end()) {
                copy_buffer.push_back(found->second);
                for (auto& i : copy_buffer.back().layers) {
                  i.shape.set_size(0);
                }
              }
            }
            else {
              for_each_rect(src, dst, [&](int i, int j) {
                auto found = map_tiles.find(fan::vec2i(i, j));
                if (found != map_tiles.end()) {
                  copy_buffer.push_back(found->second);
                  for (auto& layer : copy_buffer.back().layers) {
                    layer.shape.set_size(0);
                  }
                }
                else {
                  copy_buffer.push_back({});
                }
              });
            }
          }
        }

        if (is_mouse_right_clicked) {
          fan::vec2i mouse_grid_pos;
          if (mouse_to_grid(mouse_grid_pos)) {
            int index = 0;
            for (auto& i : copy_buffer) {
              fan::vec2i current_pos = mouse_grid_pos + fan::vec2i(index % copy_buffer_region.x, index / copy_buffer_region.x);
              if (is_in_constraints(current_pos * tile_size * 2)) {
                auto& tile = map_tiles[current_pos];
                tile = i;
                for (std::size_t k = 0; k < tile.layers.size(); ++k) {
                  auto& t = tile.layers[k];
                  fan::vec2 op = t.shape.get_position();
                  fan::vec2 offset = op - fan::vec2(t.tile.position) * tile_size * 2 - tile_size;
                  fan::vec2 draw_pos = current_pos * tile_size * 2 + tile_size + offset;

                  if (tile.layers[k].tile.position.z != brush.depth) {
                    t.shape.set_position(fan::vec3(fan::vec2(draw_pos), t.tile.position.z));
                    t.shape.set_size(t.tile.size);
                    continue;
                  }
                  if (is_in_constraints(draw_pos)) {
                    visual_layers[t.tile.position.z].positions[current_pos] = 1;
                    t.shape.set_position(fan::vec3(fan::vec2(draw_pos), t.tile.position.z));
                    t.shape.set_size(t.tile.size);
                    switch (t.tile.mesh_property) {
                      case mesh_property_t::light: {
                        visual_shapes[fan::vec3(draw_pos, brush.depth)].shape = fan::graphics::sprite_t{{
                          .render_view = render_view,
                          .position = fan::vec3(draw_pos, brush.depth + 1),
                          .size = tile.layers[k].tile.size,
                          .image = grid_visualize.light_color,
                          .blending = true
                        }};
                        break;
                      }
                      default: {
                        break;
                      }
                    }
                  }
                }
              }
              index++;
            }
          }
        }
        break;
      }
    }
  }

  void invalidate_selection() {
    grid_visualize.highlight_selected.set_size(0);
    current_tile.layer = nullptr;
  }

  void handle_tile_action(fan::vec2i& position, auto action) {
    if (!window_relative_to_grid(fan::window::get_mouse_position(), &position)) {
      if (editor_settings.hovered && current_tile.layer != nullptr) {
        invalidate_selection();
      }
      return;
    }

    fan::vec2i grid_position = position;
    convert_draw_to_grid(grid_position);
    grid_position /= tile_size * 2;
    
    if (fan::vec3i(grid_position, brush.depth) == prev_grid_position) {
      return;
    }
    prev_grid_position = fan::vec3i(grid_position, brush.depth);

    for (int i = 0; i < brush.size.y; ++i) {
      for (int j = 0; j < brush.size.x; ++j) {
        if (action(position, j, i)) {
          continue;
        }
      }
    }
  }

  void open_texture_pack(const std::string& path) {
    // TODO make multi tps
    if (texture_packs.empty()) {
      texture_packs.resize(1);
    }
    texture_packs[0]->open_compiled(path);
    texture_pack_images.clear();
    texture_pack_images.reserve(texture_packs[0]->size());

    texture_packs[0]->iterate_loaded_images([this](auto& image) {
      tile_info_t ii;
      ii.ti = fan::graphics::texture_pack::ti_t{
        .unique_id = image.unique_id,
        .position = image.position,
        .size = image.size,
        .image = texture_packs[0]->get_pixel_data(image.unique_id).image
      };

      auto& img_data = fan::graphics::image_get_data(texture_packs[0]->get_pixel_data(image.unique_id).image);
      fan::vec2 size = img_data.size;

      texture_pack_images.push_back(ii);
      texturepack_size = texturepack_size.max(fan::vec2(size));
      texturepack_single_image_size = texturepack_single_image_size.max(fan::vec2(image.size));
    });
  }

  fan::vec2 snap_line_to_angle(const fan::vec2& start, const fan::vec2& end, f32_t snap_increment = 45.0f) {
    fan::vec2 direction = end - start;
    f32_t length = direction.length();

    if (length < 1.0f) {
      return end;
    }

    f32_t current_angle = fan::math::degrees(std::atan2(direction.y, direction.x));
    f32_t snapped_angle = std::round(current_angle / snap_increment) * snap_increment;
    f32_t snapped_radians = fan::math::radians(snapped_angle);
    fan::vec2 snapped_direction = fan::vec2(std::cos(snapped_radians), std::sin(snapped_radians));

    return start + snapped_direction * length;
  }

  bool handle_editor_window(fan::vec2& editor_size) {
    if (fan::graphics::gui::begin_main_menu_bar()) {
      static std::string fn;
      if (fan::graphics::gui::begin_menu("File")) {
        if (fan::graphics::gui::menu_item("Open..")) {
          open_file_dialog.load("fte,json", &fn);
        }
        if (fan::graphics::gui::menu_item("Save")) {
          fout(previous_file_name);
        }
        if (fan::graphics::gui::menu_item("Save as")) {
          save_file_dialog.save("fte,json", &fn);
        }
        if (fan::graphics::gui::menu_item("Quit")) {
          fan::graphics::gui::end();
        }
        fan::graphics::gui::end_menu();
      }

      if (fan::graphics::gui::begin_menu("Texture Pack")) {
        if (fan::graphics::gui::menu_item("Open..")) {
          open_tp_dialog.load("ftp", &fn);
        }
        fan::graphics::gui::end_menu();
      }

      if (open_file_dialog.is_finished()) {
        if (fn.size() != 0) {
          fin(fn);
          fn.clear();
        }
        open_file_dialog.finished = false;
      }
      if (save_file_dialog.is_finished()) {
        if (fn.size() != 0) {
          fout(fn);
          fn.clear();
        }
        save_file_dialog.finished = false;
      }
      if (open_tp_dialog.is_finished()) {
        if (fn.size() != 0) {
          open_texture_pack(fn);
          fn.clear();
        }
        open_tp_dialog.finished = false;
      }

      fan::graphics::gui::end_main_menu_bar();

      fan::vec2 initial_pos = fan::graphics::gui::get_cursor_screen_pos();
      fan::graphics::gui::push_style_color(fan::graphics::gui::col_window_bg, fan::color(0, 0, 0, 0));
      fan::graphics::gui::begin("Tilemap Editor2", nullptr);
      fan::graphics::gui::pop_style_color();

      fan::vec2 viewport_size = fan::graphics::gui::get_content_region_avail();
      auto& style = fan::graphics::gui::get_style();
      fan::vec2 frame_padding = style.FramePadding;
      fan::vec2 viewport_pos = fan::graphics::gui::get_window_content_region_min() - frame_padding;
      fan::vec2 real_viewport_size = viewport_size + frame_padding * 2 + fan::vec2(0, style.WindowPadding.y * 2);
      real_viewport_size.x = std::clamp(real_viewport_size.x, 1.f, real_viewport_size.x);
      real_viewport_size.y = std::clamp(real_viewport_size.y, 1.f, real_viewport_size.y);

      fan::graphics::camera_set_ortho(
        render_view->camera,
        (fan::vec2(-real_viewport_size.x / 2, real_viewport_size.x / 2) / viewport_settings.zoom) + viewport_settings.zoom_offset.x,
        (fan::vec2(-real_viewport_size.y / 2, real_viewport_size.y / 2) / viewport_settings.zoom) + viewport_settings.zoom_offset.y
      );

      fan::graphics::viewport_set(
        render_view->viewport, 
        viewport_pos + fan::vec2(0, style.WindowPadding.y * 2),
        real_viewport_size
      );
      editor_size = real_viewport_size;
      viewport_settings.size = viewport_size;

      static int init = 0;
      if (init == 0) {
        init = 1;
      }

      viewport_settings.window_related_mouse_pos = fan::vec2(fan::vec2(fan::graphics::gui::get_window_pos()) + fan::vec2(fan::graphics::gui::get_window_size() / 2) + fan::vec2(0, style.WindowPadding.y * 2 - frame_padding.y * 2));

      fan::graphics::gui::set_window_font_scale(1.5);
      fan::graphics::gui::text("brush type: "_str + brush.type_names[(uint8_t)brush.type]);
      fan::graphics::gui::text("brush depth: " + std::to_string((int)brush.depth - shape_depths_t::max_layer_depth / 2));

      fan::vec2 prev_item_spacing = style.ItemSpacing;
      style.ItemSpacing = fan::vec2(0);
      fan::vec2 old_cursorpos = fan::graphics::gui::get_cursor_pos();
      fan::vec2 draw_start = fan::graphics::gui::get_mouse_pos();
      fan::vec2 cursor_pos = 0;
      cursor_pos.y = -(tile_viewer_sprite_size.y / std::max(1.f, current_tile_brush_count.y / 5.f));
      fan::graphics::gui::set_cursor_screen_pos(draw_start);

      for (auto& i : current_tile_images) {
        int idx = 0;
        for (auto& j : i) {
          if (idx != 0) {
            cursor_pos.x += tile_viewer_sprite_size.x / std::max(1.f, current_tile_brush_count.x / 5.f);
            fan::graphics::gui::same_line();
          }
          else {
            cursor_pos.y += (tile_viewer_sprite_size.y / std::max(1.f, current_tile_brush_count.y / 5.f));
            cursor_pos.x = 0;
            fan::graphics::gui::set_cursor_screen_pos(fan::vec2(draw_start.x, fan::graphics::gui::get_cursor_screen_pos().y));
          }
          idx++;
          
          auto& img_data = fan::graphics::image_get_data(j.ti.image);
          fan::vec2 size = img_data.size;

          fan::graphics::gui::image_rotated(
            j.ti.image,
            (tile_viewer_sprite_size / std::max(1.f, current_tile_brush_count.x / 5.f)) * viewport_settings.zoom,
            360.f - fan::math::degrees(brush.angle.z),
            j.ti.position / size,
            j.ti.position / size + j.ti.size / size,
            fan::color(1, 1, 1, 0.9)
          );
        }
      }

      fan::graphics::gui::set_cursor_pos(old_cursorpos);
      style.ItemSpacing = prev_item_spacing;

      fan::vec2 cursor_position = fan::window::get_mouse_position();
      fan::vec2i grid_pos;
      if (window_relative_to_grid(cursor_position, &grid_pos)) {
        auto str = (grid_pos / (tile_size * 2.f)).to_string();
        fan::graphics::gui::text_bottom_right(str.c_str(), 1);
        str = grid_pos.to_string();
        fan::graphics::gui::text_bottom_right(str.c_str(), 0);
      }

      if (fan::graphics::gui::begin("Layer window")) {
        for (auto& layer_pair : visual_layers) {
          auto& layer = layer_pair.second; 
          layer.text.resize(32);
          uint16_t depth = layer_pair.first;
          auto fmt = ("Layer " + std::to_string(depth - shape_depths_t::max_layer_depth / 2));

          if (fan::graphics::gui::toggle_button(("Visible " + fmt).c_str(), &layer.visible)) {
            static auto iterate_positions = [&](auto l) {
              auto visual_found = visual_layers.find(depth);
              if (visual_found == visual_layers.end()) {
                fan::throw_error("some weird bugs");
              }
              auto& position_map = visual_found->second.positions;
              for (auto& position_pair : position_map) {
                auto& position = position_pair.first;
                auto tiles_found = map_tiles.find(position);
                if (tiles_found == map_tiles.end()) {
                  fan::throw_error("more some weird bugs");
                }
                for (auto& tile_layer : tiles_found->second.layers) {
                  if (tile_layer.tile.position.z == depth) {
                    l(tile_layer);
                  }
                }
              }
            };

            if (layer.visible == false) {
              iterate_positions([&](fte_t::shapes_t::global_t::layer_t& layer) {
                auto vs_found = visual_shapes.find(layer.tile.position);
                if (layer.tile.mesh_property != fte_t::mesh_property_t::none && vs_found != visual_shapes.end()) {
                  vs_found->second.shape.set_size(0);
                }
                layer.shape.set_size(0);
              });
            }
            else {
              iterate_positions([&](fte_t::shapes_t::global_t::layer_t& layer) {
                auto vs_found = visual_shapes.find(layer.tile.position);
                if (layer.tile.mesh_property != fte_t::mesh_property_t::none && vs_found != visual_shapes.end()) {
                  vs_found->second.shape.set_size(layer.tile.size);
                }
                layer.shape.set_size(layer.tile.size);
              });
            }
            modify_cb(0);
          }
          fan::graphics::gui::same_line();
          fan::graphics::gui::input_text(fmt, &layer.text);
        }
      }
      fan::graphics::gui::end();
    }
    else {
      fan::graphics::viewport_zero(render_view->viewport);
      return true;
    }

    editor_settings.hovered = fan::graphics::gui::is_window_hovered();
    fan::graphics::gui::end();

    if (fan::window::is_key_pressed(fan::key_s) && fan::window::is_key_down(fan::key_left_control)) {
      fout(previous_file_name);
    }

    if (fan::graphics::get_window().key_state(fan::key_shift) != -1) {
      fan::vec2 line_dst = fan::graphics::get_mouse_position(render_view->camera, render_view->viewport);
      bool control_pressed = fan::graphics::get_window().key_pressed(fan::key_left_control);

      if (control_pressed) {
        line_dst = snap_line_to_angle(brush.line_src, line_dst, 45.0f);
      }

      visual_line.set_line(brush.line_src, line_dst);

      if (fan::graphics::get_window().key_state(fan::mouse_left) == 1) {
        brush.line_src = ((brush.line_src + tile_size) / (tile_size * 2)).floor() * tile_size * 2;
        fan::vec2 final_dst = line_dst;
        final_dst = ((final_dst + tile_size) / (tile_size * 2)).floor() * tile_size * 2;

        if (final_dst.x - brush.line_src.x > tile_size.x * 2) {
          final_dst.x += tile_size.x * 2;
        }
        if (final_dst.y - brush.line_src.y > tile_size.y * 2) {
          final_dst.y += tile_size.y * 2;
        }

        fan::vec2 raycast_dst = control_pressed ? line_dst : final_dst;
        f32_t divider = 2.0001;
        f32_t aim_angle = fan::math::degrees(fan::math::aim_angle(brush.line_src, final_dst));

        if (brush.line_src.y > final_dst.y && brush.line_src.x < final_dst.x) {
          divider = 1.9999;
        }
        else if (brush.line_src.y < final_dst.y && brush.line_src.x > final_dst.x) {
          divider = 1.9999;
        }

        std::vector<fan::vec2i> raycast_positions = fan::graphics::algorithm::grid_raycast(
          {brush.line_src / 2 + tile_size / divider, raycast_dst / 2 + tile_size / divider}, tile_size
        );

        for (fan::vec2i& pos : raycast_positions) {
          fan::vec2i p = pos * (tile_size * 2);
          for (int i = 0; i < brush.size.y; ++i) {
            for (int j = 0; j < brush.size.x; ++j) {
              fan::vec2i tile_pos = p + fan::vec2i(j * tile_size.x, i * tile_size.y);
              handle_tile_push(tile_pos, i, j);
            }
          }
        }
      }
    }
    else {
      visual_line.set_line(-999999999, -999999999);
    }
    return false;
  }

  bool handle_editor_settings_window() {
    if (fan::graphics::gui::begin("Editor settings")) {
      if (fan::graphics::gui::input_int("map size", &map_size)) {
        resize_map();
      }
      if (fan::graphics::gui::input_int("tile size", &tile_size)) {
        resize_map();
        for (auto& i : map_tiles) {
          for (auto& j : i.second.layers) {
            fan::vec2 s = j.shape.get_size();
            fan::vec2 sp = fan::vec2(j.shape.get_position());
            fan::vec2 p = tile_size * ((sp / s));
            j.shape.set_position(p);
            j.shape.set_size(tile_size);
          }
        }
      }

      if (fan::graphics::gui::checkbox("render grid", &grid_visualize.render_grid)) {
        if (grid_visualize.render_grid) {
          grid_visualize.grid.set_size(map_size * (tile_size * 2.f) / 2.f);
        }
        else {
          grid_visualize.grid.set_size(0);
        }
      }
    }
    fan::graphics::gui::end();
    return false;
  }

  void handle_tiles_window() {
    static f32_t zoom = 1;
    fan::graphics::gui::push_style_color(fan::graphics::gui::col_button, fan::color(0.f, 0.f, 0.f, 0.f));
    fan::graphics::gui::push_style_color(fan::graphics::gui::col_button_active, fan::color(0.f, 0.f, 0.f, 0.f));
    fan::graphics::gui::push_style_color(fan::graphics::gui::col_button_hovered, fan::color(0.f, 0.f, 0.f, 0.f));
    fan::graphics::gui::push_style_var(fan::graphics::gui::style_var_frame_padding, fan::vec2(0, 0));
    fan::graphics::gui::push_style_color(fan::graphics::gui::col_button, fan::color::rgb(31, 31, 31));
    fan::graphics::gui::push_style_color(fan::graphics::gui::col_window_bg, fan::color::rgb(31, 31, 31));

    static bool init = true;
    if (init) {
      fan::graphics::gui::set_next_window_focus();
      init = false;
    }

    if (fan::graphics::gui::begin("tiles", nullptr, fan::graphics::gui::window_flags_no_scroll_with_mouse)) {
      if (fan::graphics::gui::is_window_hovered()) {
        zoom += fan::graphics::gui::get_io().MouseWheel / 3.f;
      }

      if (fan::graphics::gui::is_window_hovered() && fan::graphics::gui::is_mouse_dragging(fan::mouse_middle)) {
        fan::vec2 mouse_delta = fan::graphics::gui::get_mouse_drag_delta(fan::mouse_middle);
        fan::graphics::gui::reset_mouse_drag_delta(fan::mouse_middle);
        fan::graphics::gui::set_scroll_x(fan::graphics::gui::get_scroll_x() - mouse_delta.x);
        fan::graphics::gui::set_scroll_y(fan::graphics::gui::get_scroll_y() - mouse_delta.y);
      }

      f32_t x_size = fan::graphics::gui::get_content_region_avail().x;
      f32_t y_size = fan::graphics::gui::get_content_region_avail().y;
      fan::graphics::gui::drag("original image width", &original_image_width, 1, 0, 1000);

      auto& style = fan::graphics::gui::get_style();
      fan::vec2 prev_item_spacing = style.ItemSpacing;
      style.ItemSpacing = fan::vec2(0);
      current_tile_brush_count = 0;

      int total_images = texture_pack_images.size();
      int images_per_row = (original_image_width / (texturepack_single_image_size.x));

      if (images_per_row == 0) {
        fan::graphics::gui::pop_style_color(5);
        fan::graphics::gui::pop_style_var();
        return;
      }

      int rows_needed = (total_images + images_per_row - 1) / images_per_row == 0 ? 1 : images_per_row;
      float image_width = x_size / images_per_row;
      float image_height = y_size / rows_needed;
      float final_image_size = std::max(image_width, image_height);

      static fan::vec2 selection_start(-1, -1);
      static fan::vec2 selection_end(-1, -1);
      static fan::vec2 min_rect = (uint32_t)~0;
      static fan::vec2 max_rect = -1;
      static fan::vec2 min_rect_draw = (uint32_t)~0;
      static fan::vec2 max_rect_draw = -1;
      static bool is_selecting = false;

      bool is_left_mouse_button_clicked = fan::window::is_mouse_clicked(0);
      bool is_left_mouse_drag = fan::window::is_mouse_down(0) && fan::graphics::gui::is_mouse_dragging(0);
      bool is_right_mouse_button_clicked = fan::window::is_mouse_clicked(1);
      bool is_right_mouse_drag = fan::window::is_mouse_down(1) && fan::graphics::gui::is_mouse_dragging(1);
      bool is_left_ctrl_key_pressed = fan::window::is_key_down(fan::key_left_control);
      bool is_left_mouse_button_released = fan::window::is_mouse_released(0);

      fan::vec2 sprite_size;
      fan::vec2 initial_pos = fan::graphics::gui::get_cursor_screen_pos();
      auto* draw_list = fan::graphics::gui::get_window_draw_list();

      for (uint32_t i = 0; i < texture_pack_images.size(); i++) {
        auto& node = texture_pack_images[i];
        fan::vec2i grid_index(i % images_per_row, i / images_per_row);

        fan::vec2 cursor_pos_global = fan::graphics::gui::get_cursor_screen_pos();
        sprite_size = fan::vec2(final_image_size * zoom);
        auto& img_data = fan::graphics::image_get_data(node.ti.image);
        fan::vec2 size = img_data.size;

        fan::graphics::gui::image_button(
          (std::string("##ibutton") + std::to_string(i)).c_str(),
          node.ti.image,
          sprite_size,
          node.ti.position / size,
          node.ti.position / size + node.ti.size / size
        );

        if (current_image_indices.find(grid_index) != current_image_indices.end()) {
          draw_list->AddRect(cursor_pos_global, cursor_pos_global + sprite_size, 0xff0077ff, 0, 0, 1);
        }

        if (!is_selecting && fan_2d::collision::rectangle::point_inside_no_rotation(
          cursor_pos_global, fan::graphics::gui::get_mouse_pos() - sprite_size / 2, sprite_size / 2
        )) {
          draw_list->AddRect(cursor_pos_global, cursor_pos_global + sprite_size, 0xff0077ff, 0, 0, 3);
        }

        bool is_mouse_hovered = fan::graphics::gui::is_item_hovered(fan::graphics::gui::hovered_flags_rect_only);

        if (is_mouse_hovered && is_left_mouse_drag) {
          min_rect_draw = min_rect_draw.min(cursor_pos_global);
          max_rect_draw = max_rect_draw.max(cursor_pos_global);
          min_rect = min_rect.min(fan::vec2(grid_index));
          max_rect = max_rect.max(fan::vec2(grid_index));
        }

        if (is_mouse_hovered && is_left_mouse_button_clicked && !is_left_ctrl_key_pressed) {
          is_selecting = true;
          selection_start = fan::graphics::gui::get_mouse_pos();
        }
        else if ((is_left_mouse_button_clicked || is_left_mouse_drag) && is_mouse_hovered) {
          current_image_indices.clear();
          if (current_image_indices.empty()) {
            current_tile_images.clear();
          }
          current_image_indices[grid_index] = i;
        }
        else if ((is_right_mouse_button_clicked || is_right_mouse_drag) && is_mouse_hovered) {
          auto found = current_image_indices.find(grid_index);
          if (found != current_image_indices.end()) {
            current_image_indices.erase(found);
          }
          if (current_image_indices.empty()) {
            current_tile_images.clear();
          }
        }

        if ((i + 1) % images_per_row != 0) {
          fan::graphics::gui::same_line();
        }
      }

      fan::vec2 cursor_grid = fan::graphics::gui::get_mouse_pos() - initial_pos;
      cursor_grid /= sprite_size;
      cursor_grid = cursor_grid.floor();

      if (is_selecting) {
        selection_end = fan::graphics::gui::get_mouse_pos();
        fan::vec2 max_rect_draw_adjusted = max_rect_draw + sprite_size;
        max_rect_draw_adjusted = max_rect_draw_adjusted.min(initial_pos + cursor_grid * sprite_size + sprite_size);
        if (min_rect != (uint32_t)~0 && max_rect != -1) {
          draw_list->AddRect(min_rect_draw, max_rect_draw_adjusted, 0xff0077ff);
        }

        if (is_left_mouse_button_released) {
          is_selecting = false;
          min_rect = (uint32_t)~0;
          max_rect = -1;
          min_rect_draw = (uint32_t)~0;
          max_rect_draw = -1;
        }
      }

      if (min_rect != (uint32_t)~0 && max_rect != -1) {
        for (int y = min_rect.y; y <= std::min(max_rect.y, cursor_grid.y); ++y) {
          for (int x = min_rect.x; x <= std::min(max_rect.x, cursor_grid.x); ++x) {
            current_image_indices[fan::vec2i(x, y)] = y * images_per_row + x;
          }
        }
      }

      style.ItemSpacing = prev_item_spacing;

      if (current_image_indices.size()) {
        current_tile_images.clear();
      }

      int prev_y = -1;
      int y = -1;
      int x = 0;
      for (auto& i : current_image_indices) {
        if (prev_y != i.first.y) {
          current_tile_images.resize(current_tile_images.size() + 1);
          prev_y = i.first.y;
          current_tile_brush_count.x = std::max(current_tile_brush_count.x, x);
          x = 0;
          y++;
        }
        current_tile_images[y].push_back(texture_pack_images[i.second]);
        x++;
      }
      current_tile_brush_count.x = std::max(current_tile_brush_count.x, x);
      current_tile_brush_count.y = y;
    }
    fan::graphics::gui::pop_style_color(5);
    fan::graphics::gui::pop_style_var();
  }

  void handle_tile_settings_window() {
    if (fan::graphics::gui::begin("Tile settings", nullptr, fan::graphics::gui::window_flags_no_focus_on_appearing)) {
      if (current_tile.layer != nullptr) {
        auto& layer = current_tile.layer[current_tile.layer_index];

        fan::vec2 offset = fan::vec2(layer.shape.get_position()) - current_tile.position;
        if (fan::graphics::gui::drag("offset", &offset, 0.1, 0, 0)) {
          layer.shape.set_position(fan::vec2(current_tile.position) + offset);
        }

        fan::vec2 tile_size = layer.shape.get_size();
        if (fan::graphics::gui::drag("tile size", &tile_size)) {
          layer.shape.set_size(tile_size);
        }

        std::string temp = layer.tile.id;
        temp.resize(max_id_len);
        if (fan::graphics::gui::input_text("id", &temp)) {
          layer.tile.id = temp.substr(0, std::strlen(temp.c_str()));
        }

        fan::vec3 angle = layer.shape.get_angle();
        if (fan::graphics::gui::drag("angle", &angle, fan::math::radians(1))) {
          layer.shape.set_angle(angle);
        }

        fan::vec2 rotation_point = layer.shape.get_rotation_point();
        if (fan::graphics::gui::drag("rotation_point", &rotation_point, 0.1, -tile_size.max() * 2, tile_size.max() * 2)) {
          layer.shape.set_rotation_point(rotation_point);
        }

        uint32_t flags = layer.shape.get_flags();
        if (fan::graphics::gui::input_int("special flags", (int*)&flags, 1, 1)) {
          layer.shape.set_flags(flags);
        }

        fan::color color = layer.shape.get_color();
        if (fan::graphics::gui::color_edit4("color", &color)) {
          layer.shape.set_color(color);
        }

        int mesh_property = (int)layer.tile.mesh_property;
        if (fan::graphics::gui::slider("mesh flags", &mesh_property, 0, (int)mesh_property_t::size - 1)) {
          layer.tile.mesh_property = (mesh_property_t)mesh_property;
        }
      }
    }
    fan::graphics::gui::end();
  }

  void handle_brush_settings_window() {
    if (fan::graphics::gui::begin("Brush settings", nullptr, fan::graphics::gui::window_flags_no_focus_on_appearing)) {
      int idx = (int)brush.depth - shape_depths_t::max_layer_depth / 2;
      if (fan::graphics::gui::drag("depth", (int*)&idx, 1, 0, shape_depths_t::max_layer_depth)) {
        brush.depth = idx + shape_depths_t::max_layer_depth / 2;
      }

      idx = (int)brush.mode;
      if (fan::graphics::gui::combo("mode", (int*)&idx, brush.mode_names, std::size(brush.mode_names))) {
        brush.mode = (brush_t::mode_e)idx;
      }

      idx = (int)brush.type;
      if (fan::graphics::gui::combo("type", (int*)&idx, brush.type_names, std::size(brush.type_names))) {
        brush.type = (brush_t::type_e)idx;
      }

      if (fan::graphics::gui::slider("jitter", &brush.jitter, 0, brush.size.min())) {
        grid_visualize.highlight_hover.set_size(tile_size * brush.size);
      }

      fan::graphics::gui::drag("jitter_chance", &brush.jitter_chance, 1, 0, 0.01);

      static int default_value = 0;
      if (fan::graphics::gui::combo("dynamics angle", &default_value, brush.dynamics_names, std::size(brush.dynamics_names))) {
        brush.dynamics_angle = (brush_t::dynamics_e)default_value;
      }

      default_value = 0;
      if (fan::graphics::gui::combo("dynamics color", &default_value, brush.dynamics_names, std::size(brush.dynamics_names))) {
        brush.dynamics_color = (brush_t::dynamics_e)default_value;
      }

      if (fan::graphics::gui::slider("size", &brush.size, 1, 4096)) {
        grid_visualize.highlight_hover.set_size(tile_size * brush.size);
      }

      fan::graphics::gui::slider("tile size", &brush.tile_size, 0.1, 1);
      fan::graphics::gui::drag("angle", &brush.angle);

      std::string temp = brush.id;
      temp.resize(max_id_len);
      if (fan::graphics::gui::input_text("id", &temp)) {
        brush.id = temp.substr(0, strlen(temp.c_str()));
      }

      fan::graphics::gui::color_edit4("color", &brush.color);

      switch (brush.type) {
        case brush_t::type_e::physics_shape: {
          fan::graphics::gui::slider("offset", &brush.offset, -1, 1);

          static int default_value = 0;
          if (fan::graphics::gui::combo("Physics shape type", &default_value, brush.physics_type_names, std::size(brush.physics_type_names))) {
            brush.physics_type = default_value;
          }

          default_value = 0;
          if (fan::graphics::gui::combo("Physics body type", &default_value, brush.physics_body_type_names, std::size(brush.physics_body_type_names))) {
            brush.physics_body_type = default_value;
          }

          static bool default_bool = 0;
          if (fan::graphics::gui::toggle_button("Physics shape draw", &default_bool)) {
            brush.physics_draw = default_bool;
          }

          static fan::physics::shape_properties_t shape_properties;
          if (fan::graphics::gui::drag("Physics shape friction", &shape_properties.friction, 0.01, 0, 1)) {
            brush.physics_shape_properties.friction = shape_properties.friction;
          }
          if (fan::graphics::gui::drag("Physics shape density", &shape_properties.density, 0.01, 0, 1)) {
            brush.physics_shape_properties.density = shape_properties.density;
          }
          if (fan::graphics::gui::toggle_button("Physics shape fixed rotation", &shape_properties.fixed_rotation)) {
            brush.physics_shape_properties.fixed_rotation = shape_properties.fixed_rotation;
          }
          if (fan::graphics::gui::toggle_button("Physics shape enable presolve events", &shape_properties.presolve_events)) {
            brush.physics_shape_properties.presolve_events = shape_properties.presolve_events;
          }
          if (fan::graphics::gui::toggle_button("Is sensor", &shape_properties.is_sensor)) {
            brush.physics_shape_properties.is_sensor = shape_properties.is_sensor;
          }
          break;
        }
      }
    }
    fan::graphics::gui::end();
  }

  void handle_lighting_settings_window() {
    if (fan::graphics::gui::begin("lighting settings", nullptr, fan::graphics::gui::window_flags_no_focus_on_appearing)) {
      static fan::vec3 ambient = fan::graphics::get_lighting().ambient;
      if (fan::graphics::gui::color_edit3("ambient", &ambient)) {
        fan::graphics::get_lighting().set_target(ambient);
      }
    }
    fan::graphics::gui::end();
  }

  void handle_physics_settings_window() {
    if (fan::graphics::gui::begin("physics settings", nullptr, fan::graphics::gui::window_flags_no_focus_on_appearing)) {
      fan::vec2 gravity = fan::physics::gphysics.get_gravity();
      if (fan::graphics::gui::drag("gravity", &gravity, 0.01)) {
        fan::physics::gphysics.set_gravity(gravity);
      }
    }
    fan::graphics::gui::end();
  }

  void handle_pick_tile() {
    fan::vec2i position;
    if (window_relative_to_grid(fan::window::get_mouse_position(), &position)) {
      fan::vec2i grid_position = position;
      convert_draw_to_grid(grid_position);
      grid_position /= tile_size * 2;
      auto found = map_tiles.find(fan::vec2i(grid_position.x, grid_position.y));
      if (found != map_tiles.end()) {
        auto& layers = found->second.layers;
        uint32_t idx = find_layer_shape(layers);
        if (idx == invalid) {
          idx = find_layer_shape(layers, true);
        }
        current_image_indices.clear();
        current_tile_images.clear();
        current_tile_images.resize(1);
        if (idx != invalid || idx < brush.depth) {
          uint16_t st = layers[idx].shape.get_shape_type();
          if (st == (uint16_t)fan::graphics::shape_type_t::sprite ||
            st == (uint16_t)fan::graphics::shape_type_t::unlit_sprite) {
            current_tile_images[0].push_back({
              .ti = layers[idx].shape.get_tp()
            });
          }
        }
      }
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
        uint32_t idx = find_layer_shape(layers);
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

  void handle_gui() {
    fan::vec2 editor_size;

    if (handle_editor_window(editor_size)) {
      fan::graphics::gui::end();
      return;
    }

    if (handle_editor_settings_window()) {
      return;
    }

    handle_tiles_window();
    handle_tile_settings_window();
    handle_brush_settings_window();
    handle_lighting_settings_window();
    handle_physics_settings_window();

    if (editor_settings.hovered && fan::window::is_mouse_down()) {
      if (fan::graphics::get_window().key_pressed(fan::key_t)) {
        handle_pick_tile();
      }
      else if (fan::graphics::get_window().key_pressed(fan::key_left_alt)) {
        handle_select_tile();
      }
    }

    handle_tile_brush();
    fan::graphics::gui::end();

    terrain_generator.render();
  }

  void render() {
    handle_gui();
  }

  void fout(std::string filename) {
    if (!filename.ends_with(".fte") && !filename.ends_with(".json")) {
      filename += ".fte";
    }

    bool is_temp = filename.find("temp") != std::string::npos;

#if defined(fan_json)
    previous_file_name = filename;

    fan::json ostr;
    ostr["version"] = 1;
    ostr["map_size"] = map_size;
    ostr["tile_size"] = tile_size;
    ostr["lighting.ambient"] = fan::graphics::get_lighting().ambient;
    ostr["gravity"] = fan::physics::gphysics.get_gravity();
    fan::json jtps = fan::json::array();
    for (auto* tp : texture_packs) {
      jtps.push_back(tp->file_path);
    }
    ostr["texture_packs"] = jtps;

    fan::json tiles = fan::json::array();

    std::unordered_set<fan::vec3> remove_duplicates;

    for (auto& i : map_tiles) {
      for (auto& j : i.second.layers) {
        fan::vec3 key(j.tile.position.x, j.tile.position.y, j.tile.position.z);
        if (remove_duplicates.find(key) == remove_duplicates.end()) {
          remove_duplicates.insert(key);
        }
        else {
          fan::graphics::gui::print_warning("warning duplicate tile positions:", key.to_string() + ", skipping...");
          continue;
        }
          
        fan::json tile;
        if (j.shape.get_size() == 0 && !is_temp) {
          fan::graphics::gui::print_warning("warning exporting tile with size 0", j.tile.position);
        }
        fan::graphics::shape_serialize(j.shape, &tile);
        fte_t::tile_t defaults;
        if (j.tile.mesh_property != defaults.mesh_property) {
          tile["mesh_property"] = j.tile.mesh_property;
        }
        if (j.tile.id != defaults.id) {
          tile["id"] = j.tile.id;
        }
        if (j.tile.action != defaults.action) {
          tile["action"] = j.tile.action;
        }
        if (j.tile.key != defaults.key) {
          tile["key"] = j.tile.key;
        }
        if (j.tile.key_state != defaults.key_state) {
          tile["key_state"] = j.tile.key_state;
        }
        if (j.tile.object_names != defaults.object_names) {
          tile["object_names"] = j.tile.object_names;
        }
        tiles.push_back(tile);
      }
    }

    remove_duplicates.clear();
    for (auto& i : physics_shapes) {
      for (auto& j : i.second) {
        fan::json tile;
        if (j.visual.get_size() == 0 && !is_temp) {
          fan::graphics::gui::print_warning("warning exporting tile with size 0", j.visual.get_position());
        }
        fan::graphics::shape_serialize(j.visual, &tile);
        if (remove_duplicates.find(j.visual.get_position()) == remove_duplicates.end()) {
          remove_duplicates.insert(j.visual.get_position());
        }
        else {
          fan::graphics::gui::print_warning("warning duplicate tile positions:", j.visual.get_position().to_string() + ", skipping...");
          continue;
        }
        tile["mesh_property"] = fte_t::mesh_property_t::physics_shape;
        if (j.id.size()) {
          tile["id"] = j.id;
        }
        //tile["action"] = tile_t().action;
        //tile["key"] = tile_t().key;
        //tile["key_state"] = tile_t().key_state;
        //tile["object_names"] = tile_t().object_names;

        fan::json physics_shape_data;

        if (j.type != decltype(j.type){}) {
          physics_shape_data["type"] = j.type;
        }
        if (j.body_type != decltype(j.body_type){}) {
          physics_shape_data["body_type"] = j.body_type;
        }
        if (j.draw != decltype(j.draw){}) {
          physics_shape_data["draw"] = j.draw;
        }
        fan::physics::shape_properties_t sp_defaults;
        if (j.shape_properties.friction != sp_defaults.friction) {
          physics_shape_data["friction"] = j.shape_properties.friction;
        }
        if (j.shape_properties.density != sp_defaults.density) {
          physics_shape_data["density"] = j.shape_properties.density;
        }
        if (j.shape_properties.fixed_rotation != sp_defaults.fixed_rotation) {
          physics_shape_data["fixed_rotation"] = j.shape_properties.fixed_rotation;
        }
        if (j.shape_properties.presolve_events != sp_defaults.presolve_events) {
          physics_shape_data["presolve_events"] = j.shape_properties.presolve_events;
        }
        if (j.shape_properties.is_sensor != sp_defaults.is_sensor) {
          physics_shape_data["is_sensor"] = j.shape_properties.is_sensor;
        }

        if (!physics_shape_data.empty()) {
          tile["physics_shape_data"] = physics_shape_data;
        }

        tiles.push_back(tile);
      }
    }

    fan::json j;
    for (auto& layer : visual_layers) {
      fan::json layer_json;
      layer_json["layer_name"] = layer.second.text;
      layer_json["depth"] = layer.first;
      j.push_back(layer_json);
    }

    ostr["layer_info"] = j;
    ostr["tiles"] = tiles;
    fan::io::file::write(filename, ostr.dump(2), std::ios_base::binary);
    if (!is_temp) {
      fan::graphics::gui::print_success("File saved to " + std::filesystem::absolute(filename).string());
    }
#else
    fan::throw_error("fan_json not enabled");
    __unreachable();
#endif
  }

  void fin(const std::string& filename, const std::source_location& callers_path = std::source_location::current()) {
    std::string out;
    fan::io::file::read(fan::io::file::find_relative_path(filename, callers_path), &out);
    fan::json json = fan::json::parse(out);
    if (json["version"] != 1) {
      fan::throw_error("version mismatch");
    }
    if (json.contains("texture_packs")) {
      std::vector<std::string> tp_paths = json["texture_packs"];
      for (auto& path : tp_paths) {
        open_texture_pack(path);
      }
    }
#if defined(fan_json)
    else if (texture_packs[0]->size() == 0) {
      fan::print("open valid texturepack");
      return;
    }
    invalidate_selection();
    previous_file_name = filename;

    map_size = json["map_size"];
    tile_size = json["tile_size"];
    if (json.contains("gravity")) {
      fan::physics::gphysics.set_gravity(json["gravity"]);
    }
    fan::graphics::get_lighting().set_target(json["lighting.ambient"]);
    map_tiles.clear();
    visual_layers.clear();
    visual_shapes.clear();
    physics_shapes.clear();
    resize_map();

    fan::graphics::shape_deserialize_t it;
    fan::graphics::shape_t shape;
    while (it.iterate(json["tiles"], &shape)) {
      const auto& shape_json = *(it.data.it - 1);
      if (shape_json.contains("mesh_property") && shape_json["mesh_property"] == fte_t::mesh_property_t::physics_shape) {
        auto& physics_shape = physics_shapes[shape.get_position().z];
        physics_shape.resize(physics_shape.size() + 1);
        auto& physics_element = physics_shape.back();
        shape.set_camera(render_view->camera);
        shape.set_viewport(render_view->viewport);
        shape.set_image(grid_visualize.collider_color);
        if (shape_json.contains("physics_shape_data")) {
          fte_t::physics_shapes_t defaults;
          physics_element.id = shape_json.value("id", defaults.id);
          const fan::json& physics_shape_data = shape_json["physics_shape_data"];
          physics_element.type = physics_shape_data.value("type", defaults.type);
          physics_element.body_type = physics_shape_data.value("body_type", defaults.body_type);
          physics_element.draw = physics_shape_data.value("draw", defaults.draw);
          physics_element.shape_properties.friction = physics_shape_data.value("friction", defaults.shape_properties.friction);
          physics_element.shape_properties.density = physics_shape_data.value("density", defaults.shape_properties.density);
          physics_element.shape_properties.fixed_rotation = physics_shape_data.value("fixed_rotation", defaults.shape_properties.fixed_rotation);
          physics_element.shape_properties.presolve_events = physics_shape_data.value("presolve_events", defaults.shape_properties.presolve_events);
          physics_element.shape_properties.is_sensor = physics_shape_data.value("is_sensor", defaults.shape_properties.is_sensor);
        }
        physics_element.visual = std::move(shape);
        continue;
      }
      fan::vec2i gp = shape.get_position();
      uint16_t depth = shape.get_position().z;
      convert_draw_to_grid(gp);
      gp /= tile_size * 2;
      auto found = map_tiles.find(gp);
      fte_t::shapes_t::global_t::layer_t* layer = nullptr;
      visual_layers[depth].positions[gp];
      if (found != map_tiles.end()) {
        found->second.layers.resize(found->second.layers.size() + 1);
        layer = &found->second.layers.back();
      }
      else {
        map_tiles[gp].layers.resize(1);
        layer = &map_tiles[gp].layers.back();
      }

      layer->tile.position = fan::vec3i(gp, depth);
      layer->tile.size = shape.get_size();
      layer->tile.angle = shape.get_angle();
      layer->tile.color = shape.get_color();
      layer->tile.id = shape_json.value("id", "");
      layer->tile.mesh_property = (mesh_property_t)shape_json.value("mesh_property", fte_t::tile_t().mesh_property);
      layer->shape = shape;

      switch (layer->tile.mesh_property) {
        case fte_t::mesh_property_t::none: {
          layer->shape.set_camera(render_view->camera);
          layer->shape.set_viewport(render_view->viewport);
          break;
        }
        case fte_t::mesh_property_t::light: {
          layer->shape = fan::graphics::light_t{{
            .render_view = render_view,
            .position = shape.get_position(),
            .size = layer->tile.size,
            .color = layer->tile.color
          }};
          visual_shapes[shape.get_position()].shape = fan::graphics::sprite_t{{
            .render_view = render_view,
            .position = fan::vec3(fan::vec2(shape.get_position()), shape.get_position().z + 1),
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
    }

    if (json.contains("layer_info")) {
      for (const auto& layer_json : json["layer_info"]) {
        layer_info_t layer_info;
        layer_info.layer_name = layer_json["layer_name"];
        layer_info.depth = layer_json["depth"];
        if (visual_layers.find(layer_info.depth) != visual_layers.end()) {
          visual_layers[layer_info.depth].text = layer_info.layer_name;
        }
      }
    }
#else 
    fan::throw_error("fan_json not enabled");
    __unreachable();
#endif
  }

  #define editor OFFSETLESS(this, fte_t, terrain_generator)

  struct terrain_generator_t {
    terrain_generator_t()
    {
      init();
    }

    void init() {
      render_view.create();
      ic.reference_camera = render_view.camera;
      ic.reference_viewport = render_view.viewport;
      ic.set_zoom(0.15f);

      tile_world.init();
      fan::vec2 map_size = tile_world.map_size;
      f32_t cell_size = tile_world.cell_size;

      rects.reserve(map_size.x * map_size.y);
      for (int y = 0; y < map_size.y; y++) {
        for (int x = 0; x < map_size.x; x++) {
          rects.push_back(fan::graphics::sprite_t{ {
            .render_view = &render_view,
            .position = fan::vec3(fan::vec2(x, y) * cell_size * 2.f + cell_size, 0xFFFA),
            .size = fan::vec2(cell_size, cell_size),
            .image = fan::graphics::tile_world_images::dirt
          } });
        }
      }

      visual_grid = fan::graphics::grid_t{ {
        .render_view = &render_view,
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
            tile_world.is_solid(x, y) ? fan::graphics::tile_world_images::dirt : fan::graphics::tile_world_images::background
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
      fan::graphics::gui::set_next_window_bg_alpha(0);
      if (fan::graphics::gui::begin("Terrain Generator", nullptr,
        fan::graphics::gui::window_flags_no_background | fan::graphics::gui::window_flags_no_focus_on_appearing)) 
      {
        {
          if (fan::graphics::gui::button("Iterate")) iterate();
          if (fan::graphics::gui::button("Reset")) reset();
          if (fan::graphics::gui::button("Insert to map")) {
            fan::graphics::gui::open_popup("Confirm Insert");
          }
        }
        {
          if (fan::graphics::gui::is_popup_open("Confirm Insert")) {
            fan::graphics::gui::set_next_window_pos(fan::window::get_size() / 2.0f, fan::graphics::gui::cond_once, 0.5);
          }

          if (fan::graphics::gui::begin_popup_modal("Confirm Insert", fan::graphics::gui::window_flags_always_auto_resize)) {
            fan::graphics::gui::text("Insert tiles into map at depth " + std::to_string((int)editor->brush.depth - shape_depths_t::max_layer_depth / 2) + 
              "?", fan::colors::yellow
            );
            fan::graphics::gui::text("It might overwrite tiles at the depth level.", fan::colors::yellow);

            if (fan::graphics::gui::button("Cancel")) {
              fan::graphics::gui::close_current_popup();
            }
            fan::graphics::gui::same_line();
            if (fan::graphics::gui::button("Confirm")) {
              insert_selected_tiles(editor->brush.depth);
              fan::graphics::gui::close_current_popup();
            }

            fan::graphics::gui::end_popup();
          }
        }
        {
          fan::vec2 need_init = !prev_viewport_size
            || prev_viewport_size != fan::graphics::gui::get_window_size();
          fan::graphics::gui::set_viewport(render_view.viewport);
          if (need_init) {
            ic.update();
          }
          prev_viewport_size = fan::graphics::gui::get_window_size();
        }
      }
      else {
        fan::graphics::viewport_zero(render_view.viewport);
      }
      fan::graphics::gui::end();
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

          uint32_t idx = editor->find_layer_shape(layers);
          if (idx == fte_t::invalid) {
            layers.resize(layers.size() + 1);
            idx = layers.size() - 1;
          }

          auto& layer = layers[idx];
          layer.tile.position = fan::vec3(grid_pos * editor->tile_size * 2, depth);
          layer.tile.size = editor->tile_size;
          layer.tile.mesh_property = fte_t::mesh_property_t::none;

          fan::graphics::image_t img = rects[x + y * map_size.x].get_image();
          if (img == fan::graphics::tile_world_images::dirt) {
            layer.tile.id = "##tile_world_dirt";
          }
          else if (img == fan::graphics::tile_world_images::background) {
            layer.tile.id = "##tile_world_background";
          }

          layer.shape = fan::graphics::sprite_t{ {
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

    fan::graphics::tile_world_generator_t tile_world;
    std::vector<fan::graphics::sprite_t> rects;

    fan::graphics::grid_t visual_grid;
    fan::graphics::interactive_camera_t ic;
    fan::graphics::render_view_t render_view;
    fan::vec2 prev_viewport_size = 0;
  }terrain_generator;

#undef editor


  std::string file_name = "tilemap_editor.json";
  fan::vec2i map_size{6, 6};
  fan::vec2i tile_size{32, 32};
  current_tile_t current_tile;
  fan::vec2i current_tile_brush_count;
  std::vector<std::vector<tile_info_t>> current_tile_images;
  std::map<fan::vec2i, int, sort_by_y_t> current_image_indices;
  std::unordered_map<fan::vec2i, shapes_t::global_t, vec2i_hasher> map_tiles;
  std::unordered_map<f32_t, std::vector<fte_t::physics_shapes_t>> physics_shapes;
  std::unordered_map<fan::vec3, visualize_t, vec3_hasher> visual_shapes;
  std::map<uint16_t, visual_layer_t> visual_layers;
  fan::vec2 texturepack_size{};
  fan::vec2 texturepack_single_image_size{};
  std::vector<tile_info_t> texture_pack_images;
  grid_visualize_t grid_visualize;
  brush_t brush;
  viewport_settings_t viewport_settings;
  editor_settings_t editor_settings;
  fan::vec3i prev_grid_position = 999999;
  fan::graphics::image_t transparent_texture;
  fan::vec2i copy_buffer_region = 0;
  std::vector<shapes_t::global_t> copy_buffer;
  fan::graphics::render_view_t* render_view = nullptr;
  std::function<void(int)> modify_cb = [](int) {};
  std::string previous_file_name;
  fan::graphics::shape_t visual_line;
  fan::window_t::buttons_handle_t buttons_handle;
  fan::window_t::keys_handle_t keys_handle;
  fan::window_t::mouse_move_handle_t mouse_move_handle;
  int original_image_width = 2048;
  inline static fan::graphics::file_save_dialog_t save_file_dialog;
  inline static fan::graphics::file_open_dialog_t open_file_dialog, open_tp_dialog;
  inline static fan::graphics::file_open_dialog_t models_open_file_dialog;
  std::vector<fan::graphics::texture_pack_t*> texture_packs;
};

#endif