module;

#include <cstring>
#include <functional>
#include <map>
#include <unordered_map>
#include <string>
#include <vector>

#include <fan/types/types.h>
#include <fan/math/math.h>

export module fan.graphics.gui.tilemap_editor.editor;

#if defined(fan_gui) && defined(fan_physics)

export import fan.graphics.gui; // export?

import fan.graphics.algorithm.raycast_grid;

import fan.types.color;
import fan.types.vector;
import fan.print;
import fan.file_dialog;
import fan.io.file;
import fan.graphics;
import fan.physics.b2_integration;

export struct fte_t {
  static constexpr int max_id_len = 48;
  static constexpr fan::vec2 default_button_size{ 100, 30 };
  static constexpr fan::vec2 tile_viewer_sprite_size{ 64, 64 };
  static constexpr fan::color highlighted_tile_color = fan::color(0.5, 0.5, 1);
  static constexpr fan::color highlighted_selected_tile_color = fan::color(0.5, 0, 0, 0.1);

  static constexpr f32_t scroll_speed = 1.2;
  static constexpr uint32_t invalid = -1;

  struct shape_depths_t {
    static constexpr int max_layer_depth = 0xFAAA - 2;
    static constexpr int cursor_highlight_depth = 0xFAAA - 1;
  };

  std::string file_name = "tilemap_editor.json";

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
        loco_t::shape_t shape;
      };
      std::vector<layer_t> layers;
    };
  };

  enum class event_type_e {
    none,
    add,
    remove
  };

  uint32_t find_top_layer_shape(const auto& vec) {
    uint32_t found = -1;
    int64_t depth = -1;
    for (std::size_t i = 0; i < vec.size(); ++i) {
      if (vec[i].tile.position.z > depth) {
        depth = vec[i].tile.position.z;
        found = i;
      }
    }
    return found;
  };

  uint32_t find_layer_shape(const auto& vec) {
    uint32_t found = -1;
    for (std::size_t i = 0; i < vec.size(); ++i) {
      if (vec[i].tile.position.z == brush.depth) {
        found = i;
        break;
      }
    }
    return found;
  };

  void resize_map() {
    grid_visualize.background.set_size(tile_size * map_size);
    grid_visualize.background.set_tc_size(fan::vec2(0.5) * map_size);
    grid_visualize.background.set_position(fan::vec3(tile_size * 2 * map_size / 2 - tile_size, 0));
    grid_visualize.grid.set_grid_size(
      map_size
    );
    if (grid_visualize.render_grid) {
      grid_visualize.grid.set_size(map_size * (tile_size / 2) * 2);
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
    auto camera_position = gloco->camera_get_position(camera->camera);
    fan::vec2 vs = gloco->viewport_get_size(camera->viewport);
    fan::vec2 p = gloco->translate_position(window_relative_position, camera->viewport, camera->camera) + camera_position;

    //p -= tile_size / 2;
    //if (!(map_size.x % 2)) {
    //  p += tile_size;
    //}
    //else {
      //p += fan::vec2i((map_size.x % 2) * tile_size.x, (map_size.y % 2) * tile_size.y);
     // p += fan::vec2i(!(map_size.x % 2) * -tile_size.x * 0.5, !(map_size.y % 2) * -tile_size.y);
    //}
    //fan::print(p.floor());
    //fan::print(p);
    
    *in = ((p + tile_size) / (tile_size * 2)).floor() * (tile_size * 2);
    //*in += tile_size;

    return fan_2d::collision::rectangle::point_inside_no_rotation(*in - map_size * tile_size / 2, map_size / 2 * tile_size - tile_size, map_size * tile_size);
  }

  void convert_draw_to_grid(fan::vec2i& p) {

  }

  void convert_grid_to_draw(fan::vec2i& p) {

  }

  struct properties_t {
    loco_t::camera_impl_t* camera = nullptr;
  };

  void open(const properties_t& properties) {
    loco_t::image_load_properties_t lp;
    lp.visual_output = loco_t::image_sampler_address_mode::repeat;
    lp.min_filter = fan::graphics::image_filter::nearest;
    lp.mag_filter = fan::graphics::image_filter::nearest;
    if (properties.camera == nullptr) {
      camera = &gloco->orthographic_camera;
    }
    else {
      camera = properties.camera;
    }

    gloco->window.add_mouse_move_callback([this](const auto& d) {
      if (viewport_settings.move) {
        fan::vec2 move_off = (d.position - viewport_settings.offset) / viewport_settings.zoom;
        gloco->camera_set_position(camera->camera, viewport_settings.pos - move_off);
      }
      fan::vec2i p;
      {
        if (window_relative_to_grid(d.position, &p)) {
          //convert_draw_to_grid(p);
          grid_visualize.highlight_hover.set_position(fan::vec2(p) + tile_size * brush.offset);
          grid_visualize.highlight_hover.set_size(tile_size * brush.tile_size * brush.size);
          grid_visualize.highlight_hover.set_color(fan::color(1, 1, 1, 0.6));
        }
        else {
          grid_visualize.highlight_hover.set_color(fan::colors::transparent);
        }
      }
      });

    gloco->window.add_buttons_callback([this](const auto& d) {
      if (d.button == fan::mouse_left && d.state == fan::mouse_state::release) {
        // reset on release
        prev_grid_position = -999999;
      }
      if (!editor_settings.hovered && d.state != fan::mouse_state::release) {
        return;
      }

      {// handle camera movement
        f32_t old_zoom = viewport_settings.zoom;

        switch (d.button) {
        case fan::mouse_middle: {
          viewport_settings.move = (bool)d.state;
          fan::vec2 old_pos = viewport_settings.pos;
          viewport_settings.offset = gloco->get_mouse_position();
          viewport_settings.pos = gloco->camera_get_position(camera->camera);
          break;
        }
        case fan::mouse_scroll_up: {
          if (gloco->window.key_pressed(fan::key_left_control)) {
            brush.depth += 1;
            brush.depth = std::min((int)brush.depth, shape_depths_t::max_layer_depth);
          }
          else if (gloco->window.key_pressed(fan::key_left_shift)) {
            brush.size += 1;
            grid_visualize.highlight_hover.set_size(tile_size * brush.size);
          }
          else {
            viewport_settings.zoom *= scroll_speed;

            auto& style = fan::graphics::gui::get_style();
            fan::vec2 frame_padding = style.FramePadding;
            fan::vec2 pos = (gloco->window.get_mouse_position() - viewport_settings.window_related_mouse_pos);
            pos /= gloco->window.get_size();
            pos *= viewport_settings.size / 2;
            viewport_settings.zoom_offset += pos / viewport_settings.zoom;

            // requires window's cursor position
            //fan::vec2 mouse_position = viewport_settings.window_related_mouse_pos;
            //mouse_position -= viewport_settings.size / 2;

            //viewport_settings.pos += (mouse_position * viewport_settings.zoom) / 4;

            //gloco->camera_set_position(camera->camera, viewport_settings.pos);
          }
          return;
        }
        case fan::mouse_scroll_down: {
          if (gloco->window.key_pressed(fan::key_left_control)) {
            brush.depth -= 1;
            brush.depth = std::max((uint32_t)brush.depth, (uint32_t)1);
          }
          else if (gloco->window.key_pressed(fan::key_left_shift)) {
            brush.size = (brush.size - 1).max(fan::vec2i(1));
            grid_visualize.highlight_hover.set_size(tile_size * brush.size);
          }
          else {
            viewport_settings.zoom /= scroll_speed;
          }
          return;
        }
        default: {return;} //?
        };
      }// handle camera movement
      });

    gloco->window.add_keys_callback([this](const auto& d) {
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
        if (gloco->window.key_pressed(fan::key_left_control)) {
          reset_map();
        }
        break;
      }
                          // change this
      case fan::key_e: {
        /* if (render_collisions) {
           draw_collisions();
         }
         else {
           undraw_collisions();
         }*/
        break;
      }
      }
      });

    viewport_settings.size = 0;

    transparent_texture = gloco->create_transparent_texture();

    grid_visualize.background = fan::graphics::sprite_t{ {
      .camera = camera,
      .position = fan::vec3(tile_size * 2 * map_size / 2 - tile_size, 0),
      .size = 0,
      .image = transparent_texture,
    } };

    lp.min_filter = fan::graphics::image_filter::nearest;
    lp.mag_filter = fan::graphics::image_filter::nearest;

    grid_visualize.highlight_color = gloco->image_load("images/highlight_hover.webp", lp);
    grid_visualize.collider_color = gloco->image_create(fan::color(0, 0.5, 0, 0.5));
    grid_visualize.light_color = gloco->image_load("images/lightbulb.webp", lp);

    grid_visualize.highlight_hover = fan::graphics::unlit_sprite_t{ {
      .camera = camera,
      .position = fan::vec3(viewport_settings.pos, shape_depths_t::cursor_highlight_depth),
      .size = tile_size,
      .image = grid_visualize.highlight_color,
      .blending = true
    } };
    grid_visualize.highlight_selected = fan::graphics::unlit_sprite_t{ {
      .camera = camera,
      .position = fan::vec3(viewport_settings.pos, shape_depths_t::cursor_highlight_depth - 1),
      .size = 0,
      .color = fan::color(2, 2, 2, 1),
      .image = grid_visualize.highlight_color,
      .blending = true
    } };

    {
      fan::vec2 p = 0;
      p = ((p - tile_size) / tile_size).floor() * tile_size;
      grid_visualize.highlight_hover.set_position(p);
    }

    //// update viewport sizes
    //gloco->process_frame();

    gloco->camera_set_position(camera->camera, viewport_settings.pos);

    loco_t::grid_t::properties_t p;
    p.viewport = camera->viewport;
    p.camera = camera->camera;
    p.position = fan::vec3(0, 0, shape_depths_t::cursor_highlight_depth + 1);
    p.size = 0;
    p.color = fan::color::rgb(0, 128, 255);

    grid_visualize.grid = p;

    resize_map();

    visual_line = fan::graphics::line_t{{
      .camera = camera,
      .src = fan::vec3(0, 0, shape_depths_t::cursor_highlight_depth + 1),
      .dst = fan::vec2(400),
      .color = fan::colors::white
    }};

  }
  void close() {

  }

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
  };

  bool handle_tile_push(fan::vec2i& position, int& pj, int& pi) {
    if (brush.jitter) {
      if ( brush.jitter_chance <= fan::random::value_f32(0, 1)) {
        return true;
      }
      position += (fan::random::vec2i(-brush.jitter, brush.jitter) * 2 + 1) * tile_size + tile_size;
    }
    if (!is_in_constraints(position, pj, pi)) {
      return true;
    }
    /*fan::vec2 start_idx = -(current_tile_brush_count / 2).floor();
    position += start_idx * tile_size * 2;*/

    f32_t inital_x = position.x;

    fan::vec2i grid_position = position;

    convert_draw_to_grid(grid_position);

    brush.line_src = gloco->get_mouse_position(camera->camera, camera->viewport);

    grid_position /= (tile_size * 2);
    auto& layers = map_tiles[grid_position].layers;
    visual_layers[brush.depth].positions[grid_position] = 1;
    uint32_t idx = find_layer_shape(layers);

    if (idx == invalid && 
    (brush.type == brush_t::type_e::light)) {
      layers.resize(layers.size() + 1);
      layers.back().tile.position = fan::vec3(position, brush.depth);
      layers.back().tile.size = tile_size * brush.tile_size;
      switch (brush.type) {
      case brush_t::type_e::light: {
        loco_t::light_t::properties_t lp;
        auto& shape = layers.back().shape;
        layers.back().tile.id = brush.id;
        lp.camera = camera->camera;
        lp.viewport = camera->viewport;
        lp.position = fan::vec3(position, brush.depth);
        lp.size = tile_size * brush.tile_size;
        lp.color = brush.dynamics_color == brush_t::dynamics_e::randomize ? fan::random::color() : brush.color;
        layers.back().shape = lp;
        layers.back().tile.mesh_property = mesh_property_t::light;
        visual_shapes[lp.position].shape = fan::graphics::sprite_t{ {
            .camera = camera,
            .position = fan::vec3(fan::vec2(lp.position), lp.position.z + 1),
            .size = tile_size,
            .image = grid_visualize.light_color,
            .blending = true
        } };
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
    else if (brush.type == brush_t::type_e::light){
      if (idx != invalid || idx < layers.size()) {
        auto& layer = layers[idx];
        layer.tile.id = brush.id;
        layer.tile.size = tile_size;
        switch (brush.type) {
        case brush_t::type_e::light: {
          loco_t::light_t::properties_t lp;
          auto& shape = layer.shape;
          lp.position = shape.get_position();
          lp.size = shape.get_size();
          lp.angle = shape.get_angle();
          lp.color = shape.get_color();
          lp.camera = camera->camera;
          lp.viewport = camera->viewport;
          layer.shape = lp;
          layer.tile.mesh_property = mesh_property_t::light;
          visual_shapes[lp.position].shape = fan::graphics::sprite_t{ {
              .camera = camera,
              .position = fan::vec3(fan::vec2(lp.position), lp.position.z + 1),
              .size = tile_size,
              .image = grid_visualize.light_color,
              .blending = true
          } };
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
          layers.back().tile.image_name = tile.image_name;
          // todo fix
          layers.back().tile.mesh_property = mesh_property_t::none;
          if (brush.type != brush_t::type_e::light) {
            layers.back().shape = fan::graphics::sprite_t{ {
                .camera = camera,
                .position = fan::vec3(position, brush.depth),
                .size = tile_size * brush.tile_size,
                .angle = brush.dynamics_angle == brush_t::dynamics_e::randomize ?
                      fan::vec3(0, 0, get_snapped_angle()) : brush.angle,
                .color = brush.dynamics_color == brush_t::dynamics_e::randomize ? fan::random::color() : brush.color,
                .blending = true
            } };
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
                  layer.shape = fan::graphics::sprite_t{ {
                      .camera = camera,
                      .position = layer.shape.get_position(),
                      .size = layer.shape.get_size(),
                      .angle = layer.shape.get_angle(),
                      .color = layer.shape.get_color()
                  } };
                   if (layer.shape.set_tp(&tile.ti)) {
                     fan::print("failed to load image");
                   }
                  layer.tile.mesh_property = mesh_property_t::none;
                  layer.tile.image_name = tile.image_name;
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

    {
      auto found = physics_shapes.find(brush.depth);
      if (found != physics_shapes.end()) {
        // find mouse hit from this depth
        for (auto it = found->second.begin(); it != found->second.end(); ++it) {
          if (fan_2d::collision::rectangle::point_inside_no_rotation(position, it->visual.get_position(), it->visual.get_size())) {
            it = found->second.erase(it);
            return false;
          }
        }
      }
    }

    if (brush.jitter) {
      if (brush.jitter_chance <= fan::random::value_f32(0, 1)) {
        return true;
      }
      position += (fan::random::vec2i(-brush.jitter, brush.jitter) * 2 + 1) * tile_size + tile_size;
    }
    
    auto found = map_tiles.find(grid_position);
    if (found != map_tiles.end()) {
      auto& layers = found->second.layers;
      uint32_t idx = find_layer_shape(layers);
      if (idx != invalid || idx < layers.size()) {
        switch (layers[idx].tile.mesh_property) {
          case mesh_property_t::light:{
            fan::vec3 erase_position = layers[idx].shape.get_position();
            //erase_position.z -= 1;
            auto found = visual_shapes.find(erase_position);
            if (found != visual_shapes.end()) {
              visual_shapes.erase(found);
            }
            break;
          }
          default: {
            break;
//            fan::throw_error("");
          }
        }
        layers.erase(layers.begin() + idx);
        auto visual_found = visual_layers.find(brush.depth);
        if (visual_found != visual_layers.end()) {
           visual_found->second.positions.erase(found->first);
          if (visual_found->second.positions.empty()) {
            visual_layers.erase(visual_found);
          }
        }
      }
      if (found->second.layers.empty()) {
        map_tiles.erase(found->first);
      }
    }
    invalidate_selection();
    return false;
  }

  bool mouse_to_grid(fan::vec2i& position) {
    if (window_relative_to_grid(gloco->get_mouse_position(), &position)) {
      convert_draw_to_grid(position);
      position /= tile_size * 2;
      return true;
    }
    return false;
  }

  void handle_tile_brush() {
    if (!editor_settings.hovered)
      return;

    static std::vector<loco_t::shape_t> select;
    static fan::vec2i copy_src;
    static fan::vec2i copy_dst;
    bool is_mouse_left_clicked = fan::window::is_mouse_clicked();
    bool is_mouse_left_down = fan::window::is_mouse_down();
    bool is_mouse_left_released = fan::window::is_mouse_released();
    bool is_mouse_right_clicked = fan::window::is_mouse_clicked(fan::mouse_right);
    bool is_mouse_right_down = fan::window::is_mouse_down(fan::mouse_right);

    static auto handle_select_tiles = [&] {
      select.clear();
      fan::vec2i mouse_grid_pos;
      if (mouse_to_grid(mouse_grid_pos)) {
        fan::vec2i src = copy_src;
        fan::vec2i dst = mouse_grid_pos;
        copy_dst = dst;
        // 2 is coordinate specific
        int stepx = (src.x <= dst.x) ? 1 : -1;
        int stepy = (src.y <= dst.y) ? 1 : -1;
        for (int j = src.y; j != dst.y + stepy; j += stepy) {
          for (int i = src.x; i != dst.x + stepx; i += stepx) {
            select.push_back(fan::graphics::unlit_sprite_t{ {
                .camera = camera,
                .position = fan::vec3(fan::vec2(i, j) * tile_size * 2, shape_depths_t::cursor_highlight_depth),
                .size = tile_size,
                .image = grid_visualize.highlight_color,
                .blending = true
            } });
          }
        }
      }
    };

    switch (brush.mode) {
      case brush_t::mode_e::draw: {
        
        switch (brush.type) {
        case brush_t::type_e::physics_shape:{

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
                  .visual = fan::graphics::sprite_t{ {
                    .camera = camera, // since its one shape, cant have offsets with multidrag
                    .position = fan::vec3(((fan::vec2(copy_src) + (fan::vec2(copy_dst) - fan::vec2(copy_src)) / 2) + brush.offset/2) * tile_size * 2, brush.depth),
                    .size = (((copy_dst - copy_src) + fan::vec2(1, 1)) * fan::vec2(tile_size)).abs() * brush.tile_size,
                    .image = grid_visualize.collider_color,
                    .blending = true
                  } },
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
        bool is_ctrl_pressed = gloco->window.key_pressed(fan::key_left_control);
        bool is_shift_pressed = gloco->window.key_pressed(fan::key_left_shift);

        if (is_mouse_left_down && !is_ctrl_pressed && !is_shift_pressed) {
          handle_tile_action(position, [this](auto...args) {auto ret = handle_tile_push(args...); return ret; });
          //modify_cb(0);
        }

        if (is_mouse_right_down) {
          handle_tile_action(position, [this](auto...args) {auto ret = handle_tile_erase(args...);  return ret; });
          //modify_cb(0);
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
                // slow
                for (auto& i : copy_buffer.back().layers) {
                  i.shape.set_size(0);
                }
              }
            }
            else {
              for (int j = src.y; j != dst.y + stepy; j += stepy) {
                for (int i = src.x; i != dst.x + stepx; i += stepx) {
                  auto found = map_tiles.find(fan::vec2i(i, j));
                  if (found != map_tiles.end()) {

                    copy_buffer.push_back(found->second);
                    for (auto& i : copy_buffer.back().layers) {
                      i.shape.set_size(0);
                    }
                  }
                  else {
                    copy_buffer.push_back({});
                  }
                }
              }
            }
          }
        }

        // paste copy buffer
        if (is_mouse_right_clicked) {
          fan::vec2i mouse_grid_pos;
          if (mouse_to_grid(mouse_grid_pos)) {
            int index = 0;
            for (auto& i : copy_buffer) {
              fan::vec2i current_pos = mouse_grid_pos + fan::vec2i(index % copy_buffer_region.x, index / copy_buffer_region.x);
              if (is_in_constraints(current_pos * tile_size * 2)) {
                auto& tile = map_tiles[current_pos];
                tile = i;
                int layer = 0;
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
                            .camera = camera,
                            .position = fan::vec3(draw_pos, brush.depth + 1),
                            .size = tile.layers[k].tile.size,
                            .image = grid_visualize.light_color,
                            .blending = true
                        } };
                        break;
                      }
                      default: {
                        break;
                      }
                    }
                  }
                }
                
                /*for (auto& t : tile.layers) {
                  fan::vec2 op = t.shape.get_position();
                  fan::vec2 offset = op - *(fan::vec2i*)&t.tile.position;
                  fan::print(offset);
                  fan::vec2 draw_pos = current_pos * tile_size * 2 + tile_size + offset;
                  if (is_in_constraints(draw_pos)) {
                    t.shape.set_position(fan::vec2(draw_pos));
                  }
                }*/
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
    if (!window_relative_to_grid(gloco->get_mouse_position(), &position)) {
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

  inline static fan::graphics::file_save_dialog_t save_file_dialog;
  inline static fan::graphics::file_open_dialog_t open_file_dialog, open_tp_dialog;
  inline static fan::graphics::file_open_dialog_t models_open_file_dialog;

  void open_texturepack(const std::string& path) {
    texturepack.open_compiled(path);
    texturepack_images.clear();
    texturepack_images.reserve(texturepack.texture_list.size());

    // loaded texturepack
    texturepack.iterate_loaded_images([this](auto& image, uint32_t pack_id) {
      tile_info_t ii;
      ii.ti = loco_t::texturepack_t::ti_t{
        .pack_id = pack_id,
        .position = image.position,
        .size = image.size,
        .image = &texturepack.get_pixel_data(pack_id).image
      };

      ii.image_name = image.image_name;

      auto& img_data = gloco->image_get_data(texturepack.get_pixel_data(pack_id).image);
      fan::vec2 size = img_data.size;

      texturepack_images.push_back(ii);
      texturepack_size = texturepack_size.max(fan::vec2(size));
      texturepack_single_image_size = texturepack_single_image_size.max(fan::vec2(image.size));
    });
  }

  bool handle_editor_window(fan::vec2& editor_size) {
    if (fan::graphics::gui::begin_main_menu_bar()) {
      {
        static std::string fn;
        if (fan::graphics::gui::begin_menu("File")) {

          if (fan::graphics::gui::menu_item("Open..")) {
            open_file_dialog.load("json;fmm", &fn);
          }

          if (fan::graphics::gui::menu_item("Save")) {
            fout(previous_file_name);
          }

          if (fan::graphics::gui::menu_item("Save as")) {
            save_file_dialog.save("json;fmm", &fn);
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
          }
          open_file_dialog.finished = false;
        }
        if (save_file_dialog.is_finished()) {
          if (fn.size() != 0) {
            fout(fn);
          }
          save_file_dialog.finished = false;
        }
        if (open_tp_dialog.is_finished()) {
          if (fn.size() != 0) {
            open_texturepack(fn);
          }
          open_tp_dialog.finished = false;
        }
      }
      fan::graphics::gui::end_main_menu_bar();

      fan::vec2 initial_pos = fan::graphics::gui::get_cursor_screen_pos();

      fan::graphics::gui::push_style_color(fan::graphics::gui::col_window_bg, fan::color(0, 0, 0, 0));

      fan::graphics::gui::begin(
        "Tilemap Editor2",
        nullptr,
        fan::graphics::gui::window_flags_no_background |
        fan::graphics::gui::window_flags_no_title_bar |
        fan::graphics::gui::window_flags_no_decoration |
        fan::graphics::gui::window_flags_no_collapse |
        fan::graphics::gui::window_flags_no_scroll_with_mouse
      );

      fan::graphics::gui::pop_style_color();

      fan::vec2 window_size = fan::graphics::gui::get_io().DisplaySize;

      fan::vec2 viewport_size = fan::graphics::gui::get_content_region_avail();

      fan::vec2 mainViewportPos = fan::graphics::gui::get_main_viewport()->Pos;

      fan::vec2 windowPos = fan::graphics::gui::get_window_pos();

      auto& style = fan::graphics::gui::get_style();
      fan::vec2 frame_padding = style.FramePadding;
      fan::vec2 viewport_pos = fan::graphics::gui::get_window_content_region_min() - frame_padding;

      fan::vec2 real_viewport_size = viewport_size + frame_padding * 2 + fan::vec2(0, style.WindowPadding.y * 2);

      real_viewport_size.x = fan::clamp(real_viewport_size.x, 1.f, real_viewport_size.x);
      real_viewport_size.y = fan::clamp(real_viewport_size.y, 1.f, real_viewport_size.y);

      gloco->camera_set_ortho(
          camera->camera,
          (fan::vec2(-real_viewport_size.x / 2, real_viewport_size.x / 2) / viewport_settings.zoom) + viewport_settings.zoom_offset.x,
          (fan::vec2(-real_viewport_size.y / 2, real_viewport_size.y / 2) / viewport_settings.zoom) + viewport_settings.zoom_offset.y
      );



      gloco->viewport_set(
        camera->viewport, 
        viewport_pos + fan::vec2(0, style.WindowPadding.y * 2),
        real_viewport_size,
        window_size
      );
      editor_size = real_viewport_size;
      viewport_settings.size = viewport_size;
      static int init = 0;
      if (init == 0) {
        //viewport_settings.pos = viewport_settings.size / 2 - tile_size * 2 * map_size / 2;
        //gloco->camera_set_position(camera->camera, viewport_settings.pos);
        init = 1;
      }

      viewport_settings.window_related_mouse_pos = fan::vec2(fan::vec2(fan::graphics::gui::get_window_pos()) + fan::vec2(fan::graphics::gui::get_window_size() / 2) + fan::vec2(0, style.WindowPadding.y * 2 - frame_padding.y * 2));

      fan::graphics::gui::set_window_font_scale(1.5);
      //viewport_settings.window_related_mouse_pos = ImGui::GetMousePos();
      {
        fan::graphics::gui::text("brush type: "_str + brush.type_names[(uint8_t)brush.type]);
        fan::graphics::gui::text("brush depth: " + std::to_string((int)brush.depth - shape_depths_t::max_layer_depth / 2));
      }

      fan::vec2 prev_item_spacing = style.ItemSpacing;

      style.ItemSpacing = fan::vec2(0);

      fan::vec2 old_cursorpos = fan::graphics::gui::get_cursor_pos();
      //fan::vec2 draw_start = ((initial_pos + fan::vec2( - initial_pos)) / tile_viewer_sprite_size).floor() * tile_viewer_sprite_size;
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
          
          auto& img_data = gloco->image_get_data(*j.ti.image);
          fan::vec2 size = img_data.size;

          fan::graphics::gui::image_rotated(
            *j.ti.image,
            (tile_viewer_sprite_size / std::max(1.f, current_tile_brush_count.x / 5.f))  * viewport_settings.zoom,
            360.f - fan::math::degrees(brush.angle.z),
            j.ti.position / size,
            j.ti.position / size +
            j.ti.size / size,
            fan::color(1, 1, 1, 0.9)
          );
        }
      }

      fan::graphics::gui::set_cursor_pos(old_cursorpos);
      style.ItemSpacing = prev_item_spacing;

      
      {// display cursor position
        fan::vec2 cursor_position = gloco->get_mouse_position();
        fan::vec2i grid_pos;
        if (window_relative_to_grid(cursor_position, &grid_pos)) {
          auto str = grid_pos.to_string();
          fan::graphics::gui::text_bottom_right(str.c_str(), 0);
        }
      }


      if (fan::graphics::gui::begin("Layer window")) {
        for (auto& layer_pair : visual_layers) {
          auto& layer = layer_pair.second; 
          layer.text.resize(32);
          uint16_t depth = layer_pair.first;
          auto fmt = ("Layer " + std::to_string(depth - shape_depths_t::max_layer_depth / 2));
          if (fan::graphics::gui::toggle_button(("Visible " + fmt).c_str(), &layer.visible)) {
            static auto iterate_positions = [&] (auto l) {
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
            if (layer.visible == false) { // hide
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
        // if (ImGui::Button("+")) {
        //   layers.push_back(data_t{ .text = "default" });
        // }
        // ImGui::SameLine();
        // if (ImGui::Button("-")) {
        //   layers.pop_back();
        // }
      }
      fan::graphics::gui::end();
    }
    else {
      gloco->viewport_zero(camera->viewport);
      return true;
    }
    editor_settings.hovered = fan::graphics::gui::is_window_hovered();
    fan::graphics::gui::end();

    if (gloco->window.key_state(fan::key_shift) != -1) {
      fan::vec2 line_dst = gloco->get_mouse_position(camera->camera, camera->viewport);
      visual_line.set_line(brush.line_src, line_dst);
      if (gloco->window.key_state(fan::mouse_left) == 1) {
        //fan::print(brush.line_src, line_dst);
         brush.line_src = ((brush.line_src + tile_size) / (tile_size * 2)).floor() * tile_size * 2;
         line_dst = ((line_dst + tile_size) / (tile_size * 2)).floor() * tile_size * 2;
         if (line_dst.x - brush.line_src.x > tile_size.x*2) {
          line_dst.x += tile_size.x * 2;
         }
         if (line_dst.y - brush.line_src.y > tile_size.y*2) {
          line_dst.y += tile_size.y * 2;
         }
        std::vector<fan::vec2i> raycast_positions = fan::graphics::algorithm::grid_raycast({ brush.line_src / 2, line_dst / 2 }, tile_size);
        int i = 0;
        for (fan::vec2i& i : raycast_positions) {
          fan::vec2i p = i;
         // p /= 2;
          //p -= tile_size;
          p *= (tile_size * 2);
          //convert_grid_to_draw(p);
        //  fan::print(p);
          //fan::vec2 p2 = 
          //int i = 0, j = 0;
          for (int i = 0; i < brush.size.y; ++i) {
            for (int j = 0; j < brush.size.x; ++j) {
              fan::vec2i abc = p + fan::vec2i(j * (tile_size.x), i * (tile_size.y));
              handle_tile_push(abc, i, j);
            }
          }
        }
        //for (auto& pos : raycast_positions) {
          //map[pos.y][pos.x].set_color(fan::colors::green);
        //}
      }
    }
    else {
      visual_line.set_line(0, 0);
    }
    return false;
  }

  bool handle_editor_settings_window() {
    if (fan::graphics::gui::begin("Editor settings")) {
      {
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
            grid_visualize.grid.set_size(map_size * (tile_size / 2) * 2);
          }
          else {
            grid_visualize.grid.set_size(0);
          }
        }

        // use ImGui::Dummy here
       /* fan::vec2 window_size = ImGui::GetWindowSize();
        fan::vec2 cursor_pos(
          window_size.x - default_button_size.x - ImGui::GetStyle().WindowPadding.x,
          window_size.y - default_button_size.y - ImGui::GetStyle().WindowPadding.y
        );*/
       /* ImGui::SetCursorPos(cursor_pos);
        if (ImGui::Button("Save")) {
          fout(file_name);
        }
        cursor_pos.x += default_button_size.x / 2;
        ImGui::SetCursorPos(cursor_pos);
        if (ImGui::Button("Quit")) {
          return true;
        }*/
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

      fan::graphics::gui::drag_int("original image width", &original_image_width, 1, 0, 1000);

      auto& style = fan::graphics::gui::get_style();
      fan::vec2 prev_item_spacing = style.ItemSpacing;
      style.ItemSpacing = fan::vec2(0);

      current_tile_brush_count = 0;

      int total_images = texturepack_images.size();

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

      for (uint32_t i = 0; i < texturepack_images.size(); i++) {
        auto& node = texturepack_images[i];
        fan::vec2i grid_index(i % images_per_row, i / images_per_row);
        bool selected = false;


        fan::vec2 cursor_pos_global = fan::graphics::gui::get_cursor_screen_pos();
        sprite_size = fan::vec2(final_image_size * zoom);

        auto& img_data = gloco->image_get_data(*node.ti.image);

        fan::vec2 size = img_data.size;

        fan::graphics::gui::image_button(
          (std::string("##ibutton") + std::to_string(i)).c_str(),
          *node.ti.image,
          sprite_size,
          node.ti.position / size,
          node.ti.position / size + node.ti.size / size
        );

        if (current_image_indices.find(grid_index) != current_image_indices.end()) {
           draw_list->AddRect(cursor_pos_global, cursor_pos_global + sprite_size, 0xff0077ff, 0, 0, 1);
           selected = true;
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
          //current_image_indices[grid_index] = i;
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
        current_tile_images[y].push_back(texturepack_images[i.second]);
        x++;
      }
      current_tile_brush_count.x = std::max(current_tile_brush_count.x, x);
      current_tile_brush_count.y = y;
    }
    fan::graphics::gui::pop_style_color(5);
    fan::graphics::gui::pop_style_var();
  }


  void handle_tile_settings_window() {
    if (fan::graphics::gui::begin("Tile settings")) {
      if (current_tile.layer != nullptr) {
        auto& layer = current_tile.layer[current_tile.layer_index];

        {
          fan::vec2 offset = fan::vec2(layer.shape.get_position()) - current_tile.position;

          if (fan::graphics::gui::drag_float("offset", &offset, 0.1, 0, 0)) {
            layer.shape.set_position(fan::vec2(current_tile.position) + offset);
          }
        }
        {
          fan::vec2 tile_size = layer.shape.get_size();
          if (fan::graphics::gui::drag_float("tile size", &tile_size)) {
            layer.shape.set_size(tile_size);
          }
        }
         {
          std::string temp = layer.tile.id;
          temp.resize(max_id_len);
          if (fan::graphics::gui::input_text("id", &temp)) {
            layer.tile.id = temp.substr(0, std::strlen(temp.c_str()));
          }
        }
        {
          fan::vec3 angle = layer.shape.get_angle();
          if (fan::graphics::gui::drag_float("angle", &angle, fan::math::radians(1))) {
            layer.shape.set_angle(angle);
          }
        }
        {
          fan::vec2 rotation_point = layer.shape.get_rotation_point();
          
          if (fan::graphics::gui::drag_float("rotation_point", &rotation_point, 0.1, -tile_size.max() * 2, tile_size.max() * 2)) {
            layer.shape.set_rotation_point(rotation_point);
          }
        }/*
        {
          fan::vec3 rotation_vector = layer.shape.get_rotation_vector();
          if (ImGui::SliderFloat3("rotation vector", rotation_vector.data(), 0, 1)) {
            layer.shape.set_rotation_vector(rotation_vector);
          }
        }*/
        {
          uint32_t flags = layer.shape.get_flags();
          if (fan::graphics::gui::input_int("special flags", (int*)&flags, 1, 1)) {
            layer.shape.set_flags(flags);
          }
        }
        {
          fan::color color = layer.shape.get_color();
          if (fan::graphics::gui::color_edit4("color", &color)) {
            layer.shape.set_color(color);
          }
        }
        {
          int mesh_property = (int)layer.tile.mesh_property;
          if (fan::graphics::gui::slider_int("mesh flags", &mesh_property, 0, (int)mesh_property_t::size - 1)) {
            layer.tile.mesh_property = (mesh_property_t)mesh_property;
          } 
        }
        //if (layer.tile.mesh_property == mesh_property_t::sensor) {
        //  if (ImGui::BeginChild("Actions")) {
        //    ImGui::Text("Actions:");
        //    if (ImGui::Combo("##actions", (int*)&layer.tile.action, actions_e_strings, std::size(actions_e_strings))) {

        //    }

        //    switch(layer.tile.action) {
        //      case actions_e::open_model: {
        //        static std::vector<std::string> model_names;
        //        if (ImGui::Button("select models")) {
        //          model_names.clear();
        //          models_open_file_dialog.load("json", &model_names);
        //        }

        //        if (models_open_file_dialog.is_finished()) {
        //          layer.tile.object_names.clear();
        //          for (const auto& model_name : model_names) {
        //            std::string base_filename = model_name.substr(model_name.find_last_of("/\\") + 1);
        //            std::string extension = base_filename.substr(base_filename.find_last_of('.') + 1);
        //            base_filename = base_filename.substr(0, base_filename.find_last_of('.'));
        //            layer.tile.object_names.push_back(base_filename + "." + extension);
        //          }
        //          models_open_file_dialog.finished = true;
        //        }

        //        /*static int key = fan::input_enum_to_array_index(layer.tile.key);
        //        if (ImGui::ComboAutoSelect("Open Key", key, fan::input_strings, gloco->item_getter1, ImGuiComboFlags_HeightRegular)) {
        //          layer.tile.key = fan::array_index_to_enum_input(key);
        //        }*/
        //        break;
        //      }
        //      default: {
        //        fan::throw_error("unimplemented");
        //      }
        //    }

        //  }
        //  ImGui::EndChild();
        //}
      }
    }
    fan::graphics::gui::end();
  }

  void handle_brush_settings_window() {
    if (fan::graphics::gui::begin("Brush settings")) {
      {
        int idx = (int)brush.depth - shape_depths_t::max_layer_depth / 2;
        if (fan::graphics::gui::drag_int("depth", (int*)&idx, 1, 0, shape_depths_t::max_layer_depth)) {
          brush.depth = idx + shape_depths_t::max_layer_depth / 2;
        }
      }
      {
        int idx = (int)brush.mode;
        if (fan::graphics::gui::combo("mode", (int*)&idx, brush.mode_names, std::size(brush.mode_names))) {
          brush.mode = (brush_t::mode_e)idx;
        }
      }
      {
        int idx = (int)brush.type;
        if (fan::graphics::gui::combo("type", (int*)&idx, brush.type_names, std::size(brush.type_names))) {
          brush.type = (brush_t::type_e)idx;
        }
      }

      {
        if (fan::graphics::gui::slider_int("jitter", &brush.jitter, 0, brush.size.min())) {
          grid_visualize.highlight_hover.set_size(tile_size * brush.size);
        }
      }
      {
        
        if (fan::graphics::gui::drag_float("jitter_chance", &brush.jitter_chance, 1, 0, 0.01)) {

        }
      }

      {
        static int default_value = 0;
        if (fan::graphics::gui::combo("dynamics angle", &default_value, brush.dynamics_names, std::size(brush.dynamics_names))) {
          brush.dynamics_angle = (brush_t::dynamics_e)default_value;
        }
      }
      {
        static int default_value = 0;
        if (fan::graphics::gui::combo("dynamics color", &default_value, brush.dynamics_names, std::size(brush.dynamics_names))) {
          brush.dynamics_color = (brush_t::dynamics_e)default_value;
        }
      }


      {
        if (fan::graphics::gui::slider_int("size", &brush.size, 1, 4096)) {
          grid_visualize.highlight_hover.set_size(tile_size * brush.size);
        }
      }
      {
        if (fan::graphics::gui::slider_float("tile size", &brush.tile_size, 0.1, 1)) {
          //brush.tile_size = brush.tile_size;
        }
      }

      fan::graphics::gui::drag_float("angle", &brush.angle);
      {
        std::string temp = brush.id;
        temp.resize(max_id_len);
        if (fan::graphics::gui::input_text("id", &temp)) {
          brush.id = temp.substr(0, strlen(temp.c_str()));
        }
      }

      fan::graphics::gui::color_edit4("color", &brush.color);

      switch (brush.type) {
      case brush_t::type_e::physics_shape: {
        {
          if (fan::graphics::gui::slider_float("offset", &brush.offset, -1, 1)) {
            
          }
        }
        {
          static int default_value = 0;
          if (fan::graphics::gui::combo("Physics shape type", &default_value, brush.physics_type_names, std::size(brush.physics_type_names))) {
            brush.physics_type = default_value;
          }
        }
        {
          static int default_value = 0;
          if (fan::graphics::gui::combo("Physics body type", &default_value, brush.physics_body_type_names, std::size(brush.physics_body_type_names))) {
            brush.physics_body_type = default_value;
          }
        }
        {
          static bool default_value = 0;
          if (fan::graphics::gui::toggle_button("Physics shape draw", &default_value)) {
            brush.physics_draw = default_value;
          }
        }
        {
          static fan::physics::shape_properties_t shape_properties;
          if (fan::graphics::gui::drag_float("Physics shape friction", &shape_properties.friction, 0.01, 0, 1)) {
            brush.physics_shape_properties.friction = shape_properties.friction;
          }
          if (fan::graphics::gui::drag_float("Physics shape density", &shape_properties.density, 0.01, 0, 1)) {
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
        }

        break;
      }
      }
    }
    fan::graphics::gui::end();
  }

  void handle_lighting_settings_window() {
    if (fan::graphics::gui::begin("lighting settings")) {
      if (fan::graphics::gui::color_edit3("ambient", &gloco->lighting.ambient)) {

      }
    }
    fan::graphics::gui::end();
  }

  void handle_physics_settings_window() {
    if (fan::graphics::gui::begin("physics settings")) {
      fan::vec2 gravity = gloco->physics_context.get_gravity();
      if (fan::graphics::gui::drag_float("gravity", &gravity, 0.01)) {
        gloco->physics_context.set_gravity(gravity);
      }
    }
    fan::graphics::gui::end();
  }

  void handle_pick_tile() {
    fan::vec2i position;
    if (window_relative_to_grid(gloco->get_mouse_position(), &position)) {
      fan::vec2i grid_position = position;
      convert_draw_to_grid(grid_position);
      grid_position /= tile_size * 2;
      auto found = map_tiles.find(fan::vec2i(grid_position.x, grid_position.y));
      if (found != map_tiles.end()) {
        auto& layers = found->second.layers;
        uint32_t idx = find_layer_shape(layers);
        if (idx == invalid) {
          idx = find_top_layer_shape(layers);
        }
        current_image_indices.clear();
        current_tile_images.clear();
        current_tile_images.resize(1);
        if (idx != invalid || idx < brush.depth) {
          uint16_t st = layers[idx].shape.get_shape_type();
          if (st == (uint16_t)loco_t::shape_type_t::sprite ||
            st == (uint16_t)loco_t::shape_type_t::unlit_sprite) {
            current_tile_images[0].push_back({
              .ti = layers[idx].shape.get_tp(),
              .image_name = layers[idx].tile.image_name
            });
          }
        }
      }
    }
  }

  void handle_select_tile() {
    fan::vec2i position;
    if (window_relative_to_grid(gloco->get_mouse_position(), &position)) {
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

  void handle_imgui() {
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
      if (gloco->window.key_pressed(fan::key_left_control)) {
        handle_pick_tile();
      }
      else if (gloco->window.key_pressed(fan::key_left_shift)) {
        handle_select_tile();
      }
    }

    handle_tile_brush();

    fan::graphics::gui::end();
  }

  void render() {
    handle_imgui();
  }

  /*
  * header
  header version 4 byte
  map size 8 byte
  tile size 8 byte
  element count
  struct size x byte
  shape data{
    ...
  }
  */

  struct layer_info_t {
    std::string layer_name;
    uint16_t depth;
  };

  void fout(const std::string& filename) {
#if defined(fan_json)
    previous_file_name = filename;

    fan::json ostr;
    ostr["version"] = 1;
    ostr["map_size"] = map_size;
    ostr["tile_size"] = tile_size;
    ostr["lighting.ambient"] = gloco->lighting.ambient;
    ostr["gravity"] = gloco->physics_context.get_gravity();

    fan::json tiles = fan::json::array();

    for (auto& i : map_tiles) {
      for (auto& j : i.second.layers) {
        // hardcoded to only tile_t
        fan::json tile;
        if (j.shape.get_size() == 0) {
          fan::print("warning out size 0", j.tile.position);
        }
        fan::graphics::shape_serialize(j.shape, &tile);
        tile["image_name"] = j.tile.image_name;
        tile["mesh_property"] = j.tile.mesh_property;
        tile["id"] = j.tile.id;
        tile["action"] = j.tile.action;
        tile["key"] = j.tile.key;
        tile["key_state"] = j.tile.key_state;
        tile["object_names"] = j.tile.object_names;
        tiles.push_back(tile);
      }
    }
    for (auto& i : physics_shapes) {
      for (auto& j : i.second) {
        fan::json tile;
        if (j.visual.get_size() == 0) {
          fan::print("warning out size 0", j.visual.get_position());
        }
        fan::graphics::shape_serialize(j.visual, &tile);
        tile["image_name"] = tile_t().image_name;
        tile["mesh_property"] = fte_t::mesh_property_t::physics_shape;
        tile["id"] = j.id;
        tile["action"] = tile_t().action;
        tile["key"] = tile_t().key;
        tile["key_state"] = tile_t().key_state;
        tile["object_names"] = tile_t().object_names;

        /*
        uint8_t type = type_e::box;
    uint8_t body_type = fan::physics::body_type_e::static_body;
    bool draw = false;
    fan::physics::shape_properties_t shape_properties;
        */

        fan::json physics_shape_data;
        physics_shape_data["type"] = j.type;
        physics_shape_data["body_type"] = j.body_type;
        physics_shape_data["draw"] = j.draw;
        physics_shape_data["friction"] = j.shape_properties.friction;
        physics_shape_data["density"] = j.shape_properties.density;
        physics_shape_data["fixed_rotation"] = j.shape_properties.fixed_rotation;
        physics_shape_data["presolve_events"] = j.shape_properties.presolve_events;
        physics_shape_data["is_sensor"] = j.shape_properties.is_sensor;
        tile["physics_shape_data"] = physics_shape_data;
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
#else
    fan::throw_error("fan_json not enabled");
    __unreachable();
#endif
  }

  /*
* header
header version 4 byte
map size 8 byte
tile size 8 byte
struct size x byte
shape data{
  ...
}
*/
  void fin(const std::string& filename) {
#if defined(fan_json)
    if (texturepack.texture_list.size() == 0) {
      fan::print("open valid texturepack");
      return;
    }
    invalidate_selection();
    previous_file_name = filename;
    std::string out;
    fan::io::file::read(filename, &out);
    fan::json json = fan::json::parse(out);
    if (json["version"] != 1) {
      fan::throw_error("version mismatch");
    }

    map_size = json["map_size"];
    tile_size = json["tile_size"];
    if (json.contains("gravity")) {
      gloco->physics_context.set_gravity(json["gravity"]);
    }
    gloco->lighting.ambient = json["lighting.ambient"];
    map_tiles.clear();
    visual_layers.clear();
    visual_shapes.clear();
    physics_shapes.clear();
    resize_map();

    fan::graphics::shape_deserialize_t it;
    loco_t::shape_t shape;
    while (it.iterate(json["tiles"], &shape)) {
      const auto& shape_json = *(it.data.it - 1);
      if (shape_json["mesh_property"] == fte_t::mesh_property_t::physics_shape) {
        auto& physics_shape = physics_shapes[shape.get_position().z];
        physics_shape.resize(physics_shape.size() + 1);
        auto& physics_element = physics_shape.back();
        shape.set_camera(camera->camera);
        shape.set_viewport(camera->viewport);
        shape.set_image(grid_visualize.collider_color);
        if (shape_json.contains("physics_shape_data")) {
          physics_element.id  = shape_json["id"];
          const fan::json& physics_shape_data = shape_json["physics_shape_data"];
          physics_element.type = physics_shape_data["type"];
          physics_element.body_type = physics_shape_data["body_type"];
          physics_element.draw = physics_shape_data["draw"];
          physics_element.shape_properties.friction = physics_shape_data["friction"] ;
          physics_element.shape_properties.density = physics_shape_data["density"] ;
          physics_element.shape_properties.fixed_rotation = physics_shape_data["fixed_rotation"] ;
          physics_element.shape_properties.presolve_events = physics_shape_data["presolve_events"];
          physics_element.shape_properties.is_sensor = physics_shape_data["is_sensor"];
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
      layer->tile.image_name = shape_json.value("image_name", "");
      layer->tile.id = shape_json["id"];
      layer->tile.mesh_property = (mesh_property_t)shape_json["mesh_property"];

      layer->shape = shape;

      switch (layer->tile.mesh_property) {
        case fte_t::mesh_property_t::none: {
          loco_t::texturepack_t::ti_t ti;
          if (texturepack.qti(layer->tile.image_name, &ti)) {
            fan::throw_error("failed to read image from .fte - editor save file corrupted");
          }
          layer->shape.set_camera(camera->camera);
          layer->shape.set_viewport(camera->viewport);
          layer->shape.set_tp(&ti);
          break;
        }
        case fte_t::mesh_property_t::light: {
          layer->shape = fan::graphics::light_t{{
            .camera = camera,
            .position = shape.get_position(),
            .size = layer->tile.size,
            .color = layer->tile.color
          }};
          visual_shapes[shape.get_position()].shape = fan::graphics::sprite_t{{
            .camera = camera,
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
        // Add layer_info to visual_layers
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

  fan::vec2i map_size{ 6, 6 };
  fan::vec2i tile_size{ 32, 32 };

  struct tile_info_t {
    loco_t::texturepack_t::ti_t ti;
    std::string image_name;
    mesh_property_t mesh_property = mesh_property_t::none;
  };

  struct current_tile_t {
    fan::vec2i position = 0;
    shapes_t::global_t::layer_t* layer = nullptr;
    uint32_t layer_index;
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

  current_tile_t current_tile;
  fan::vec2i current_tile_brush_count;
  std::vector<std::vector<tile_info_t>> current_tile_images;
  struct sort_by_y_t {
    bool operator()(const fan::vec2i& a, const fan::vec2i& b) const {
      if (a.y == b.y)
        return a.x < b.x;
      else
        return a.y < b.y;
    }
  };
  std::map<fan::vec2i, int, sort_by_y_t> current_image_indices;

  std::unordered_map<fan::vec2i, shapes_t::global_t, vec2i_hasher> map_tiles;
  // key by depth, slow to loop all and check aabb collision

  std::unordered_map<f32_t, std::vector<fte_t::physics_shapes_t>> physics_shapes;
  struct visualize_t {
    loco_t::shape_t shape;
  };
  std::unordered_map<fan::vec3, visualize_t, vec3_hasher> visual_shapes;

  struct visual_layer_t {
    std::unordered_map<fan::vec2i, bool, vec2i_hasher> positions;
    std::string text = "default";
    bool visible = true;
  };

  // depth key
  std::map<uint16_t, visual_layer_t> visual_layers;

  loco_t::texturepack_t texturepack;

  fan::vec2 texturepack_size{};
  fan::vec2 texturepack_single_image_size{};
  std::vector<tile_info_t> texturepack_images;

  struct {
    loco_t::shape_t background;
    loco_t::shape_t highlight_selected;
    loco_t::shape_t highlight_hover;
    loco_t::shape_t grid;

    loco_t::image_t highlight_color;
    loco_t::image_t collider_color;
    loco_t::image_t light_color;
    bool render_grid = false;
  }grid_visualize;


  struct brush_t {
    enum class mode_e : uint8_t{
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
    static constexpr const char* type_names[] = { "Texture", "Physics shape", "Light"};
    type_e type = type_e::texture;

    enum class dynamics_e : uint8_t {
      original,
      randomize
    };
    static constexpr const char* dynamics_names[] = { "Original", "Randomize" };
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

    // physics stuff
    fan::vec2 offset = 0;
    uint8_t physics_type = physics_shapes_t::type_e::box;
    static constexpr const char* physics_type_names[] = { "Box", "Circle" };
    uint8_t physics_body_type = fan::physics::body_type_e::static_body;
    static constexpr const char* physics_body_type_names[] = { "Static", "Kinematic", "Dynamic" };
    bool physics_draw = false;
    fan::physics::shape_properties_t physics_shape_properties;
  }brush;

  struct {
    f32_t zoom = 1;
    bool move = false;
    fan::vec2 pos = 0;
    fan::vec2 size = 0;
    fan::vec2 offset = 0;
    fan::vec2 window_related_mouse_pos = 0;
    fan::vec2 zoom_offset = 0;
  }viewport_settings;

  struct {
    bool hovered = false;
  }editor_settings;

  fan::vec3i prev_grid_position = 999999;

  loco_t::image_t transparent_texture;
  fan::vec2i copy_buffer_region = 0;
  std::vector<shapes_t::global_t> copy_buffer;
  loco_t::camera_impl_t* camera = nullptr;
  std::function<void(int)> modify_cb = [](int) {};

  std::string previous_file_name;
  loco_t::shape_t visual_line;
  int original_image_width = 2048;
};

#endif