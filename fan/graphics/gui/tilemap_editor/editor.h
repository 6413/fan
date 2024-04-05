#pragma once

#include <map>

struct fte_t {
  static constexpr int max_id_len = 20;
  static constexpr fan::vec2 default_button_size{ 100, 30 };
  static constexpr fan::vec2 tile_viewer_sprite_size{ 64, 64 };
  static constexpr fan::color highlighted_tile_color = fan::color(0.5, 0.5, 1);
  static constexpr fan::color highlighted_selected_tile_color = fan::color(0.5, 0, 0, 0.1);

  static constexpr f32_t scroll_speed = 1.2;
  static constexpr uint32_t invalid = -1;

  struct shape_depths_t {
    static constexpr uint32_t max_layer_depth = 0xffff;
    static constexpr int cursor_highlight_depth = 10000;
  };

  fan::string file_name = "file.fte";

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

#include "common.h"

  enum class event_type_e {
    none,
    add,
    remove
  };

  uint32_t find_top_layer_shape(const auto& vec) {
    uint32_t found = -1;
    int64_t depth = -1;
    for (int i = 0; i < vec.size(); ++i) {
      if (vec[i].tile.position.z > depth) {
        depth = vec[i].tile.position.z;
        found = i;
      }
    }
    return found;
  };

  uint32_t find_layer_shape(const auto& vec) {
    uint32_t found = -1;
    for (int i = 0; i < vec.size(); ++i) {
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
    gloco->shapes.line_grid.sb_set_vi(
      grid_visualize.line_grid,
      &loco_t::shapes_t::line_grid_t::vi_t::grid_size,
      map_size
    );
    if (grid_visualize.render_grid) {
      grid_visualize.line_grid.set_size(map_size * (tile_size / 2) * 2);
    }
    else {
      grid_visualize.line_grid.set_size(0);
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
    map_tiles.clear();
    resize_map();
  }

  bool window_relative_to_grid(const fan::vec2& window_relative_position, fan::vec2i* in) {
    fan::vec2 p = gloco->translate_position(window_relative_position, &camera->viewport, &camera->camera) / 2 + camera->camera.get_position() / 2;
    fan::vec2 ws = gloco->window.get_size();
    if (map_size.x % 2) {
      p.x += tile_size.x / 2;
    }
    if (map_size.y % 2) {
      p.y += tile_size.y / 2;
    }
    fan::vec2i f = (p / tile_size).floor();
    p = f * tile_size * 2;
    if (!(map_size.x % 2)) {
      p.x += tile_size.x;
    }
    if (!(map_size.y % 2)) {
      p.y += tile_size.y;
    }
    //p -= tile_size / 2;
    //if (!(map_size.x % 2)) {
    //  p += tile_size;
    //}
    //else {
      //p += fan::vec2i((map_size.x % 2) * tile_size.x, (map_size.y % 2) * tile_size.y);
     // p += fan::vec2i(!(map_size.x % 2) * -tile_size.x * 0.5, !(map_size.y % 2) * -tile_size.y);
    //}
    //fan::print(p.floor());
    *in = p.floor();

    return fan_2d::collision::rectangle::point_inside_no_rotation(p / tile_size, 0, map_size);
  }

  void convert_draw_to_grid(fan::vec2i& p) {
    if (!(map_size.x % 2)) {
      p.x -= tile_size.x;
    }
    if (!(map_size.y % 2)) {
      p.y -= tile_size.y;
    }
    p /= 2;
  }

  void convert_grid_to_draw(fan::vec2i& p) {
    p *= 2;
    p += tile_size;
  }

  struct properties_t {
    fan::string texturepack_name;
    fan::graphics::camera_t* camera = nullptr;
  };

  void open(const properties_t& properties) {
    loco_t::image_t::load_properties_t lp;
    lp.visual_output = loco_t::image_t::sampler_address_mode::clamp_to_border;
    lp.min_filter = fan::opengl::GL_NEAREST;
    lp.mag_filter = fan::opengl::GL_NEAREST;
    texturepack.open_compiled(properties.texturepack_name, lp);
    if (properties.camera == nullptr) {
      camera = gloco->default_camera;
    }
    else {
      camera = properties.camera;
    }

    gloco->window.add_mouse_move_callback([this](const auto& d) {
      if (viewport_settings.move) {
        fan::vec2 move_off = (d.position - viewport_settings.offset) / viewport_settings.zoom * 2;
        camera->camera.set_position(viewport_settings.pos - move_off);        
      }
      fan::vec2i p;
      {
        if (window_relative_to_grid(d.position, &p)) {
          //convert_draw_to_grid(p);
          grid_visualize.highlight_hover.set_position(fan::vec2(p));
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
          viewport_settings.pos = camera->camera.get_position();
          break;
        }
        case fan::mouse_scroll_up: {
          if (gloco->window.key_pressed(fan::key_left_control)) {
            brush.depth += 1;
            brush.depth = std::min((uint32_t)brush.depth, shape_depths_t::max_layer_depth);
          }
          else if (gloco->window.key_pressed(fan::key_left_shift)) {
            brush.size += 1;
            grid_visualize.highlight_hover.set_size(tile_size * brush.size);
          }
          else {
            viewport_settings.zoom *= scroll_speed;
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
      if (ImGui::IsAnyItemActive()) {
        return;
      }

      switch (d.key) {
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

      ii.image_hash = image.hash;

      texturepack_images.push_back(ii);
    });

    grid_visualize.background = fan::graphics::sprite_t{ {
      .camera = camera,
      .position = fan::vec3(viewport_settings.pos, 0),
      .size = 0,
      .image = &gloco->transparent_texture,
    } };

    grid_visualize.highlight_color.create(fan::colors::red, 1);
    grid_visualize.collider_color.create(fan::color(0, 0.5, 0, 0.5), 1);
    grid_visualize.light_color.create(fan::color(0, 0.0, 0.5, 0.5), 1);

    grid_visualize.highlight_hover = fan::graphics::unlit_sprite_t{ {
      .camera = camera,
      .position = fan::vec3(viewport_settings.pos, shape_depths_t::cursor_highlight_depth),
      .size = tile_size,
      .image = &grid_visualize.highlight_color,
      .blending = true
    } };
    static bool init = true;
    static loco_t::image_t highlight_selected_texture;
    if (init) {
      init = false;
      highlight_selected_texture.create(highlighted_selected_tile_color, 1);
    }
    grid_visualize.highlight_selected = fan::graphics::unlit_sprite_t{ {
      .position = fan::vec3(viewport_settings.pos, shape_depths_t::cursor_highlight_depth - 1),
      .size = 0,
      .image = &highlight_selected_texture,
      .blending = true
    } };

    {
      fan::vec2 p = 0;
      p = ((p - tile_size) / tile_size).floor() * tile_size;
      grid_visualize.highlight_hover.set_position(p);
    }

    //// update viewport sizes
    //gloco->process_frame();

    camera->camera.set_position(viewport_settings.pos);

    loco_t::shapes_t::line_grid_t::properties_t p;
    p.viewport = &camera->viewport;
    p.camera = &camera->camera;
    p.position = fan::vec3(0, 0, shape_depths_t::cursor_highlight_depth + 1);
    p.size = 0;
    p.color = fan::color::rgb(0, 128, 255);

    grid_visualize.line_grid = p;

    resize_map();
  }
  void close() {
    texturepack.close();
  }

  bool is_in_constraints(const fan::vec2i& position) {
    if (position.x > map_size.x * tile_size.x || position.x < -(map_size.x * tile_size.x)) {
      return false;
    }
    if (position.y > map_size.y * tile_size.y || position.y < -(map_size.y * tile_size.y)) {
      return false;
    }
    return true;
  }

  bool is_in_constraints(fan::vec2i& position, int& j, int& i) {
    position += (-brush.size / 2) * tile_size * 2 + tile_size * 2 * fan::vec2(j, i);
    return is_in_constraints(position);
  }

  f32_t get_snapped_angle() {
    switch (fan::random::value_i64(0, 3)) {
    case 0: return 0;
    case 1: return fan::math::pi * 0.5;
    case 2: return fan::math::pi;
    case 3: return fan::math::pi * 1.5;
    }
  };

  bool handle_tile_push(fan::vec2i& position, int& j, int& i) {
    if (brush.jitter) {
      if ( brush.jitter_chance <= fan::random::value_f32(0, 1)) {
        return true;
      }
      position += (fan::random::vec2i(-brush.jitter, brush.jitter) * 2 + 1) * tile_size + tile_size;
    }
    if (!is_in_constraints(position, j, i)) {
      return true;
    }
    fan::vec2 start_idx = -(current_tile_brush_count / 2).floor();
    position += start_idx * tile_size * 2;

    f32_t inital_x = position.x;

    for (auto& i : current_tile_images) {
      for (auto& tile : i) {
        fan::vec2i grid_position = position;
        convert_draw_to_grid(grid_position);

        grid_position /= tile_size;
        auto& layers = map_tiles[grid_position].layers;
        uint32_t idx = find_layer_shape(layers);

        if (idx == invalid) {
          layers.resize(layers.size() + 1);
          layers.back().tile.position = fan::vec3(position, brush.depth);
          layers.back().tile.image_hash = tile.image_hash;
          layers.back().tile.id = brush.id;
          // todo fix
          layers.back().tile.mesh_property = mesh_property_t::none;
          if (brush.type != brush_t::type_e::light)
            layers.back().shape = fan::graphics::sprite_t{ {
                .camera = camera,
                .position = fan::vec3(position, brush.depth),
                .size = tile_size * brush.tile_size,
                .angle = brush.dynamics_angle == brush_t::dynamics_e::randomize ?
                      get_snapped_angle() : brush.angle,
                .color = brush.dynamics_color == brush_t::dynamics_e::randomize ? fan::random::color() : brush.color,
                .blending = true
            } };
          switch (brush.type) {
            case brush_t::type_e::texture: {
              if (layers.back().shape.set_tp(&tile.ti)) {
                fan::print("failed to load image");
              }
              break;
            }
            case brush_t::type_e::sensor:
            case brush_t::type_e::collider: {
              layers.back().shape.set_image(&grid_visualize.collider_color);
              if (brush.type == brush_t::type_e::sensor) {
                layers.back().tile.mesh_property = mesh_property_t::sensor;
              }
              else {
                layers.back().tile.mesh_property = mesh_property_t::collider;
              }
              break;
            }
            case brush_t::type_e::light: {
              loco_t::shapes_t::light_t::properties_t lp;
              auto& shape = layers.back().shape;
              lp.position = fan::vec3(position, brush.depth);
              lp.size = tile_size * brush.tile_size;
              lp.color = brush.dynamics_color == brush_t::dynamics_e::randomize ? fan::random::color() : brush.color;
              layers.back().shape = lp;
              layers.back().tile.mesh_property = mesh_property_t::light;
              visual_shapes[lp.position].shape = fan::graphics::sprite_t{{
                  .camera = camera,
                  .position = fan::vec3(fan::vec2(lp.position), lp.position.z + 1),
                  .size = tile_size,
                  .image = &grid_visualize.light_color,
                  .blending = true
              }};
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
              }
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
                  layer.tile.image_hash = tile.image_hash;
                  break;
                }
                case brush_t::type_e::sensor:
                case brush_t::type_e::collider: {
                  layer.shape.set_image(&grid_visualize.collider_color);
                  if (brush.type == brush_t::type_e::sensor) {
                    layer.tile.mesh_property = mesh_property_t::sensor;
                  }
                  else {
                    layer.tile.mesh_property = mesh_property_t::collider;
                  }
                  break;
                }
                case brush_t::type_e::light: {
                  loco_t::shapes_t::light_t::properties_t lp;
                  auto& shape = layer.shape;
                  lp.position = shape.get_position();
                  lp.size = shape.get_size();
                  lp.angle = shape.get_angle();
                  lp.color = shape.get_color();
                  lp.camera = &camera->camera;
                  lp.viewport = &camera->viewport;
                  layer.shape = lp;
                  layer.tile.mesh_property = mesh_property_t::light;
                  visual_shapes[lp.position].shape = fan::graphics::sprite_t{{
                      .position = fan::vec3(fan::vec2(lp.position), lp.position.z + 1),
                      .size = tile_size,
                      .image = &grid_visualize.light_color,
                      .blending = true
                  }};
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
    if (brush.jitter) {
      if (brush.jitter_chance <= fan::random::value_f32(0, 1)) {
        return true;
      }
      position += (fan::random::vec2i(-brush.jitter, brush.jitter) * 2 + 1) * tile_size + tile_size;
    }
    if (!is_in_constraints(position, j, i)) {
      return true;
    }
    fan::vec2i grid_position = position;
    convert_draw_to_grid(grid_position);
    grid_position /= tile_size;
    
    auto found = map_tiles.find(grid_position);
    if (found != map_tiles.end()) {
      auto& layers = found->second.layers;
      uint32_t idx = find_layer_shape(layers);
      if (idx != invalid || idx < layers.size()) {
        switch (layers[idx].tile.mesh_property) {
          case mesh_property_t::light:{
            visual_shapes.erase(layers[idx].shape.get_position());
            break;
          }
        }
        layers.erase(layers.begin() + idx);
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
      position /= tile_size;
      return true;
    }
    return false;
  }

  void handle_tile_brush() {
    if (!editor_settings.hovered)
      return;

    switch (brush.mode) {
      case brush_t::mode_e::draw: {
        fan::vec2i position;
        bool is_mouse_left_down = ImGui::IsMouseDown(fan::window_input::fan_to_imguikey(fan::mouse_left));
        bool is_mouse_right_down = ImGui::IsMouseDown(fan::window_input::fan_to_imguikey(fan::mouse_right));
        bool is_ctrl_pressed = gloco->window.key_pressed(fan::key_left_control);
        bool is_shift_pressed = gloco->window.key_pressed(fan::key_left_shift);

        if (is_mouse_left_down && !is_ctrl_pressed && !is_shift_pressed) {
          handle_tile_action(position, [this](auto...args) {auto ret = handle_tile_push(args...);  modify_cb(0); return ret; });
        }

        if (is_mouse_right_down) {
          handle_tile_action(position, [this](auto...args) {auto ret = handle_tile_erase(args...);  modify_cb(0); return ret; });
        }

        break;
      }
      case brush_t::mode_e::copy: {

        static fan::vec2i copy_src;
        static fan::vec2i copy_dst;
        bool is_mouse_left_clicked = ImGui::IsMouseClicked(fan::window_input::fan_to_imguikey(fan::mouse_left));
        bool is_mouse_left_held = ImGui::IsMouseDown(fan::window_input::fan_to_imguikey(fan::mouse_left));
        bool is_mouse_left_released = ImGui::IsMouseReleased(fan::window_input::fan_to_imguikey(fan::mouse_left));
        bool is_mouse_right_clicked = ImGui::IsMouseClicked(fan::window_input::fan_to_imguikey(fan::mouse_right));

        static std::vector<loco_t::shape_t> select;

        if (is_mouse_left_clicked) {
          copy_buffer.clear();
          fan::vec2i mouse_grid_pos;
          if (mouse_to_grid(mouse_grid_pos)) {
            copy_src = mouse_grid_pos;
          }
        }
        if (is_mouse_left_held) {
          fan::vec2i mouse_grid_pos;
          if (mouse_to_grid(mouse_grid_pos)) {
            select.clear();
            fan::vec2i src = copy_src;
            fan::vec2i dst = mouse_grid_pos;
            copy_dst = dst;
            // 2 is coordinate specific
            int stepx = (src.x <= dst.x) ? 1 : -1;
            int stepy = (src.y <= dst.y) ? 1 : -1;
            for (int j = src.y; j != dst.y + stepy; j += stepy) {
              for (int i = src.x; i != dst.x + stepx; i += stepx) {
                select.push_back(fan::graphics::rectangle_t{ {
                    .position = fan::vec3(fan::vec2(i, j) * tile_size * 2 + tile_size, shape_depths_t::cursor_highlight_depth),
                    .size = tile_size,
                    .color = fan::color(1, 0, 0, 0.5),
                    .blending = true
                } });
              }
            }
          }
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
              }
            }
            else {
              for (int j = src.y; j != dst.y + stepy; j += stepy) {
                for (int i = src.x; i != dst.x + stepx; i += stepx) {
                  auto found = map_tiles.find(fan::vec2i(i, j));
                  if (found != map_tiles.end()) {
                    copy_buffer.push_back(found->second);
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
              if (is_in_constraints(current_pos * tile_size)) {
                auto& tile = map_tiles[current_pos];
                tile = i;
                int layer = 0;
                for (auto& t : tile.layers) {
                  fan::vec2 op = t.shape.get_position();
                  fan::vec2 offset = op - *(fan::vec2i*)&t.tile.position;
                  fan::print(offset);
                  fan::vec2 draw_pos = current_pos * tile_size * 2 + tile_size + offset;
                  if (is_in_constraints(draw_pos)) {
                    t.shape.set_position(fan::vec2(draw_pos));
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
    if (!window_relative_to_grid(gloco->get_mouse_position(), &position)) {
      if (editor_settings.hovered && current_tile.layer != nullptr) {
        invalidate_selection();
      }
      return;
    }
    fan::vec2i grid_position = position;
    convert_draw_to_grid(grid_position);
    grid_position /= tile_size;
    if (grid_position == prev_grid_position) {
      return;
    }
    prev_grid_position = grid_position;
    for (int i = 0; i < brush.size.y; ++i) {
      for (int j = 0; j < brush.size.x; ++j) {
        if (action(position, j, i)) {
          continue;
        }
      }
    }
  }

  bool handle_editor_window(fan::vec2& editor_size) {
    if (ImGui::Begin("Editor")) {
      fan::vec2 window_size = gloco->window.get_size();
      fan::vec2 viewport_size = ImGui::GetContentRegionAvail();
      fan::vec2 viewport_pos = fan::vec2(ImGui::GetWindowPos() + fan::vec2(0, ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y * 2));
      fan::vec2 offset = viewport_size - viewport_size / viewport_settings.zoom;
      fan::vec2 s = viewport_size;
      camera->camera.set_ortho(
        fan::vec2(-s.x, s.x) / viewport_settings.zoom,
        fan::vec2(-s.y, s.y) / viewport_settings.zoom
      );

      //gloco->default_camera->camera.set_camera_zoom(viewport_settings.zoom);
      camera->viewport.set(viewport_pos, viewport_size, window_size);
      editor_size = ImGui::GetContentRegionAvail();
      viewport_settings.size = editor_size;
      ImGui::SetWindowFontScale(1.5);
      ImGui::TextColored(ImVec4(1, 1, 1, 1), fan::format("brush type: {}", brush.type_names[(uint8_t)brush.type]).c_str());
      ImGui::TextColored(ImVec4(1, 1, 1, 1), fan::format("brush depth: {}", (int)brush.depth).c_str());

      auto& style = ImGui::GetStyle();
      fan::vec2 prev_item_spacing = style.ItemSpacing;

      style.ItemSpacing = fan::vec2(1);

      for (auto& i : current_tile_images) {
        int idx = 0;
        for (auto& j : i) {
          if (idx != 0) {
            ImGui::SameLine();
          }
          idx++;
          
          //ImGui::Button((fan::to_string(j.ti.position.x, 1).c_str()), tile_viewer_sprite_size);
          ImGui::Image(
            (ImTextureID)j.ti.image->get_texture(),
            tile_viewer_sprite_size / std::max(1.f, current_tile_brush_count.x / 5.f),
            j.ti.position / j.ti.image->size,
            j.ti.position / j.ti.image->size +
            j.ti.size / j.ti.image->size
          );
        }
      }

      style.ItemSpacing = prev_item_spacing;

      if (ImGui::Begin("Layer window")) {
        struct data_t {
          fan::string text;
        };
        static std::vector<data_t> layers{ {.text = "default"} };
        for (int i = 0; i < layers.size(); ++i) {
          layers[i].text.resize(32);
          ImGui::Text(fan::format("Layer {}", i).c_str());
          ImGui::SameLine();
          ImGui::InputText(fan::format("##layer{}", i).c_str(), layers[i].text.data(), layers[i].text.size());
        }
        if (ImGui::Button("+")) {
          layers.push_back(data_t{ .text = "default" });
        }
        ImGui::SameLine();
        if (ImGui::Button("-")) {
          layers.pop_back();
        }
      }
        ImGui::End();
    }
    else {
      camera->viewport.zero();
      return true;
    }
    editor_settings.hovered = ImGui::IsWindowHovered();
    ImGui::End();
    return false;
  }

  bool handle_editor_settings_window() {
    if (ImGui::Begin("Editor settings")) {
      {
        if (ImGui::InputInt2("map size", map_size.data())) {
          resize_map();
        }
        if (ImGui::InputInt2("tile size", tile_size.data())) {
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

        if (ImGui::Checkbox("render grid", &grid_visualize.render_grid)) {
          if (grid_visualize.render_grid) {
            grid_visualize.line_grid.set_size(map_size * (tile_size / 2) * 2);
          }
          else {
            grid_visualize.line_grid.set_size(0);
          }
        }

        // use ImGui::Dummy here
        fan::vec2 window_size = ImGui::GetWindowSize();
        fan::vec2 cursor_pos(
          window_size.x - default_button_size.x - ImGui::GetStyle().WindowPadding.x,
          window_size.y - default_button_size.y - ImGui::GetStyle().WindowPadding.y
        );
        ImGui::SetCursorPos(cursor_pos);
        if (ImGui::Button("Save")) {
          fout(file_name);
        }
        cursor_pos.x += default_button_size.x / 2;
        ImGui::SetCursorPos(cursor_pos);
        if (ImGui::Button("Quit")) {
          return true;
        }
      }
    }
    ImGui::End();
    return false;
  }

  void handle_tiles_window() {
    if (ImGui::Begin("tiles", nullptr, ImGuiWindowFlags_HorizontalScrollbar)) {

      {
        f32_t x_size = ImGui::GetContentRegionAvail().x;
        static int offset = 0;
        ImGui::DragInt("offset", &offset, 1, 0);
        int divider = std::sqrt(texturepack_images.size());
        int images_per_row = divider + offset;

        auto& style = ImGui::GetStyle();
        fan::vec2 prev_item_spacing = style.ItemSpacing;

        style.ItemSpacing = fan::vec2(1);

        current_tile_brush_count = 0;

        for (uint32_t i = 0; i < texturepack_images.size(); i++) {
          auto& node = texturepack_images[i];

          fan::vec2i grid_index(i % (divider + offset), i / (divider + offset));

          bool selected = false;
          if (current_image_indices.find(grid_index) != current_image_indices.end()) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1.0f, 0.0f, 0.0f, 0.5f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1.0f, 0.0f, 0.0f, 0.5f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.0f, 0.0f, 0.5f));
            selected = true;
          }

          if (ImGui::ImageButton(
            (fan::string("##ibutton") + std::to_string(i)).c_str(),
            (void*)(intptr_t)node.ti.image->get_texture(),
            tile_viewer_sprite_size,
            node.ti.position / node.ti.image->size,
            node.ti.position / node.ti.image->size + node.ti.size / node.ti.image->size
          )) {

          }
          if ((ImGui::IsMouseClicked(0) || (ImGui::IsMouseDown(0) && 
            ImGui::IsMouseDragging(0))) && 
            ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly)) {
            current_image_indices[grid_index] = i;
          }
          else if ((ImGui::IsMouseClicked(1) || (ImGui::IsMouseDown(1) &&
            ImGui::IsMouseDragging(1))) &&
            ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly)) {
            auto found = current_image_indices.find(grid_index);
            if (found != current_image_indices.end()) {
              current_image_indices.erase(found);
            }
          }

          if (selected) {
            ImGui::PopStyleColor(3);
          }

          if (images_per_row != 0 && (i + 1) % images_per_row != 0) {
            ImGui::SameLine();
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
    }
  }

  void handle_tile_settings_window() {
    if (ImGui::Begin("Tile settings")) {
      if (current_tile.layer != nullptr) {
        auto& layer = current_tile.layer[current_tile.layer_index];

        {
          fan::vec2 offset = fan::vec2(layer.shape.get_position()) - current_tile.position;

          if (fan_imgui_dragfloat(offset, 0.1, 0, 0)) {
            layer.shape.set_position(fan::vec2(current_tile.position) + offset);
          }
        }
        {
          fan::vec2 tile_size = layer.shape.get_size();
          if (ImGui::DragFloat2("tile size", tile_size.data())) {
            layer.shape.set_size(tile_size);
          }
        }
         {
          fan::string temp = layer.tile.id;
          temp.resize(max_id_len);
          if (ImGui::InputText("id", temp.data(), temp.size())) {
            layer.tile.id = temp.substr(0, strlen(temp.c_str()));
          }
        }
        {
           fan::vec3 angle = layer.shape.get_angle();
          if (ImGui::DragFloat3("angle", angle.data(), fan::math::radians(1))) {
            layer.shape.set_angle(angle);
          }
        }
        {
          fan::vec2 rotation_point = layer.shape.get_rotation_point();
          
          if (fan_imgui_dragfloat(rotation_point, 0.1, -tile_size.max() * 2, tile_size.max() * 2)) {
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
          if (ImGui::ColorEdit4("color", (float*)brush.color.data())) {
            layer.shape.set_color(brush.color);
          }
        }
        {
          int mesh_property = (int)layer.tile.mesh_property;
          if (ImGui::SliderInt("mesh flags", &mesh_property, 0, (int)mesh_property_t::size - 1)) {
            layer.tile.mesh_property = (mesh_property_t)mesh_property;
          }
        }
      }
    }
    ImGui::End();
  }

  void handle_brush_settings_window() {
    if (ImGui::Begin("Brush settings")) {
      {
        int idx = (int)brush.mode;
        if (ImGui::Combo("mode", (int*)&idx, brush.mode_names, std::size(brush.mode_names))) {
          brush.mode = (brush_t::mode_e)idx;
        }
      }
      {
        int idx = (int)brush.type;
        if (ImGui::Combo("type", (int*)&idx, brush.type_names, std::size(brush.type_names))) {
          brush.type = (brush_t::type_e)idx;
        }
      }

      {
        if (ImGui::SliderInt("jitter", &brush.jitter, 0, brush.size.min())) {
          grid_visualize.highlight_hover.set_size(tile_size * brush.size);
        }
      }
      {
        
        if (fan_imgui_dragfloat(brush.jitter_chance, 1, 0, 1)) {

        }
      }

      {
        static int default_value = 0;
        if (ImGui::Combo("dynamics angle", &default_value, brush.dynamics_names, std::size(brush.dynamics_names))) {
          brush.dynamics_angle = (brush_t::dynamics_e)default_value;
        }
      }
      {
        static int default_value = 0;
        if (ImGui::Combo("dynamics color", &default_value, brush.dynamics_names, std::size(brush.dynamics_names))) {
          brush.dynamics_color = (brush_t::dynamics_e)default_value;
        }
      }


      {
        if (ImGui::SliderInt2("size", brush.size.data(), 1, 1e+6)) {
          grid_visualize.highlight_hover.set_size(tile_size * brush.size);
        }
      }
      {
        if (ImGui::SliderInt("tile size", &brush.tile_size, 1, 100)) {
          //brush.tile_size = brush.tile_size;
        }
      }

      ImGui::DragFloat3("angle", brush.angle.data());
      {
        fan::string temp = brush.id;
        temp.resize(max_id_len);
        if (ImGui::InputText("id", temp.data(), temp.size())) {
          brush.id = temp.substr(0, strlen(temp.c_str()));
        }
      }

      ImGui::ColorEdit4("color", (float*)brush.color.data());
    }
    ImGui::End();
  }

  void handle_lighting_settings_window() {
    if (ImGui::Begin("lighting settings")) {
      if (ImGui::ColorEdit4("ambient", gloco->lighting.ambient.data())) {

      }
    }
    ImGui::End();
  }

  void handle_pick_tile() {
    fan::vec2i position;
    if (window_relative_to_grid(gloco->get_mouse_position(), &position)) {
      fan::vec2i grid_position = position;
      convert_draw_to_grid(grid_position);
      grid_position /= tile_size;
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
          current_tile_images[0].push_back({
            .ti = layers[idx].shape.get_tp(),
            .image_hash = layers[idx].tile.image_hash
          });
        }
      }
    }
  }

  void handle_select_tile() {
    fan::vec2i position;
    if (window_relative_to_grid(gloco->get_mouse_position(), &position)) {
      fan::vec2i grid_position = position;
      convert_draw_to_grid(grid_position);
      grid_position /= tile_size;
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
      ImGui::End();
      return;
    }

    if (handle_editor_settings_window()) {
      return;
    }

    handle_tiles_window();

    handle_tile_settings_window();

    handle_brush_settings_window();
    
    handle_lighting_settings_window();

    if (editor_settings.hovered && ImGui::IsMouseDown(fan::window_input::fan_to_imguikey(fan::mouse_left))) {
      if (gloco->window.key_pressed(fan::key_left_control)) {
        handle_pick_tile();
      }
      else if (gloco->window.key_pressed(fan::key_left_shift)) {
        handle_select_tile();
      }
    }

    handle_tile_brush();

    ImGui::End();
  }

  fan::graphics::imgui_element_t main_view =
    fan::graphics::imgui_element_t([&] {handle_imgui(); });

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
  void fout(const fan::string& filename) {
    fan::string ostr;
    ostr.append((char*)&current_version, sizeof(current_version));
    ostr.append((char*)map_size.data(), sizeof(map_size));
    ostr.append((char*)tile_size.data(), sizeof(tile_size));
    ostr.append((char*)gloco->lighting.ambient.data(), sizeof(gloco->lighting.ambient));

    for (auto& i : map_tiles) {
      fan::mp_t<current_version_t::shapes_t> shapes;

      shapes.iterate([&]<auto i0, typename T>(T & l) {
        fan::mp_t<T> shape;
        shape.init(this, &i.second);

        fan::string shape_str;
        shape.iterate([&]<auto i1, typename T2>(T2 & v) {
          if constexpr (std::is_same_v<T2, fan::string>) {
            uint64_t string_length = v.size();
            shape_str.append((char*)&string_length, sizeof(string_length));
            shape_str.append(v);
          }
          else if constexpr (fan_requires_rule(T2, typename T2::value_type)) {
            if constexpr (std::is_same_v<T2, std::vector<typename T2::value_type>>) {
              uint32_t len = v.size();
              shape_str.append((char*)&len, sizeof(uint32_t));
              for (auto& ob : v) {
                fan::mp_t<std::remove_reference_t<decltype(ob)>> tile;
                *dynamic_cast<std::remove_reference_t<decltype(ob)>*>(&tile) = ob;
                tile.iterate([&]<auto i, typename T3>(T3 & v){
                  if constexpr(std::is_same_v<T3, fan::string>) {
                    uint32_t len = v.size();
                    shape_str.append((char*)&len, sizeof(len));
                    shape_str.append(v.data(), len);
                  }
                  else {
                    shape_str.append((char*)&v, sizeof(v));
                  }
                });
              }
            }
          }
          else {
            shape_str.append((char*)&v, sizeof(T2));
          }
        });
        uint32_t struct_size = shape_str.size();
        ostr.append((char*)&struct_size, sizeof(struct_size));

        ostr += shape_str;
      });
    }
    fan::io::file::write(filename, ostr, std::ios_base::binary);
    fan::print("file saved to:" + filename);
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
  void fin(const fan::string& filename) {
#include _FAN_PATH(graphics/gui/tilemap_editor/loader_versions/1.h)
  }

  fan::vec2i map_size{ 64, 64 };
  fan::vec2i tile_size{ 32, 32 };

  struct tile_info_t {
    loco_t::texturepack_t::ti_t ti;
    uint64_t image_hash;
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
  struct visualize_t {
    loco_t::shape_t shape;
  };
  std::unordered_map<fan::vec3, visualize_t, vec3_hasher> visual_shapes;

  loco_t::texturepack_t texturepack;

  std::vector<tile_info_t> texturepack_images;

  struct {
    loco_t::shape_t background;
    loco_t::shape_t highlight_selected;
    loco_t::shape_t highlight_hover;
    loco_t::shape_t line_grid;

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
    static constexpr const char* mode_names[] = {"Draw", "Copy"};
    enum class type_e : uint8_t {
      texture,
      collider,
      sensor,
      light
    };
    static constexpr const char* type_names[] = { "Texture", "Collider", "Sensor", "Light"};
    type_e type = type_e::texture;

    enum class dynamics_e : uint8_t {
      original,
      randomize
    };
    static constexpr const char* dynamics_names[] = { "Original", "Randomize" };
    dynamics_e dynamics_angle = dynamics_e::original;
    dynamics_e dynamics_color = dynamics_e::original;

    fan::vec2i size = 1;
    int tile_size = 1;
    fan::vec3 angle = 0;
    f32_t depth = 1;
    int jitter = 0;
    f32_t jitter_chance = 0.33;
    fan::string id;
    fan::color color = fan::color(1);
  }brush;

  struct {
    f32_t zoom = 1;
    bool move = false;
    fan::vec2 pos = 0;
    fan::vec2 size = 0;
    fan::vec2 offset = 0;
  }viewport_settings;

  struct {
    bool hovered = false;
  }editor_settings;

  fan::vec2i prev_grid_position = 999999;

  fan::vec2i copy_buffer_region = 0;
  std::vector<shapes_t::global_t> copy_buffer;
  fan::graphics::camera_t* camera = nullptr;
  fan::function_t<void(int)> modify_cb = [](int) {};
};
