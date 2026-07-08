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

  void open(const properties_t& properties);

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

  bool handle_tile_push(fan::vec2i& position, int& pj, int& pi);

  bool handle_tile_erase(fan::vec2i& position, int& j, int& i);

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

  void fout(std::string filename);

  void fin(const std::string& filename, const std::source_location& callers_path = std::source_location::current());

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
