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

    void json_write(fan::json& j) const;
    void json_read(const fan::json& j);
  };

  // --- Declarations (Implemented in .cpp) ---
  std::uint32_t find_layer_shape(const std::vector<shapes_t::global_t::layer_t>& vec, bool top = false);
  void resize_map();
  void reset_map();
  bool window_relative_to_grid(const fan::vec2& window_relative_position, fan::vec2i* in);
  void convert_draw_to_grid(fan::vec2i& p);
  void convert_grid_to_draw(fan::vec2i& p);
  void open(const properties_t& properties);
  void close();
  bool is_in_constraints(const fan::vec2i& position);
  bool is_in_constraints(fan::vec2i& position, int j, int i);
  f32_t get_snapped_angle();
  fan::vec2 snap_to_tile_center(const fan::vec2& world_pos);
  bool apply_jitter(fan::vec2i& position);
  sprite_t make_sprite(const fan::vec3& pos, const fan::vec2& size, const fan::color& color, render_view_t* rv, image_t image);
  image_t get_marker_image(mesh_property_t type);
  bool handle_tile_push(fan::vec2i& position, int& pj, int& pi);
  bool handle_tile_erase(fan::vec2i& position, int& j, int& i);
  bool mouse_to_grid(fan::vec2i& position);
  bool physics_shape_exists(const fan::vec3& position, const fan::vec2& size);
  void invalidate_selection();
  void open_texture_pack(const std::string& path);
  void apply_brush_settings(const std::string& id, f32_t depth, const fan::vec2& size, const fan::color& color, const fan::vec3& angle = fan::vec3(0));
  void handle_pick_tile();
  void handle_select_tile();
  bool same_visual(const fte_t::shapes_t::global_t::layer_t& a, const fte_t::shapes_t::global_t::layer_t& b);
  void fout(std::string filename);
  void fin(const std::string& filename, const std::source_location& callers_path = std::source_location::current());
  
  // --- Inline Templates (Must stay in interface) ---
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

  struct terrain_generator_t {
    terrain_generator_t();

    void init();
    void rebuild_colors();
    void iterate();
    void reset();
    void render();
    void insert_selected_tiles(int depth);

    tile_world_generator_t tile_world;
    std::vector<sprite_t> rects;
    grid_t visual_grid;
    interactive_camera_t ic;
    fan::vec2 prev_viewport_size = 0;
  } terrain_generator;

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