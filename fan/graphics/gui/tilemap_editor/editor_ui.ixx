module;

#if defined(FAN_2D)
#endif

export module fan.graphics.gui.tilemap_editor.core:ui;

#if defined (FAN_WINDOW)

import std;

#if defined(FAN_2D)
#if defined(FAN_GUI) && defined(FAN_PHYSICS_2D)

import fan.physics.types;
import fan.texture_pack.tp0;
import fan.window.input;
import fan.types;
import fan.graphics.loco;
import fan.graphics.shapes;
import fan.graphics;
import fan.graphics.common_context;
import fan.graphics.gui.types;
import fan.graphics.gui.base;
import fan.graphics.gui;
import fan.graphics.gui.tilemap_editor.core;
import fan.graphics.gui.text_logger;
import fan.types.vector;
import fan.types.color;
import fan.math;
import fan.file_dialog;
import fan.print.error;
import fan.graphics.algorithm.raycast_grid;
import fan.math.intersection;
import fan.physics.b2_integration;
import fan.io.file;

export namespace fan::graphics::gui::tilemap_editor::ui {

  template <typename enum_t, std::size_t N>
  bool combo_enum(const char* label, enum_t& val, const char* const (&names)[N]) {
    int idx = static_cast<int>(val);
    if (fan::graphics::gui::combo(label, &idx, N, [&](int i) -> const char* { return names[i]; })) {
      val = static_cast<enum_t>(idx);
      return true;
    }
    return false;
  }

  void draw_id_label(const std::string& id, const fan::vec3& world_pos, f32_t base_font_size, f32_t zoom, fan::graphics::render_view_t* render_view, fan::graphics::gui::draw_list_t* draw_list);
  void draw_id_labels(fte_t& editor);
  bool handle_editor_window(fte_t& editor, fan::vec2& editor_size);
  bool handle_editor_settings_window(fte_t& editor);
  void handle_tiles_window(fte_t& editor);
  void handle_tile_settings_window(fte_t& editor);
  void handle_brush_settings_window(fte_t& editor);
  void handle_lighting_settings_window(fte_t& editor);
  void handle_physics_settings_window(fte_t& editor);
  void handle_custom_tools_window(fte_t& editor);
  void handle_tile_brush(fte_t& editor);
  void handle_gui(fte_t& editor);

}

#endif
#endif

#endif