module;

export module fan.graphics.editor:viewport;

import std;

import fan.types.vector;

export namespace fan::graphics::editor {
  struct viewport_settings_t {
    bool move = false;
    bool editor_hovered = false;
    fan::vec2 pos = 0;
    fan::vec2 size = 0;
    fan::vec2 start_pos = 0;
    fan::vec2 offset = 0;
  };

  struct viewport_t {
    static fan::vec2 get_mouse_position(const viewport_settings_t& settings, const fan::vec2& mouse_pos, f32_t zoom, const fan::vec2& window_padding) {
      fan::vec2 pos = (mouse_pos - settings.start_pos + window_padding) - settings.size / 2.0f;
      return settings.pos + pos / zoom;
    }
  };
}