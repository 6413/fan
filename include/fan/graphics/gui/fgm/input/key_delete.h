case fan::key_delete: {
  switch (key_state) {
    case fan::key_state::press: {
      switch (pile->editor.selected_type) {
        case builder_draw_type_t::sprite: {
          #include _FAN_PATH(graphics/gui/fgm/sprite/erase_active.h)
          break;
        }
        case builder_draw_type_t::text_renderer: {
          #include _FAN_PATH(graphics/gui/fgm/text_renderer/erase_active.h)
          break;
        }
        case builder_draw_type_t::hitbox: {
          #include _FAN_PATH(graphics/gui/fgm/hitbox/erase_active.h)
          break;
        }
      }
    }
  }
}