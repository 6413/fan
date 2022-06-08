case fan::key_delete: {
  switch (key_state) {
    case fan::key_state::press: {
      switch (pile->editor.selected_type) {
        #include _FAN_PATH(graphics/gui/fgm/includes/erase_active.h)
      }
    }
  }
}