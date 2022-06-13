builder_types.m_button_event.set_on_input(pile,
  [](
  fan::window_t* window,
  fan::opengl::context_t* context,
  uint32_t index,
  uint16_t key,
  fan::key_state key_state,
  fan_2d::graphics::gui::mouse_stage mouse_stage,
  void* user_ptr
) {

  if (key != fan::mouse_left) {
    return;
  }
  if (mouse_stage != fan_2d::graphics::gui::mouse_stage::inside && key_state == fan::key_state::press) {
    return;
  }

  pile_t* pile = (pile_t*)user_ptr;

  if (!pile->editor.is_inside_types_viewport(pile, pile->editor.get_mouse_position(pile)) && key_state == fan::key_state::press) {
    return;
  }

  switch (index) {
    #include _FAN_PATH(graphics/gui/fgm/includes/create.h)
  }
});

pile->window.add_mouse_move_callback(pile,
  [](fan::window_t* window, const fan::vec2i& position, void* user_ptr) {
  pile_t* pile = (pile_t*)user_ptr;

  if (!(pile->editor.flags & flags_t::ignore_moving)) {

    if (!(pile->editor.flags & flags_t::moving)) {
      return;
    }

    switch (pile->editor.selected_type) {
      #include _FAN_PATH(graphics/gui/fgm/includes/move.h)
      default: {
        fan::throw_error("failed to move current shape - add it to includes");
      }
    }
    pile->editor.update_resize_rectangles(pile);
  }
  if (pile->editor.flags & flags_t::resizing) {
    switch (pile->editor.selected_type) {
      #include _FAN_PATH(graphics/gui/fgm/includes/resize.h)
    }
  }

});

pile->window.add_keys_callback(pile,
  [](fan::window_t* window, uint16_t key, fan::key_state key_state, void* user_ptr)
{
  pile_t* pile = (pile_t*)user_ptr;

  switch (key) {
    #include _FAN_PATH(graphics/gui/fgm/input/mouse_left.h)
    #include _FAN_PATH(graphics/gui/fgm/input/mouse_scroll_up.h)
    #include _FAN_PATH(graphics/gui/fgm/input/mouse_scroll_down.h)
    #include _FAN_PATH(graphics/gui/fgm/input/key_delete.h)
    #include _FAN_PATH(graphics/gui/fgm/input/copy_paste.h)
  }

});

pile->editor.resize_rectangles.m_button_event.set_on_input(pile,
  [](
  fan::window_t* window,
  fan::opengl::context_t* context,
  uint32_t index,
  uint16_t key,
  fan::key_state key_state,
  fan_2d::graphics::gui::mouse_stage mouse_stage,
  void* user_ptr
) {

  if (key != fan::mouse_left) {
    return;
  }

  pile_t* pile = (pile_t*)user_ptr;

  if (key_state == fan::key_state::release) {
    pile->editor.flags &= ~flags_t::ignore_properties_close;
    pile->editor.flags &= ~flags_t::ignore_moving;
    pile->editor.flags &= ~flags_t::resizing;
    return;
  }

  if (mouse_stage != fan_2d::graphics::gui::mouse_stage::inside) {
    return;
  }

  if (!pile->editor.is_inside_builder_viewport(pile, pile->editor.get_mouse_position(pile))) {
    return;
  }
  
  pile->editor.resize_stage = index;
  pile->editor.click_position = pile->editor.get_mouse_position(pile);
  pile->editor.flags |= flags_t::ignore_properties_close;
  pile->editor.flags |= flags_t::ignore_moving;
  pile->editor.flags |= flags_t::resizing;

});

pile->editor.builder_types.m_key_event.set_on_focus_loss_callback(pile,
  [](fan::window_t* window, fan::graphics::context_t* context, uint32_t i, void* userptr) {

    pile_t* pile = (pile_t*)userptr;

    switch (i) {
      case 4: {
        fan::utf16_string sw = pile->editor.builder_types.get_text(window, context, i);
        std::string s(sw.begin(), sw.end());
        s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
        break;
      }
  }
});