case builder_draw_type_t::button: {
  pile->builder.button.set_position(
    &pile->window,
    &pile->context,
    pile->editor.selected_type_index,
    pile->editor.get_mouse_position(pile) + pile->editor.move_offset
  );

  break;
}