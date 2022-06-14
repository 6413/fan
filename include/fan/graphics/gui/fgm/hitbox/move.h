case builder_draw_type_t::hitbox: {
  pile->builder.hitbox.set_position(
    &pile->context,
    pile->editor.selected_type_index,
    pile->editor.get_mouse_position(pile) + pile->editor.move_offset
  );

  break;
}