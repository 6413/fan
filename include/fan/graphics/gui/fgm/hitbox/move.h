case builder_draw_type_t::hitbox: {
  pile->builder.hitbox.set_position(
    &pile->context,
    pile->editor.selected_type_index,
    position + pile->editor.move_offset
  );

  break;
}