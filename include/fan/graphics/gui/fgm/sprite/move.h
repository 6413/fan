case builder_draw_type_t::sprite: {
  pile->builder.sprite.set_position(
    &pile->context,
    pile->editor.selected_type_index,
    position + pile->editor.move_offset
  );

  if (pile->editor.resize_rectangles.size(&pile->window, &pile->context)) {
    pile->editor.update_resize_rectangles(pile);
  }

  break;
}