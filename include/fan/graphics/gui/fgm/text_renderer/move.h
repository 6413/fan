case builder_draw_type_t::text_renderer: {
  pile->builder.tr.set_position(
    &pile->context,
    pile->editor.selected_type_index,
    pile->editor.get_mouse_position(pile) + pile->editor.move_offset
  );

  break;
}