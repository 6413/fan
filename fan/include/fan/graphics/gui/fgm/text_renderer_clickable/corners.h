case builder_draw_type_t::text_renderer_clickable: {

  fan::vec2 middle = pile->builder.trc.get_position(&pile->window, &pile->context, pile->editor.selected_type_index);
  fan::vec2 size = pile->builder.trc.get_size(&pile->window, &pile->context, pile->editor.selected_type_index);

  positions[0] = middle + fan::vec2(-size.x, -size.y);
  positions[1] = middle + fan::vec2(0, -size.y);
  positions[2] = middle + fan::vec2(size.x, -size.y);
  positions[3] = middle + fan::vec2(-size.x, 0);
  positions[4] = middle + fan::vec2(size.x, 0);
  positions[5] = middle + fan::vec2(-size.x, size.y);
  positions[6] = middle + fan::vec2(0, size.y);
  positions[7] = middle + fan::vec2(size.x, size.y);

  break;
}