case builder_draw_type_t::text_renderer_clickable: {
  pile->builder_draw_types.trc.set_position(
    &pile->window,
    &pile->context,
    pile->editor_draw_types.selected_draw_type_index,
    position + pile->editor_draw_types.moving_position
  );
  break;
}