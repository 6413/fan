case builder_draw_type_t::rectangle_text_button_sized: {
  pile->builder_draw_types.rtbs.set_position(
    &pile->window,
    &pile->context,
    pile->editor_draw_types.selected_draw_type_index,
    position + pile->editor_draw_types.moving_position
  );
  break;
}