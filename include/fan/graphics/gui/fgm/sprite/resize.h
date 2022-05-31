case builder_draw_type_t::sprite: {
  switch (pile->editor.resize_stage) {
    case 0: {
      fan::vec2 offset = pile->window.get_mouse_position() - pile->editor.click_position;

      fan::vec2 size = pile->builder.sprite.get_size(&pile->context, pile->editor.selected_type_index);
      if (offset.x < 5) {
        offset.x = 5;
      }
      if (offset.y < 5) {
        offset.y = 5;
      }
      pile->builder.sprite.set_size(&pile->context, pile->editor.selected_type_index, offset.abs());

      fan::vec2 position = pile->builder.sprite.get_position(&pile->context, pile->editor.selected_type_index);

      pile->builder.sprite.set_position(&pile->context, pile->editor.selected_type_index,
        position + fan::vec2(
        (fan::math::abs(offset.x) - fan::math::abs(size.x)) * (!std::signbit(offset.x) ? 1 : -1), 0
      ));

      break;
    }
    case 3: {
      fan::vec2 offset = pile->window.get_mouse_position() - pile->editor.click_position;

      fan::vec2 size = pile->builder.sprite.get_size(&pile->context, pile->editor.selected_type_index);

      offset = offset.abs();

      if (offset.x < 5) {
        offset.x = 5;
      }
      if (offset.y < 5) {
        offset.y = 5;
      }
      fan::print(offset, pile->window.get_mouse_position(), pile->editor.click_position);
      pile->builder.sprite.set_size(&pile->context, pile->editor.selected_type_index, offset.abs());

      fan::vec2 position = pile->builder.sprite.get_position(&pile->context, pile->editor.selected_type_index);

      pile->builder.sprite.set_position(&pile->context, pile->editor.selected_type_index, 
        position - (offset - size)
      );

      break;
    }
  }
}