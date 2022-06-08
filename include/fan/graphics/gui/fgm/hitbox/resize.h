case builder_draw_type_t::hitbox: {

  fan::vec2 offset = pile->window.get_mouse_position() - pile->editor.click_position;
  fan::vec2 position = pile->builder.hitbox.get_position(&pile->context, pile->editor.selected_type_index);
  fan::vec2 size = pile->builder.hitbox.get_size(&pile->context, pile->editor.selected_type_index);

  switch (pile->editor.resize_stage) {
    case 0: {
      if (size.x * 2 - offset.x >= 32) {
        size.x += -offset.x / 2;
        position.x += offset.x / 2;
        pile->editor.click_position.x = pile->window.get_mouse_position().x;
      }
      if (size.y * 2 - offset.y >= 32) {
        size.y += -offset.y / 2;
        position.y -= -offset.y / 2;
        pile->editor.click_position.y = pile->window.get_mouse_position().y;
      }
      break;
    }
    case 1: {
      /*if (size.x * 2 - offset.x >= 32) {
      size.x += -offset.x / 2;
      position.x -= -offset.x / 2;
      pile->editor.click_position.x = pile->window.get_mouse_position().x;
      }*/
      if (size.y * 2 - offset.y >= 32) {
        size.y += -offset.y / 2;
        position.y -= -offset.y / 2;
        pile->editor.click_position.y = pile->window.get_mouse_position().y;
      }
      break;
    }
    case 2: {
      if (size.x * 2 + offset.x >= 32) {
        size.x += offset.x / 2;
        position.x += offset.x / 2;
        pile->editor.click_position.x = pile->window.get_mouse_position().x;
      }
      if (size.y * 2 - offset.y >= 32) {
        size.y += -offset.y / 2;
        position.y -= -offset.y / 2;
        pile->editor.click_position.y = pile->window.get_mouse_position().y;
      }
      break;
    }
    case 3: {
      if (size.x * 2 + offset.x >= 32) {
        size.x += offset.x / 2;
        position.x += offset.x / 2;
        pile->editor.click_position.x = pile->window.get_mouse_position().x;
      }
      /*if (size.y * 2 - offset.y >= 32) {
      size.y += -offset.y / 2;
      position.y -= -offset.y / 2;
      pile->editor.click_position.y = pile->window.get_mouse_position().y;
      }*/
      break;
    }
    case 4: {
      if (size.x * 2 + offset.x >= 32) {
        size.x += offset.x / 2;
        position.x += offset.x / 2;
        pile->editor.click_position.x = pile->window.get_mouse_position().x;
      }
      if (size.y * 2 + offset.y >= 32) {
        size.y += offset.y / 2;
        position.y += offset.y / 2;
        pile->editor.click_position.y = pile->window.get_mouse_position().y;
      }
      break;
    }
    case 5: {
      /*if (size.x * 2 - offset.x >= 32) {
      size.x += -offset.x / 2;
      position.x -= -offset.x / 2;
      pile->editor.click_position.x = pile->window.get_mouse_position().x;
      }*/
      if (size.y * 2 + offset.y >= 32) {
        size.y += offset.y / 2;
        position.y += offset.y / 2;
        pile->editor.click_position.y = pile->window.get_mouse_position().y;
      }
      break;
    }
    case 6: {
      if (size.x * 2 - offset.x >= 32) {
        size.x -= offset.x / 2;
        position.x -= -offset.x / 2;
        pile->editor.click_position.x = pile->window.get_mouse_position().x;
      }
      if (size.y * 2 + offset.y >= 32) {
        size.y += offset.y / 2;
        position.y += offset.y / 2;
        pile->editor.click_position.y = pile->window.get_mouse_position().y;
      }
      break;
    }
    case 7: {
      if (size.x * 2 - offset.x >= 32) {
        size.x += -offset.x / 2;
        position.x += offset.x / 2;
        pile->editor.click_position.x = pile->window.get_mouse_position().x;
      }
      /*if (size.y * 2 - offset.y >= 32) {
      size.y += -offset.y / 2;
      position.y -= -offset.y / 2;
      pile->editor.click_position.y = pile->window.get_mouse_position().y;
      }*/
      break;
    }
  }

  pile->builder.hitbox.set_position(&pile->context, pile->editor.selected_type_index,
    position
  );
  pile->builder.hitbox.set_size(&pile->context, pile->editor.selected_type_index, size);
  pile->editor.update_resize_rectangles(pile);
}