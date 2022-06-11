case fan::key_c: {
  if (!pile->window.key_pressed(fan::key_left_control)) {
    break;
  }
  if (key_state != fan::key_state::press) {
    break;
  }
  if (pile->editor.selected_type == fan::uninitialized) {
    break;
  }

  pile->editor.copied_type = pile->editor.selected_type;
  pile->editor.copied_type_index = pile->editor.selected_type_index;

  break;
}
case fan::key_v: {
  if (!pile->window.key_pressed(fan::key_left_control)) {
    break;
  }
  if (key_state != fan::key_state::press) {
    break;
  }
  if (pile->editor.copied_type == fan::uninitialized) {
    break;
  }

  switch (pile->editor.copied_type) {
    case builder_draw_type_t::sprite: {

      fan_2d::graphics::sprite_t::properties_t sprite_p;
      sprite_p.position = pile->builder.sprite.get_position(&pile->context, pile->editor.copied_type_index);
      sprite_p.size = pile->builder.sprite.get_size(&pile->context, pile->editor.copied_type_index);
      sprite_p.image = pile->builder.sprite.get_image(&pile->context, pile->editor.copied_type_index);
      sprite_p.texture_coordinates =
      {
        pile->builder.sprite.get_texture_coordinates(&pile->context, pile->editor.copied_type_index, 0),
        pile->builder.sprite.get_texture_coordinates(&pile->context, pile->editor.copied_type_index, 1),
        pile->builder.sprite.get_texture_coordinates(&pile->context, pile->editor.copied_type_index, 2),
        pile->builder.sprite.get_texture_coordinates(&pile->context, pile->editor.copied_type_index, 4)
      };
      pile->builder.sprite.push_back(&pile->context, sprite_p);

      pile->editor.depth_map_push(pile, builder_draw_type_t::sprite, pile->builder.sprite.size(&pile->context) - 1);

      pile->editor.builder_draw_type = editor_t::builder_draw_type_t::sprite;
      pile->editor.builder_draw_type_index = pile->builder.sprite.size(&pile->context) - 1;

      pile->editor.close_build_properties(pile);

      pile->editor.selected_type = editor_t::builder_draw_type_t::sprite;
      pile->editor.selected_type_index = pile->editor.builder_draw_type_index;

      click_collision_t click_collision;
      click_collision.builder_draw_type = pile->editor.selected_type;
      click_collision.builder_draw_type_index = pile->editor.selected_type_index;
      pile->editor.open_build_properties(pile, click_collision);

      break;
    }
    case builder_draw_type_t::text_renderer: {
      fan_2d::graphics::gui::text_renderer_t::properties_t tr_p;
      tr_p.position = pile->builder.tr.get_position(&pile->context, pile->editor.copied_type_index);
      tr_p.text = pile->builder.tr.get_text(&pile->context, pile->editor.copied_type_index);
      tr_p.font_size = pile->builder.tr.get_font_size(&pile->context, pile->editor.copied_type_index);
      tr_p.text_color = pile->builder.tr.get_text_color(&pile->context, pile->editor.copied_type_index);
      tr_p.outline_color = pile->builder.tr.get_outline_color(&pile->context, pile->editor.copied_type_index);
      tr_p.outline_size = pile->builder.tr.get_outline_size(&pile->context, pile->editor.copied_type_index);
     
      pile->builder.tr.push_back(&pile->context, tr_p);

      pile->editor.depth_map_push(pile, builder_draw_type_t::text_renderer, pile->builder.tr.size(&pile->context) - 1);

      pile->editor.builder_draw_type = editor_t::builder_draw_type_t::text_renderer;
      pile->editor.builder_draw_type_index = pile->builder.tr.size(&pile->context) - 1;

      pile->editor.close_build_properties(pile);

      pile->editor.selected_type = editor_t::builder_draw_type_t::text_renderer;
      pile->editor.selected_type_index = pile->editor.builder_draw_type_index;

      click_collision_t click_collision;
      click_collision.builder_draw_type = pile->editor.selected_type;
      click_collision.builder_draw_type_index = pile->editor.selected_type_index;
      pile->editor.open_build_properties(pile, click_collision);

      break;
    }
    case builder_draw_type_t::hitbox: {

      fan_2d::graphics::sprite_t::properties_t sprite_p;
      sprite_p.position = pile->window.get_mouse_position();
      sprite_p.size = pile->editor.builder_types.get_size(&pile->window, &pile->context, 0);
      sprite_p.image = pile->builder.hitbox.get_image(&pile->context, pile->builder.hitbox.size(&pile->context) - 1);
      pile->builder.hitbox.push_back(&pile->context, sprite_p);

      pile->editor.depth_map_push(pile, builder_draw_type_t::hitbox, pile->builder.hitbox.size(&pile->context) - 1);

      pile->editor.builder_draw_type = editor_t::builder_draw_type_t::hitbox;
      pile->editor.builder_draw_type_index = pile->builder.hitbox.size(&pile->context) - 1;

      pile->editor.close_build_properties(pile);

      pile->editor.selected_type = editor_t::builder_draw_type_t::hitbox;
      pile->editor.selected_type_index = pile->editor.builder_draw_type_index;

      bool result;
      uint32_t i = 0;
      for (; result = pile->editor.check_for_colliding_hitbox_ids(std::to_string(i)); i++) {}
      pile->editor.hitbox_ids.push_back(std::to_string(i));

      click_collision_t click_collision;
      click_collision.builder_draw_type = pile->editor.selected_type;
      click_collision.builder_draw_type_index = pile->editor.selected_type_index;
      pile->editor.open_build_properties(pile, click_collision);

      break;
    }
    case builder_draw_type_t::button: {
      fan_2d::graphics::gui::rectangle_text_button_sized_t::properties_t button_p;
      button_p.position = pile->builder.button.get_position(&pile->window, &pile->context, pile->editor.copied_type_index);
      button_p.text = pile->builder.button.get_text(&pile->window, &pile->context, pile->editor.copied_type_index);
      button_p.font_size = pile->builder.button.get_font_size(&pile->window, &pile->context, pile->editor.copied_type_index);
      button_p.theme = pile->builder.button.get_theme(&pile->window, &pile->context, pile->editor.copied_type_index);

      pile->builder.button.push_back(&pile->window, &pile->context, button_p);

      pile->editor.depth_map_push(pile, builder_draw_type_t::button, pile->builder.button.size(&pile->window, &pile->context) - 1);

      pile->editor.builder_draw_type = editor_t::builder_draw_type_t::text_renderer;
      pile->editor.builder_draw_type_index = pile->builder.button.size(&pile->window, &pile->context) - 1;

      pile->editor.close_build_properties(pile);

      pile->editor.selected_type = editor_t::builder_draw_type_t::text_renderer;
      pile->editor.selected_type_index = pile->editor.builder_draw_type_index;

      click_collision_t click_collision;
      click_collision.builder_draw_type = pile->editor.selected_type;
      click_collision.builder_draw_type_index = pile->editor.selected_type_index;
      pile->editor.open_build_properties(pile, click_collision);

      break;
    }
  }

  break;
}