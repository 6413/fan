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
  }

  break;
}