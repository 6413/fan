case builder_draw_type_t::text_renderer_clickable: {
  switch (key_state) {
    case fan::key_state::press: {
      pile->editor_draw_types.flags |= flags_t::moving;
      pile->editor_draw_types.moving_position = pile->window.get_mouse_position();

      decltype(pile->builder_draw_types.trc)::properties_t trc_p;
      trc_p.position = pile->window.get_mouse_position();
      // fetch same size as gui
      trc_p.hitbox_size = pile->editor_draw_types.builder_types.get_size(&pile->window, &pile->context, 0);
      trc_p.hitbox_position = trc_p.position;
      trc_p.text = "Clickable text";
      trc_p.font_size = constants::gui_size;
      trc_p.text_color = fan::colors::white;
      pile->builder_draw_types.trc.push_back(&pile->window, &pile->context, trc_p);

      pile->editor_draw_types.builder_draw_type = editor_draw_types_t::builder_draw_type_t::text_renderer_clickable;
      pile->editor_draw_types.builder_draw_type_index = pile->builder_draw_types.trc.size(&pile->window, &pile->context) - 1;
      pile->editor_draw_types.selected_draw_type = editor_draw_types_t::builder_draw_type_t::text_renderer_clickable;
      pile->editor_draw_types.selected_draw_type_index = pile->editor_draw_types.builder_draw_type_index;

      depth_map_t map;
      map.builder_draw_type = pile->editor_draw_types.builder_draw_type;
      map.builder_draw_type_index = pile->editor_draw_types.builder_draw_type_index;
      pile->editor_draw_types.depth_map.push_back(map);

      break;
    }
    case fan::key_state::release: {
      pile->editor_draw_types.flags &= ~flags_t::moving;

      switch (pile->editor_draw_types.builder_draw_type) {
        case builder_draw_type_t::text_renderer_clickable: {

          // if object is not within builder_viewport we will delete it
          if (!pile->editor_draw_types.is_inside_builder_viewport(
            pile,
            pile->builder_draw_types.trc.get_hitbox_position(
            &pile->window,
            &pile->context,
            pile->editor_draw_types.builder_draw_type_index
            ) +
            pile->builder_draw_types.trc.get_hitbox_size(
            &pile->window,
            &pile->context,
            pile->editor_draw_types.builder_draw_type_index
            )
            ))
          {
            pile->editor_draw_types.close_build_properties(pile);
            pile->builder_draw_types.trc.erase(
              &pile->window,
              &pile->context,
              pile->editor_draw_types.builder_draw_type_index
            );
            pile->editor_draw_types.depth_map.erase(pile->editor_draw_types.depth_map.size() - 1);
          }
          else {
            pile->editor_draw_types.close_build_properties(pile);
            click_collision_t click_collision;
            click_collision.builder_draw_type = pile->editor_draw_types.builder_draw_type;
            click_collision.builder_draw_type_index = pile->editor_draw_types.builder_draw_type_index;
            pile->editor_draw_types.open_build_properties(pile, click_collision);
          }
          break;
        }
      }
      break;
    }
  }
  break;
}