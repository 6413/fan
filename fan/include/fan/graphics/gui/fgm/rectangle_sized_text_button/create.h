case builder_draw_type_t::rectangle_text_button_sized: {
  switch (key_state) {
    case fan::key_state::press: {
      pile->editor_draw_types.flags |= flags_t::moving;
      pile->editor_draw_types.moving_position = pile->window.get_mouse_position();

      decltype(pile->builder_draw_types.rtbs)::properties_t rtbs_p;
      rtbs_p.position = pile->window.get_mouse_position();
      // fetch same size as gui
      rtbs_p.size = pile->editor_draw_types.builder_types.get_size(&pile->window, &pile->context, 0);
      rtbs_p.text = "Button";
      rtbs_p.font_size = constants::gui_size;
      rtbs_p.theme = fan_2d::graphics::gui::themes::gray();
      rtbs_p.userptr = (void*)pile->builder_draw_types.rtbs_id_counter++;
      pile->builder_draw_types.rtbs.push_back(&pile->window, &pile->context, rtbs_p);

      pile->editor_draw_types.builder_draw_type = editor_draw_types_t::builder_draw_type_t::rectangle_text_button_sized;
      pile->editor_draw_types.builder_draw_type_index = pile->builder_draw_types.rtbs.size(&pile->window, &pile->context) - 1;
      pile->editor_draw_types.selected_draw_type = editor_draw_types_t::builder_draw_type_t::rectangle_text_button_sized;
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
        case builder_draw_type_t::rectangle_text_button_sized: {

          // if object is not within builder_viewport we will delete it
          if (!pile->editor_draw_types.is_inside_builder_viewport(
            pile,
            pile->builder_draw_types.rtbs.get_position(
            &pile->window,
            &pile->context,
            pile->editor_draw_types.builder_draw_type_index
            ) +
            pile->builder_draw_types.rtbs.get_size(
            &pile->window,
            &pile->context,
            pile->editor_draw_types.builder_draw_type_index
            )
            ))
          {
            pile->editor_draw_types.close_build_properties(pile);
            pile->builder_draw_types.rtbs.erase(
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