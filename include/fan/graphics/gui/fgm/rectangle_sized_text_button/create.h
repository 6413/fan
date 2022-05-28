case builder_draw_type_t::rectangle_text_button_sized: {
  switch (key_state) {
    case fan::key_state::press: {
      pile->editor.flags |= flags_t::moving;
      pile->editor.click_position = pile->window.get_mouse_position();

      decltype(pile->builder.rtbs)::properties_t rtbs_p;
      rtbs_p.position = pile->window.get_mouse_position();
      // fetch same size as gui
      rtbs_p.size = pile->editor.builder_types.get_size(&pile->window, &pile->context, 0);
      rtbs_p.text = "Button";
      rtbs_p.font_size = constants::gui_size;
      rtbs_p.theme = fan_2d::graphics::gui::themes::gray();
      rtbs_p.userptr = (void*)pile->builder.rtbs_id_counter++;
      depth_t depth;
      depth.depth = pile->editor.depth_index++;
      depth.type = builder_draw_type_t::rectangle_text_button_sized;
      depth.index = pile->builder.rtbs.size(&pile->window, &pile->context);
      pile->editor.depth_map.push_back(depth);
      pile->builder.rtbs.push_back(&pile->window, &pile->context, rtbs_p);
      pile->editor.builder_draw_type = editor_t::builder_draw_type_t::rectangle_text_button_sized;
      pile->editor.builder_draw_type_index = pile->builder.rtbs.size(&pile->window, &pile->context) - 1;
      pile->editor.selected_type = editor_t::builder_draw_type_t::rectangle_text_button_sized;
      pile->editor.selected_type_index = pile->editor.builder_draw_type_index;

      break;
    }
    case fan::key_state::release: {
      pile->editor.flags &= ~flags_t::moving;

      switch (pile->editor.builder_draw_type) {
        case builder_draw_type_t::rectangle_text_button_sized: {

          // if object is not within builder_viewport we will delete it
          if (!pile->editor.is_inside_builder_viewport(
            pile, 
            pile->builder.rtbs.get_position(
            &pile->window,
            &pile->context,
            pile->editor.builder_draw_type_index
            ) +
            pile->builder.rtbs.get_size(
            &pile->window,
            &pile->context,
            pile->editor.builder_draw_type_index
            )
            ))
          {
            pile->editor.close_build_properties(pile);
            pile->builder.rtbs.erase(
              &pile->window,
              &pile->context,
              pile->editor.builder_draw_type_index
            );
            pile->editor.depth_map.erase(pile->editor.depth_map.size() - 1);
          }
          else {
            pile->editor.close_build_properties(pile);
            click_collision_t click_collision;
            click_collision.builder_draw_type = pile->editor.builder_draw_type;
            click_collision.builder_draw_type_index = pile->editor.builder_draw_type_index;
            pile->editor.open_build_properties(pile, click_collision);
          }
          break;
        }
      }
      break;
    }
  }
  break;
}