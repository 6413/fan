case builder_draw_type_t::text_renderer: {
  switch (key_state) {
    case fan::key_state::press: {
      pile->editor.flags |= flags_t::moving;
      pile->editor.click_position = pile->window.get_mouse_position();

      fan_2d::graphics::gui::text_renderer_t::properties_t tr_p;
      tr_p.position = pile->window.get_mouse_position();
      tr_p.text = "Clickable text";
      tr_p.font_size = constants::gui_size;
      tr_p.text_color = fan::colors::white;
     
      pile->builder.tr.push_back(&pile->context, tr_p);

      pile->editor.depth_map_push(pile, builder_draw_type_t::text_renderer, pile->builder.tr.size(&pile->context) - 1);

      pile->editor.builder_draw_type = editor_t::builder_draw_type_t::text_renderer;
      pile->editor.builder_draw_type_index = pile->builder.tr.size(&pile->context) - 1;
      pile->editor.selected_type = editor_t::builder_draw_type_t::text_renderer;
      pile->editor.selected_type_index = pile->editor.builder_draw_type_index;

      break;
    }
    case fan::key_state::release: {
      pile->editor.flags &= ~flags_t::moving;

      switch (pile->editor.builder_draw_type) {
        case builder_draw_type_t::text_renderer: {

          //// if object is not within builder_viewport we will delete it
          //if (!pile->editor.is_inside_builder_viewport(
          //  pile,
          //  pile->builder.tr.get_hitbox_position(
          //  &pile->window,
          //  &pile->context,
          //  pile->editor.builder_draw_type_index
          //  ) +
          //  pile->builder.tr.get_hitbox_size(
          //  &pile->window,
          //  &pile->context,
          //  pile->editor.builder_draw_type_index
          //  )
          //  ))
          //{
          //  pile->editor.close_build_properties(pile);
          //  pile->builder.tr.erase(
          //    &pile->context,
          //    pile->editor.builder_draw_type_index
          //  );
          //  pile->editor.depth_map.erase(pile->editor.depth_map.size() - 1);
          //}
          //else {
            pile->editor.close_build_properties(pile);
            click_collision_t click_collision;
            click_collision.builder_draw_type = pile->editor.builder_draw_type;
            click_collision.builder_draw_type_index = pile->editor.builder_draw_type_index;
            pile->editor.open_build_properties(pile, click_collision);
          //}
          break;
        }
      }
      break;
    }
  }
  break;
}