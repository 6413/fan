 case fan::mouse_left: {
   switch (key_state) {
     case fan::key_state::press: {
       click_collision_t click_collision;
       if (pile->editor_draw_types.click_collision(pile, &click_collision)) {
         pile->editor_draw_types.close_build_properties(pile);
         pile->editor_draw_types.open_build_properties(pile, click_collision);

         pile->editor_draw_types.flags |= flags_t::moving;
         switch (click_collision.builder_draw_type) {
           case builder_draw_type_t::rectangle_text_button_sized: {

             pile->editor_draw_types.moving_position = pile->builder_draw_types.rtbs.get_position(&pile->window, &pile->context, click_collision.builder_draw_type_index);
             break;
           }
           case builder_draw_type_t::text_renderer_clickable: {

             pile->editor_draw_types.moving_position = pile->builder_draw_types.trc.get_position(&pile->window, &pile->context, click_collision.builder_draw_type_index);
             break;
           }
         }
         pile->editor_draw_types.moving_position -= pile->window.get_mouse_position();

         return;
       }
       for (int i = 0; i < pile->editor_draw_types.properties_button.size(&pile->window, &pile->context); i++) {
         if (pile->editor_draw_types.properties_button.inside(&pile->window, &pile->context, i, pile->window.get_mouse_position())) {
           return;
         }
       }
       if (!pile->editor_draw_types.is_inside_builder_viewport(pile, pile->window.get_mouse_position())) {
         return;
       }
       pile->editor_draw_types.close_build_properties(pile);

       break;
     }
     case fan::key_state::release: {
       pile->editor_draw_types.flags &= ~flags_t::moving;

       if (!pile->editor_draw_types.is_inside_builder_viewport(pile, pile->window.get_mouse_position())) {
         return;
       }

       if (pile->editor_draw_types.properties_button.size(&pile->window, &pile->context) == 0) {
         return;
       }

       click_collision_t click_collision;
       click_collision.builder_draw_type = pile->editor_draw_types.selected_draw_type;
       click_collision.builder_draw_type_index = pile->editor_draw_types.selected_draw_type_index;

       // update on release
       pile->editor_draw_types.close_build_properties(pile);
       pile->editor_draw_types.open_build_properties(pile, click_collision);
       break;
     }
   }
   break;
 }