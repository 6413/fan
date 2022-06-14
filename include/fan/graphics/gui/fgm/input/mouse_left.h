 case fan::mouse_left: {
   switch (key_state) {
     case fan::key_state::press: {

       if (pile->editor.flags & flags_t::resizing) {
         break;
       }

       click_collision_t click_collision;
       if (pile->editor.click_collision(pile, &click_collision)) {
         pile->editor.close_build_properties(pile);
         pile->editor.open_build_properties(pile, click_collision);

         pile->editor.flags |= flags_t::moving;
         switch (click_collision.builder_draw_type) {
           case builder_draw_type_t::sprite: {

             pile->editor.click_position = pile->builder.sprite.get_position(&pile->context, click_collision.builder_draw_type_index);
             break;
           }
           case builder_draw_type_t::text_renderer: {

             pile->editor.click_position = pile->builder.tr.get_position(&pile->context, click_collision.builder_draw_type_index);
             break;
           }
           case builder_draw_type_t::hitbox: {

             pile->editor.click_position = pile->builder.hitbox.get_position(&pile->context, click_collision.builder_draw_type_index);
             break;
           }
           case builder_draw_type_t::button: {

             pile->editor.click_position = pile->builder.button.get_position(&pile->window, &pile->context, click_collision.builder_draw_type_index);
             break;
           }
           default: {
             fan::throw_error("click position not set for current shape");
           }
         }
         pile->editor.move_offset = pile->editor.click_position - pile->editor.get_mouse_position(pile);
         pile->editor.click_position -= pile->editor.get_mouse_position(pile);

         return;
       }
       for (int i = 0; i < pile->editor.properties_button.size(&pile->window, &pile->context); i++) {
         if (pile->editor.properties_button.inside(&pile->window, &pile->context, i, pile->window.get_mouse_position())) {
           return;
         }
       }
       if (!pile->editor.is_inside_builder_viewport(pile, pile->editor.get_mouse_position(pile))) {
         return;
       }
       pile->editor.close_build_properties(pile);

       break;
     }
     case fan::key_state::release: {

      for (uint32_t i = 0; i < pile->editor.builder_types.size(&pile->window, &pile->context); i++) {
        if (pile->editor.builder_types.inside(&pile->window, &pile->context, i, pile->window.get_mouse_position()) &&
            pile->editor.builder_types.get_text(&pile->window, &pile->context, i) == L"export"
          ) {
          pile->save("123");
          return;
        }
      }

       pile->editor.move_offset = 0;

       pile->editor.flags &= ~flags_t::moving;

       if (!pile->editor.is_inside_builder_viewport(pile, pile->editor.get_mouse_position(pile))) {
         return;
       }

       if (pile->editor.properties_button.size(&pile->window, &pile->context) == 0) {
         return;
       }

       click_collision_t click_collision;
       click_collision.builder_draw_type = pile->editor.selected_type;
       click_collision.builder_draw_type_index = pile->editor.selected_type_index;

       // update on release
       pile->editor.close_build_properties(pile);
       pile->editor.open_build_properties(pile, click_collision);
       break;
     }
   }
   break;
 }