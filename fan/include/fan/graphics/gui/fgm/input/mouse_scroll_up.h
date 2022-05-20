 case fan::mouse_scroll_up: {

   if (!pile->editor_draw_types.is_inside_properties_viewport(pile, pile->window.get_mouse_position())) {
     break;
   }

   pile->editor_draw_types.properties_camera.y -= constants::scroll_speed;
   pile->editor_draw_types.properties_camera.y = fan::clamp(pile->editor_draw_types.properties_camera.y, 0.f, pile->editor_draw_types.properties_camera.y);
   pile->editor_draw_types.gui_properties_matrices.set_camera_position(&pile->context, pile->editor_draw_types.properties_camera);
   pile->editor_draw_types.properties_button.set_viewport_collision_offset(
     pile->editor_draw_types.origin_properties - pile->editor_draw_types.properties_camera
   );

   pile->editor_draw_types.properties_button.m_button_event.lose_focus(&pile->window);

   break;
 }