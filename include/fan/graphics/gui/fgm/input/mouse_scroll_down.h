case fan::mouse_scroll_down: {

  if (!pile->editor.is_inside_properties_viewport(pile, pile->window.get_mouse_position())) {
    break;
  }

  pile->editor.properties_camera.y += constants::scroll_speed;
  pile->editor.gui_properties_matrices.set_camera_position(&pile->context, pile->editor.properties_camera);
  pile->editor.properties_button.set_viewport_collision_offset(
    pile->editor.origin_properties - pile->editor.properties_camera
  );
  pile->editor.properties_button.m_button_event.lose_focus(&pile->window);
  break;
}