pile->builder.rtbs.properties_button.m_button_event.set_on_input(pile, [](
  fan::window_t* window, fan::opengl::context_t* context, uint32_t i, uint16_t key, fan::key_state key_state, mouse_stage mouse_stage, void* userptr) {
  pile_t* pile = (pile_t*)userptr;

  if (key != fan::mouse_left) {
    return;
  }

  if (key_state != fan::key_state::release) {
    return;
  } 
  if (mouse_stage != fan_2d::graphics::gui::mouse_stage::inside) {
    return;
  }
  
}