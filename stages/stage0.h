void open(auto* loco) {
  
}

void close(auto* loco){
		
}

void window_resize_callback(auto* loco){
		
}

void update(auto* loco){
	
}

int button0_click_cb(const loco_t::mouse_button_data_t& mb){
  fan::print("button state", (int)mb.button_state);
  return 0;
}

int hitbox0_mouse_button_cb(const loco_t::mouse_button_data_t& mb){
  if (mb.button != fan::mouse_left) {
    return 0;
  }
  if (mb.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
    return 0;
  }
  pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco);

  if (mb.button_state == fan::mouse_state::press) {
    fan::print("fake highlight - too low budget");
  }
  else {
    fan::print("fake highlight off");
  }
  return 0;
}

int hitbox0_mouse_move_cb(const loco_t::mouse_move_data_t& mb){
  return 0;
}

int hitbox0_keyboard_cb(const loco_t::keyboard_data_t& mb){
  return 0;
}

int hitbox0_text_cb(const loco_t::text_data_t& mb){
  return 0;
}
