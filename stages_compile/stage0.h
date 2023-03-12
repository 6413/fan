void open(auto& loco) {
  pile_t* l_pile = OFFSETLESS(&loco, pile_t, loco);
  l_pile->stage_loader.get_id(this, "button_health")->set_text("some button");
}

void close(auto& loco){
		
}

void window_resize_callback(auto& loco){
		
}

void update(auto& loco){
	
}

int button0_mouse_button_cb(const loco_t::mouse_button_data_t& mb){
  return 0;
}

int button0_mouse_move_cb(const loco_t::mouse_move_data_t& mb){
  return 0;
}

int button0_keyboard_cb(const loco_t::keyboard_data_t& mb){
  return 0;
}

int button0_text_cb(const loco_t::text_data_t& mb){
  return 0;
}

int buttonhealth_mouse_button_cb(const loco_t::mouse_button_data_t& mb){
  return 0;
}

int buttonhealth_mouse_move_cb(const loco_t::mouse_move_data_t& mb){
  return 0;
}

int buttonhealth_keyboard_cb(const loco_t::keyboard_data_t& mb){
  return 0;
}

int buttonhealth_text_cb(const loco_t::text_data_t& mb){
  return 0;
}
