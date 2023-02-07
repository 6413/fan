void open(auto& loco) {
  
}

void close(auto& loco){
		
}

void window_resize_callback(auto& loco){
		
}

void update(auto& loco){
	
}

int button0_mouse_button_cb(const loco_t::mouse_button_data_t& mb){
  pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco);
  pile->stage_loader.erase_stage(&pile->loco, stage_id);
  pile_t::stage_loader_t::stage_open_properties_t op;
  op.matrices = &pile->matrices;
  op.viewport = &pile->viewport;
  op.theme = &pile->theme;
  pile->stage_loader.push_and_open_stage<pile_t::stage_loader_t::stage::stage1_t>(&pile->loco, op);
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

int button1_mouse_button_cb(const loco_t::mouse_button_data_t& mb){
  return 0;
}

int button1_mouse_move_cb(const loco_t::mouse_move_data_t& mb){
  return 0;
}

int button1_keyboard_cb(const loco_t::keyboard_data_t& mb){
  return 0;
}

int button1_text_cb(const loco_t::text_data_t& mb){
  return 0;
}
