void open(auto* loco) {
  
}

void close(auto* loco){
		
}

void window_resize_callback(auto* loco){
		
}

void update(auto* loco){
	
}

int button0_click_cb(const loco_t::mouse_button_data_t& mb){
  game::pile->StageList.Remove(this->parent_id);
  game::pile->StageList.Remove(this->stage_id);
  game::pile->StageList.Add<game::pile_t::StageList_t::stage_loader_t::stage::stage2_t>();
  return 1;
}

int button1_click_cb(const loco_t::mouse_button_data_t& mb){
  return 0;
}
