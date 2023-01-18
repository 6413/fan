void open(auto* loco) {
  
}

void close(auto* loco){
		
}

void window_resize_callback(auto* loco){
		
}

void update(auto* loco){
	
}

int button0_click_cb(const loco_t::mouse_button_data_t& mb){
  if (mb.button != fan::mouse_left) {
    return 0;
  }

  if (mb.button_state != fan::mouse_state::release) {
    return 0;
  }

  pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco);

  using sl = pile_t::stage_loader_t;

  sl::stage_open_properties_t op;
  op.matrices = &pile->matrices;
  op.viewport = &pile->viewport;
  op.theme = &pile->theme;
  pile->nrs[1] = pile->stage_loader.push_and_open_stage<sl::stage::stage1_t>(&pile->loco, op);
  pile->stage_loader.erase_stage<sl::stage::stage0_t>(&pile->loco, pile->nrs[0]);

  return 1;
}
