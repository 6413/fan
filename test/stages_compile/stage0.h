void open(auto* loco) {
  
}

void close(auto* loco){
		
}

void window_resize_callback(auto* loco){
		
}

void update(auto* loco){
	
}

int button0_click_cb(const loco_t::mouse_button_data_t& mb){
  return 0;
}

loco_t::vfi_id_t vfiBaseID = loco_t::vfi_id_t(fan_init_struct(loco_t::vfi_id_t::properties_t, .shape_type = loco_t::vfi_t::shape_t::always, .shape.always.z = 0));