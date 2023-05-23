void open(void* sod) {
  auto& id = gstage->get_id(this, "hitbox_6");
  gloco->vfi.shape_list[*(loco_t::vfi_t::shape_id_t*)&id].shape_data.mouse_button_cb = [&](const auto& d) -> int {
    if (d.button != fan::mouse_left) {
      return 0;
    }
    if (d.button_state != fan::mouse_state::press) {
      return 0;
    }

    fan::print("click");
  };
}

void close() {
		
}

void window_resize_callback(){
		
}

void update(){
	
}
