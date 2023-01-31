fan::graphics::cid_t* button1_cid;

void open(auto& loco) {
  auto* pile = OFFSETLESS(&loco, pile_t, loco);
  button1_cid = pile->stage_loader.get_cid(this, "button1");
  fan::print(button1_cid);
  fan::print(loco.button.get_text(button1_cid));
}

void close(auto& loco){
		
}

void window_resize_callback(auto& loco){

}

void update(auto& loco){

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

int button2_mouse_button_cb(const loco_t::mouse_button_data_t& mb){
  return 0;
}

int button2_mouse_move_cb(const loco_t::mouse_move_data_t& mb){
  return 0;
}

int button2_keyboard_cb(const loco_t::keyboard_data_t& mb){
  return 0;
}

int button2_text_cb(const loco_t::text_data_t& mb){
  return 0;
}

int button3_mouse_button_cb(const loco_t::mouse_button_data_t& mb){
  return 0;
}

int button3_mouse_move_cb(const loco_t::mouse_move_data_t& mb){
  return 0;
}

int button3_keyboard_cb(const loco_t::keyboard_data_t& mb){
  return 0;
}

int button3_text_cb(const loco_t::text_data_t& mb){
  return 0;
}

int button4_mouse_button_cb(const loco_t::mouse_button_data_t& mb){
  return 0;
}

int button4_mouse_move_cb(const loco_t::mouse_move_data_t& mb){
  return 0;
}

int button4_keyboard_cb(const loco_t::keyboard_data_t& mb){
  return 0;
}

int button4_text_cb(const loco_t::text_data_t& mb){
  return 0;
}

int button5_mouse_button_cb(const loco_t::mouse_button_data_t& mb){
  return 0;
}

int button5_mouse_move_cb(const loco_t::mouse_move_data_t& mb){
  return 0;
}

int button5_keyboard_cb(const loco_t::keyboard_data_t& mb){
  return 0;
}

int button5_text_cb(const loco_t::text_data_t& mb){
  return 0;
}

int button6_mouse_button_cb(const loco_t::mouse_button_data_t& mb){
  return 0;
}

int button6_mouse_move_cb(const loco_t::mouse_move_data_t& mb){
  return 0;
}

int button6_keyboard_cb(const loco_t::keyboard_data_t& mb){
  return 0;
}

int button6_text_cb(const loco_t::text_data_t& mb){
  return 0;
}

int button7_mouse_button_cb(const loco_t::mouse_button_data_t& mb){
  return 0;
}

int button7_mouse_move_cb(const loco_t::mouse_move_data_t& mb){
  return 0;
}

int button7_keyboard_cb(const loco_t::keyboard_data_t& mb){
  return 0;
}

int button7_text_cb(const loco_t::text_data_t& mb){
  return 0;
}
