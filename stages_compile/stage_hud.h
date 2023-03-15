struct hud_element_t {
	loco_t::id_t* text;
	// other shapes can be placed here like sprites (id_t)
};

hud_element_t hud_health;
hud_element_t hud_fuel;

void set_health() {

}

void open(auto& loco) {
	hud_health.text = &game::pile->StageList._StageLoader.get_id(this, "text_health");
	hud_fuel.text = &game::pile->StageList._StageLoader.get_id(this, "text_health");
}

void close(auto& loco){
		
}

void window_resize_callback(auto& loco){
		
}

void update(auto& loco){
	
}