static constexpr uint32_t group_id_indicator = 1;
static constexpr uint32_t group_id_indicator_rp = 2;
static constexpr uint32_t group_id_text_E = 3;
static constexpr uint32_t group_id_text_F = 4;

engine::ml_t::cm_t HudFuelModel{"hud_fuel.fmm"};
engine::ml_t::id_t HudFuelID;

struct hud_element_t {
	loco_t::id_t* text;
	// other shapes can be placed here like sprites (id_t)
};

hud_element_t hud_health;
//hud_element_t hud_fuel;

struct hud_fuel_t {
	//loco_t::id_t text_e;
	//loco_t::id_t text_f;
}hud_fuel;


void set_health(f32_t health) {
	// represents health by *.x e.g 10.1
	//hud_health.text->set_text(fan::to_string(health, 1));
}

void set_fuel(f32_t fuel, f32_t max_fuel) {
	static constexpr f32_t min_angle = 0.29;
	static constexpr f32_t max_angle = min_angle + fan::math::radians(240);
	f32_t scaled_value = ((fuel / max_fuel) - 0) / (1 - 0);
	HudFuelID.set_angle(group_id_indicator, -(scaled_value * (max_angle - min_angle) + min_angle));
}

void open(auto& loco) {
	hud_health.text = &game::pile->StageList._StageLoader.get_id(this, "text_health");
	//hud_fuel.text = &game::pile->StageList._StageLoader.get_id(this, "text_fuel");

	engine::ml_t::properties_t properties;
	properties.viewport = &game::pile->viewport;
	properties.camera = &game::pile->gui_camera;
	properties.position = hud_health.text->get_position();
	properties.position.y *= 1.5;
	properties.position.z += 1;
	HudFuelID.add(&HudFuelModel, properties);


	fan::vec2 offset = 0;
	// lazy
	HudFuelID.iterate(group_id_indicator, [&]<typename T>(auto shape_id, const T& data) {
		offset = data.position;
	});

	HudFuelID.iterate_marks(group_id_text_E, [this]<typename T>(auto shape_id, const T& properties) {
		HudFuelID.add_shape(group_id_text_E, fan_init_struct(
			loco_t::text_t::properties_t,
			.viewport = &game::pile->viewport,
			.camera = &game::pile->gui_camera,
			.position = properties.position,
			.color = fan::colors::red,
			.text = "E",
			.font_size = 0.2
			),
			properties
		);
	});
	HudFuelID.iterate_marks(group_id_text_F, [this, &offset, position = properties.position]<typename T>(auto shape_id, const T& properties) {
		HudFuelID.add_shape(group_id_text_F, fan_init_struct(
			loco_t::text_t::properties_t,
			.viewport = &game::pile->viewport,
			.camera = &game::pile->gui_camera,
			.position = properties.position,
			.color = fan::colors::white,
			.text = "F",
			.font_size = 0.2
			),
			properties
		);
	});

	HudFuelID.set_size(0.3);
	// set rotation point after set_size
	HudFuelID.iterate_marks(group_id_indicator_rp, [this, &offset, position = properties.position]<typename T>(auto shape_id, const T& properties) {
		HudFuelID.set_rotation_point(group_id_indicator, offset + fan::vec2(properties.position));
	});


	hud_health.text->erase();

	//HudFuelModel.
}

void close(auto& loco){
		
}

void window_resize_callback(auto& loco){
		
}

void update(auto& loco){

}