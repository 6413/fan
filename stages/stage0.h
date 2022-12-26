stage_common_t stage_common = {
	.open = [this] () {
		
	},
	.close = [this] {
		
	},
	.window_resize_callback = [this] {
		
	},
	.update = [this] {
		
	}
};

static void lib_open(loco_t* loco, stage_common_t* sc, const stage_common_t::open_properties_t& op) {

	sc->instances.Open();

	fan::string fgm_name = fan::file_name(__FILE__);
	fgm_name.pop_back(); // remove
	fgm_name.pop_back(); // .h
	fan::string full_path = fan::string("stages/") + fgm_name + ".fgm";
	fan::string f;
	if (!fan::io::file::exists(full_path)) {
		return;
	}
	fan::io::file::read(full_path, &f);
	uint64_t off = 0;
	uint32_t instance_count = fan::io::file::read_data<uint32_t>(f, off);
	for (uint32_t i = 0; i < instance_count; i++) {
		auto p = fan::io::file::read_data<fan::vec3>(f, off);
		auto s = fan::io::file::read_data<fan::vec2>(f, off);
		auto fs = fan::io::file::read_data<f32_t>(f, off);
		auto text = fan::io::file::read_data<fan::wstring>(f, off);
		fan::io::file::read_data<fan_2d::graphics::gui::theme_t>(f, off);
		typename loco_t::button_t::properties_t bp;
		bp.position = p;
		bp.size = s;
		bp.font_size = fs;
		bp.text = text;
		bp.theme = op.theme;
		bp.matrices = op.matrices;
		bp.viewport = op.viewport;
		bp.mouse_button_cb = mouse_button_cb0;
		auto nr = sc->instances.NewNodeLast();

		loco->button.push_back(&sc->instances[nr].cid, bp);
	}
}

static void lib_close(stage_common_t* sc) {
	sc->instances.Close();
}