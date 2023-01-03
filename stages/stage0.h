
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

  while (off < f.size()) {
    format::shape_type_t::_t shape_type = fan::io::file::read_data<format::shape_type_t::_t>(f, off);
    uint32_t instance_count = fan::io::file::read_data<uint32_t>(f, off);

    for (uint32_t i = 0; i < instance_count; ++i) {
      switch (shape_type) {
      case format::shape_type_t::button: {
        auto data = fan::io::file::read_data<format::shape_button_t>(f, off);
        auto text = fan::io::file::read_data<fan::wstring>(f, off);
        loco_t::button_t::properties_t bp;
        bp.position = data.position;
        bp.size = data.size;
        bp.font_size = data.font_size;
        bp.text = text;
        bp.theme = &data.theme;
        bp.matrices = op.matrices;
        bp.viewport = op.viewport;
        loco->button.push_back(&cid_table[shape_type][i], bp);
        break;
      }
      case format::shape_type_t::sprite: {
        auto data = fan::io::file::read_data<format::shape_sprite_t>(f, off);
        loco_t::sprite_t::properties_t sp;
        sp.position = data.position;
        sp.size = data.size;
        loco_t::texturepack::ti_t ti;
        if (loco->stage_loader.texturepack.qti(data.hash_path, &ti)) {
          fan::throw_error("failed to load texture from texturepack");
        }
        auto pd = loco->stage_loader.texturepack.get_pixel_data(ti.pack_id);
        sp.image = &pd.image;
        sp.tc_position = ti.position / pd.size;
        sp.tc_size = ti.size / pd.size;
        sp.matrices = op.matrices;
        sp.viewport = op.viewport;
        loco->sprite.push_back(&cid_table[shape_type][i], sp);
        break;
      }
      default: {
        fan::throw_error("i cant find what you talk about - fgm");
        break;
      }
      }
    }
	}
}

static void lib_close(stage_common_t* sc) {

}