while (off < f.size()) {
  loco_t::shape_type_t::_t shape_type = fan::io::file::read_data<loco_t::shape_type_t::_t>(f, off);
  uint32_t instance_count = fan::io::file::read_data<uint32_t>(f, off);
  for (uint32_t i = 0; i < instance_count; ++i) {
    switch (shape_type) {
      case loco_t::shape_type_t::button: {
        auto data = fan::io::file::read_data<stage_maker_shape_format::shape_button_t>(f, off);
        auto text = fan::io::file::read_data<fan::string>(f, off);
        button_t::properties_t bp;
        bp.position = data.position;
        bp.size = data.size;
        bp.font_size = data.font_size;
        bp.text = text;
        bp.theme = &theme;
        bp.matrices = &matrices[viewport_area::editor];
        bp.viewport = &viewport[viewport_area::editor];
        bp.id = data.id;
        button.push_back(bp);
        break;
      }
      case loco_t::shape_type_t::sprite: {
        auto data = fan::io::file::read_data<stage_maker_shape_format::shape_sprite_t>(f, off);
        auto t = fan::io::file::read_data<fan::string>(f, off);
        sprite_t::properties_t sp;
        sp.position = data.position;
        sp.size = data.size;
        sp.parallax_factor = data.parallax_factor;
        loco_t::texturepack_t::ti_t ti;
        if (texturepack.qti(t, &ti)) {
          sp.image = &get_loco()->default_texture;
        }
        else {
          auto& pd = texturepack.get_pixel_data(ti.pack_id);
          sp.image = &pd.image;
          sp.tc_position = ti.position / pd.image.size;
          sp.tc_size = ti.size / pd.image.size;
        }
        sp.matrices = &matrices[viewport_area::editor];
        sp.viewport = &viewport[viewport_area::editor];
        sp.texturepack_name = t;
        sprite.push_back(sp);
        break;
      }
      case loco_t::shape_type_t::text: {
        auto data = fan::io::file::read_data<stage_maker_shape_format::shape_text_t>(f, off);
        auto t = fan::io::file::read_data<fan::string>(f, off);
        text_t::properties_t p;
        p.position = data.position;
        p.font_size = data.size;
        p.matrices = &matrices[viewport_area::editor];
        p.viewport = &viewport[viewport_area::editor];
        p.text = t;
        text.push_back(p);
        break;
      }
      case loco_t::shape_type_t::hitbox: {
        auto data = fan::io::file::read_data<stage_maker_shape_format::shape_hitbox_t>(f, off);
        hitbox_t::properties_t sp;
        sp.position = data.position;
        sp.size = data.size;
        sp.image = &hitbox_image;
        sp.matrices = &matrices[viewport_area::editor];
        sp.viewport = &viewport[viewport_area::editor];
        sp.shape_type = data.shape_type;
        sp.id = data.id;
        hitbox.push_back(sp);
        break;
      }
      default: {
        fan::throw_error("i cant find what you talk about - fgm");
        break;
      }
    }
  }     
}