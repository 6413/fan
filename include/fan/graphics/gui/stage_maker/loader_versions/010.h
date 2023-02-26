while (off < f.size()) {
  auto shape_type = fan::io::file::read_data<loco_t::shape_type_t::_t>(f, off);
  uint32_t instance_count = fan::io::file::read_data<uint32_t>(f, off);

  for (uint32_t i = 0; i < instance_count; ++i) {
    auto nr = stage->cid_list.NewNodeLast();
    switch (shape_type) {
      case loco_t::shape_type_t::button: {
        auto data = fan::io::file::read_data<stage_maker_shape_format::shape_button_t>(f, off);
        auto text = fan::io::file::read_data<fan::string>(f, off);
        loco_t::button_t::properties_t bp;
        bp.position = data.position;
        bp.position.z += stage->it * op.itToDepthMultiplier;
        bp.size = data.size;
        bp.font_size = data.font_size;
        bp.text = text;
        bp.theme = op.theme;
        bp.matrices = op.matrices;
        bp.viewport = op.viewport;
        bp.mouse_button_cb = [stage, i](const loco_t::mouse_button_data_t& d) {
          return (stage->*(stage->button_mouse_button_cb_table[i]))(d);
        };
        loco->button.push_back(&stage->cid_list[nr].cid, bp);
        cid_map[std::make_pair(stage, "button" + data.id)] = &stage->cid_list[nr].cid;
        break;
      }
      case loco_t::shape_type_t::sprite: {
        auto data = fan::io::file::read_data<stage_maker_shape_format::shape_sprite_t>(f, off);
        auto t = fan::io::file::read_data<fan::string>(f, off);
        loco_t::sprite_t::properties_t sp;
        sp.position = data.position;
        sp.position.z += stage->it * op.itToDepthMultiplier;
        sp.size = data.size;
        sp.parallax_factor = data.parallax_factor;
        loco_t::texturepack_t::ti_t ti;
        if (texturepack->qti(t, &ti)) {
          sp.image = &loco->default_texture;
        }
        else {
          auto& pd = texturepack->get_pixel_data(ti.pack_id);
          sp.image = &pd.image;
          sp.tc_position = ti.position / pd.image.size;
          sp.tc_size = ti.size / pd.image.size;
        }
        sp.matrices = op.matrices;
        sp.viewport = op.viewport;
        loco->sprite.push_back(&stage->cid_list[nr].cid, sp);
        break;
      }
      case loco_t::shape_type_t::text: {
        auto data = fan::io::file::read_data<stage_maker_shape_format::shape_text_t>(f, off);
        auto t = fan::io::file::read_data<fan::string>(f, off);
        loco_t::text_t::properties_t p;
        p.matrices = op.matrices;
        p.viewport = op.viewport;

        p.position = data.position;
        p.position.z += stage->it * op.itToDepthMultiplier;
        p.font_size = data.size;
        p.text = t;
        loco->text.push_back(&stage->cid_list[nr].cid, p);
        break;
      }
      case loco_t::shape_type_t::hitbox: {
        auto data = fan::io::file::read_data<stage_maker_shape_format::shape_hitbox_t>(f, off);
        loco_t::vfi_t::properties_t vfip;
        switch (data.shape_type) {
          case loco_t::vfi_t::shape_t::always: {
            vfip.shape_type = loco_t::vfi_t::shape_t::always;
            vfip.shape.always.z = data.position.z;
            break;
          }
          case loco_t::vfi_t::shape_t::rectangle: {
            vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
            vfip.shape.rectangle.position = data.position;
            vfip.shape.rectangle.size = data.size;
            vfip.shape.rectangle.matrices = op.matrices;
            vfip.shape.rectangle.viewport = op.viewport;
            break;
          }
        }
        vfip.mouse_button_cb = [stage, i](const loco_t::mouse_button_data_t& d) {
          return (stage->*(stage->hitbox_mouse_button_cb_table[i]))(d);
        };
        vfip.mouse_move_cb = [stage, i](const loco_t::mouse_move_data_t& d) {
          return (stage->*(stage->hitbox_mouse_move_cb_table[i]))(d);
        };
        vfip.keyboard_cb = [stage, i](const loco_t::keyboard_data_t& d) {
          return (stage->*(stage->hitbox_keyboard_cb_table[i]))(d);
        };
        vfip.text_cb = [stage, i](const loco_t::text_data_t& d) {
          return (stage->*(stage->hitbox_text_cb_table[i]))(d);
        };
        vfip.ignore_init_move = true;

        loco->push_back_input_hitbox((loco_t::vfi_t::shape_id_t*)&stage->cid_list[nr].cid, vfip);

        cid_map[std::make_pair(stage, "hitbox" + data.id)] = &stage->cid_list[nr].cid;

        break;
      }
      default: {
        fan::throw_error("i cant find what you talk about - fgm");
        break;
      }
    }
    stage->cid_list[nr].type = shape_type;
  }
}