while (off < f.size()) {
  auto shape_type = read_data<loco_t::shape_type_t::_t>(f, off);
  switch (shape_type) {
    case loco_t::shape_type_t::button: {
      stage_maker_shape_format::shape_button_t data;
      data.iterate_masterpiece([&f, &off](auto& o) {
        o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, off);
        });
      loco_t::button_t::properties_t bp;
      bp.position = data.position;
      bp.position.z += stage->it * op.itToDepthMultiplier;
      bp.size = data.size;
      bp.font_size = data.font_size;
      bp.text = data.text;
      bp.theme = op.theme;
      bp.camera = op.camera;
      bp.viewport = op.viewport;
      auto it = stage->cid_list.NewNodeLast();
      stage->cid_list[it] = bp;
      cid_map[std::make_pair(stage, "button_" + data.id)] = it;
      break;
    }
    case loco_t::shape_type_t::sprite: {
      stage_maker_shape_format::shape_sprite_t data;
      data.iterate_masterpiece([&f, &off](auto& o) {
        o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, off);
        });
      loco_t::unlit_sprite_t::properties_t sp;
      sp.position = data.position;
      sp.position.z += stage->it * op.itToDepthMultiplier;
      sp.size = data.size;
      sp.parallax_factor = data.parallax_factor;
      loco_t::texturepack_t::ti_t ti;
      if (texturepack->qti(data.texturepack_name, &ti)) {
        sp.image = &gloco->default_texture;
      }
      else {
        auto& pd = texturepack->get_pixel_data(ti.pack_id);
        sp.image = &pd.image;
        sp.tc_position = ti.position / pd.image.size;
        sp.tc_size = ti.size / pd.image.size;
      }
      sp.camera = op.camera;
      sp.viewport = op.viewport;

      auto it = stage->cid_list.NewNodeLast();
      stage->cid_list[it] = sp;
      cid_map[std::make_pair(stage, "sprite_" + data.id)] = it;
      break;
    }
    case loco_t::shape_type_t::text: {
      stage_maker_shape_format::shape_text_t data;
      data.iterate_masterpiece([&f, &off](auto& o) {
        o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, off);
        });
      loco_t::text_t::properties_t p;
      p.camera = op.camera;
      p.viewport = op.viewport;

      p.position = data.position;
      p.position.z += stage->it * op.itToDepthMultiplier;
      p.font_size = data.size;
      p.text = data.text;

      auto it = stage->cid_list.NewNodeLast();
      stage->cid_list[it] = p;
      cid_map[std::make_pair(stage, "text_" + data.id)] = it;
      break;
    }
    case loco_t::shape_type_t::hitbox: {
      stage_maker_shape_format::shape_hitbox_t data;
      data.iterate_masterpiece([&f, &off](auto& o) {
        o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, off);
        });
      /*loco_t::vfi_t::properties_t vfip;
      switch (data.vfi_type) {
        case loco_t::vfi_t::shape_t::always: {
          vfip.shape_type = loco_t::vfi_t::shape_t::always;
          vfip.shape.always.z = data.position.z;
          break;
        }
        case loco_t::vfi_t::shape_t::rectangle: {
          vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
          vfip.shape.rectangle.position = data.position;
          vfip.shape.rectangle.size = data.size;
          vfip.shape.rectangle.camera = op.camera;
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
      vfip.ignore_init_move = true;*/
      //stage->cid_list.push_back({});
      //(loco_access)->push_back_input_hitbox((loco_t::vfi_t::shape_id_t*)&(loco_access)->cid_list[stage->cid_list.back().cid].cid, vfip);
      //(loco_access)->cid_list[stage->cid_list.back().cid].cid.shape_type = loco_t::shape_type_t::hitbox;
      //cid_map[std::make_pair(stage, "hitbox_" + data.id)] = stage->cid_list.back().cid;
      fan::throw_error("hitbox not implemented");
      break;
    }
    default: {
      fan::throw_error("i cant find what you talk about - fgm");
      break;
    }
  }
}