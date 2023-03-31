struct empty_t {
  uint32_t from_string(const fan::string&) {
    return 0;
  }
  fan::string to_string() const {
    return "";
  }
};

#if defined(fgm_button)
struct button_t {

  static constexpr const char* cb_names[] = { "mouse_button","mouse_move", "keyboard", "text" };

  struct properties_t : loco_t::button_t::properties_t {
    fan::string id;
  };

  uint8_t holding_special_key = 0;

  #define fgm_shape_name button
  #define fgm_shape_manual_properties
  #define fgm_shape_instance_data \
    fan::graphics::cid_t cid; \
    uint16_t shape; \
    loco_t::theme_t theme; \
    fan::string id;
  #include "shape_builder.h"

  void close_properties() {
    if (get_fgm()->properties_open) {
      get_fgm()->properties_open = false;
      get_fgm()->properties_nrs.clear();
      get_fgm()->text_box_menu.erase(get_fgm()->properties_nr);
    }
  }

  void open_properties(button_t::instance_t* instance) {

    close_properties();

    get_fgm()->properties_open = true;
    text_box_menu_t::open_properties_t menup;
    menup.camera = &get_fgm()->camera[viewport_area::properties];
    menup.viewport = &get_fgm()->viewport[viewport_area::properties];
    menup.theme = &get_fgm()->theme;
    menup.position = fan::vec2(0, -0.8);
    menup.gui_size = 0.08;
    auto nr = get_fgm()->text_box_menu.push_menu(menup);
    get_fgm()->properties_nr = nr;
    text_box_menu_t::properties_t p;
    auto position = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::vi_t::position);
    p.text = fan::format("{:.2f}, {:.2f}, {:.2f}", position.x, position.y, position.z);
    p.text_value = "add cbs";
    p.mouse_button_cb = [this, instance](const loco_t::mouse_button_data_t& mb) -> int {
      return 0;
    };
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[0]];
      auto text = pile->loco.text_box.get_text(&it.cid);
      fan::vec3 position = fan::string_to<fan::vec3>(text);

      pile->loco.button.set_position(&instance->cid, position);
      pile->loco.button.set_depth(&instance->cid, position.z);

      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));

    auto size = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::vi_t::size);
    p.text = fan::format("{:.2f}, {:.2f}", size.x, size.y);
    p.text_value = "add cbs";
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[1]];
      auto text = pile->loco.text_box.get_text(&it.cid);
      fan::vec2 size = fan::string_to<fan::vec2>(text);

      pile->loco.button.set(&instance->cid, &loco_t::button_t::vi_t::size, size);

      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));

    const auto& text = pile->loco.button.get_text(&instance->cid);
    p.text = text;
    p.text_value = "text";
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[2]];
      auto text = pile->loco.text_box.get_text(&it.cid);
      pile->loco.button.set_text(&instance->cid, text);

      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));
    {
      p.text = instance->id;
      p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
        if (d.key != fan::key_enter) {
          return 0;
        }
        if (d.keyboard_state != fan::keyboard_state::press) {
          return 0;
        }

        auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[3]];
        instance->id = pile->loco.text_box.get_text(&it.cid);

        auto stage_name = get_fgm()->get_stage_maker()->get_selected_name(
          get_fgm()->get_stage_maker()->instances[stage_maker_t::stage_t::stage_instance].menu_id
        );
        auto file_name = get_fgm()->get_stage_maker()->get_file_fullpath(stage_name);

        write_stage_functions(get_fgm(), this, file_name, stage_name, "button", button_t::cb_names);

        return 0;
      };
      get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));
    }
  }
  void push_back(properties_t& p) {
    shape_builder_push_back();
    uint32_t i = instances.size() - 1;
    if (p.id.empty()) {
      static uint32_t id = 0;
      while (does_id_exist(this, std::to_string(id))) { ++id; }
      instances[i]->id = std::to_string(id);
    }
    else {
      instances[i]->id = p.id;
    }

    instances[i]->theme = *(loco_t::theme_t*)pile->loco.get_context()->theme_list[p.theme].theme_id;
    p.mouse_button_cb = [this, instance = instances[i]](const loco_t::mouse_button_data_t& ii_d) -> int {
      switch (ii_d.button) {
        case fan::mouse_scroll_up: {
          if (get_fgm()->action_flag & action::move) {
            fan::vec3 ps = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::vi_t::position);
            ps.z += 1;
            pile->loco.button.set_position(&instance->cid, ps);
            pile->loco.button.set_depth(&instance->cid, ps.z);
            get_fgm()->button.open_properties(instance);
          }
          return 0;
        }
        case fan::mouse_scroll_down: {
          if (get_fgm()->action_flag & action::move) {
            fan::vec3 ps = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::vi_t::position);
            ps.z -= 1;
            ps.z = fan::clamp((f32_t)ps.z, (f32_t)0.f, (f32_t)ps.z);
            pile->loco.button.set_position(&instance->cid, ps);
            pile->loco.button.set_depth(&instance->cid, ps.z);
            get_fgm()->button.open_properties(instance);
          }
          return 0;
        }
        case fan::mouse_left: {
          break;
        }
        default: {
          return 0;
        }
      }
      if (ii_d.button_state == fan::mouse_state::release) {
        get_fgm()->button.release();
        // TODO FIX, erases in near bottom
        if (!get_fgm()->viewport[viewport_area::editor].inside(pile->loco.get_mouse_position()) && !holding_special_key) {
          get_fgm()->button.erase(instance);
        }
        return 0;
      }
      if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
        return 0;
      }
      get_fgm()->action_flag |= action::move;
      auto viewport = pile->loco.button.get_viewport(&instance->cid);
      get_fgm()->click_position = ii_d.position;
      get_fgm()->move_offset = fan::vec2(pile->loco.button.get_button(&instance->cid, &loco_t::button_t::vi_t::position)) - get_fgm()->click_position;
      get_fgm()->resize_offset = get_fgm()->click_position;
      fan::vec3 rp = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::vi_t::position);
      fan::vec3 rs = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::vi_t::size);
      get_fgm()->resize_side = fan_2d::collision::rectangle::get_side_collision(ii_d.position, rp, rs);
      get_fgm()->button.open_properties(instance);
      return 0;
    };
    p.mouse_move_cb = [this, instance = instances[i]](const loco_t::mouse_move_data_t& ii_d) -> int {
      create_shape_move_resize(instance, ii_d);

      return 0;
    };
    p.keyboard_cb = keyboard_cb(instances[i]);
    pile->loco.button.push_back(&instances[i]->cid, p);
    pile->loco.button.set_theme(&instances[i]->cid, loco_t::button_t::released);
    auto builder_cid = &instances[i]->cid;
    auto ri = pile->loco.button.get_ri(builder_cid);
    pile->loco.vfi.set_focus_mouse(ri.vfi_id);
  }
  void erase(instance_t* instance) {
    close_properties();

    pile->loco.button.erase(&instance->cid);

    #if defined(fgm_build_stage_maker)
    auto stage_name = get_fgm()->get_stage_maker()->get_selected_name(
      get_fgm()->get_stage_maker()->instances[stage_maker_t::stage_t::stage_instance].menu_id
    );
    auto file_name = get_fgm()->get_stage_maker()->get_file_fullpath(stage_name);

    get_fgm()->erase_cbs(file_name, stage_name, "button", instance, cb_names);
    #endif

    for (uint32_t i = 0; i < instances.size(); i++) {
      if (&instances[i]->cid == &instance->cid) {
        instances.erase(instances.begin() + i);
        break;
      }
    }
    release();
  }

  fgm_make_clear_f(
    pile->loco.button.erase(&it->cid);
  );

  fan::vec2 get_size(instance_t* instance) {
    return pile->loco.button.get(&instance->cid, &loco_t::button_t::vi_t::size);
  }
  void set_size(instance_t* instance, const fan::vec2& size) {
    pile->loco.button.set_size(&instance->cid, size);
  }

  fan::vec3 get_position(instance_t* instance) {
    return pile->loco.button.get(&instance->cid, &loco_t::button_t::vi_t::position);
  }
  void set_position(instance_t* instance, const fan::vec3& position) {
    pile->loco.button.set_position(&instance->cid, position);
  }

  fan::string to_string() const {
    fan::string f;
    uint32_t instances_count = instances.size();
    fan::write_to_string(f, loco_t::shape_type_t::button);
    fan::write_to_string(f, instances_count);
    for (auto it : instances) {
      stage_maker_shape_format::shape_button_t data;
      data.position = pile->loco.button.get(&it->cid, &loco_t::button_t::vi_t::position);
      data.size = pile->loco.button.get(&it->cid, &loco_t::button_t::vi_t::size);
      data.font_size = pile->loco.text.get_instance(
        &pile->loco.button.get_ri(&it->cid).text_id
      ).font_size;
      data.text = pile->loco.button.get_text(&it->cid);
      data.id = it->id;
      f += shape_to_string(data);
    }
    return f;
  }
  uint64_t from_string(const fan::string& f) {
    uint64_t off = 0;
    loco_t::shape_type_t::_t shape_type = fan::read_data<loco_t::shape_type_t::_t>(f, off);
    if (shape_type != loco_t::shape_type_t::fgm_shape_loco_name) {
      return 0;
    }
    uint32_t instance_count = fan::read_data<uint32_t>(f, off);
    for (uint32_t i = 0; i < instance_count; ++i) {
      stage_maker_shape_format::shape_button_t data;
      data.iterate_masterpiece([&f, &off](auto& o) {
        o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, off);
        });
      button_t::properties_t bp;
      bp.position = data.position;
      bp.size = data.size;
      bp.font_size = data.font_size;
      bp.text = data.text;
      bp.theme = &get_fgm()->theme;
      bp.camera = &get_fgm()->camera[viewport_area::editor];
      bp.viewport = &get_fgm()->viewport[viewport_area::editor];
      bp.id = data.id;
      push_back(bp);
    }
    return off;
  }

  #include "shape_builder_close.h"
};

#else
struct button_t : empty_t {

};
#endif

#if defined(fgm_sprite)
struct sprite_t {
  struct properties_t : loco_t::sprite_t::properties_t {
    fan::string id;
    fan::string texturepack_name;
    #if defined(fgm_build_model_maker)
    uint32_t group_id = 0;
    #endif
  };
  uint8_t holding_special_key = 0;

  #define fgm_shape_name sprite
  #define fgm_shape_manual_properties
  #if defined(fgm_build_model_maker)
    #define fgm_shape_instance_data \
      fan::graphics::cid_t cid; \
      loco_t::vfi_t::shape_id_t vfi_id; \
      uint16_t shape; \
      fan::string texturepack_name; \
      fan::string id; \
      uint32_t group_id;
  #else
    #define fgm_shape_instance_data \
      fan::graphics::cid_t cid; \
      loco_t::vfi_t::shape_id_t vfi_id; \
      uint16_t shape; \
      fan::string texturepack_name; \
      fan::string id;
  #endif
  
  #include "shape_builder.h"

  void close_properties() {
    if (get_fgm()->properties_open) {
      get_fgm()->properties_open = false;
      get_fgm()->properties_nrs.clear();
      get_fgm()->text_box_menu.erase(get_fgm()->properties_nr);
    }
  }

  void open_properties(instance_t* instance) {
    close_properties();
    get_fgm()->properties_open = true;

    text_box_menu_t::open_properties_t menup;
    menup.camera = &get_fgm()->camera[viewport_area::properties];
    menup.viewport = &get_fgm()->viewport[viewport_area::properties];
    menup.theme = &get_fgm()->theme;
    menup.position = fan::vec2(0, -0.8);
    menup.gui_size = 0.08;
    auto nr = get_fgm()->text_box_menu.push_menu(menup);
    get_fgm()->properties_nr = nr;
    text_box_menu_t::properties_t p;
    auto position = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
    p.text = fan::format("{:.2f}, {:.2f}, {:.2f}", position.x, position.y, position.z);
    p.text_value = "add cbs";
    p.mouse_button_cb = [this, instance](const loco_t::mouse_button_data_t& mb) -> int {
      use_key_lambda(fan::mouse_left, fan::mouse_state::release);
      // open cb here

      return 0;
    };
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[0]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      fan::vec3 position = fan::string_to<fan::vec3>(text);

      //f32_t parallax_factor = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::parallax_factor);
      // EDIT PARALAX HERE
      pile->loco.vfi.shape_list[instance->vfi_id].shape_data.shape.rectangle.position = position /* (1 - parallax_factor)*/;
      pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, position);
      pile->loco.sprite.sb_set_depth(&instance->cid, position.z);

      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));

    auto size = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::size);
    p.text = fan::format("{:.2f}, {:.2f}", size.x, size.y);
    p.text_value = "";
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[1]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      fan::vec2 size = fan::string_to<fan::vec2>(text);

      pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::size, size);
      pile->loco.vfi.shape_list[instance->vfi_id].shape_data.shape.rectangle.size = size;

      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));

    p.text = instance->texturepack_name;
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[2]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      loco_t::texturepack_t::ti_t ti;
      if (get_fgm()->texturepack.qti(text, &ti)) {
        fan::print_no_space("failed to load texture:", fan::string(text).c_str());
        return 0;
      }
      auto& data = get_fgm()->texturepack.get_pixel_data(ti.pack_id);
      pile->loco.sprite.set_image(&instance->cid, &data.image);
      pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::tc_position, ti.position / data.image.size);
      pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::tc_size, ti.size / data.image.size);
      instance->texturepack_name = text;
      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));

    auto parallax_factor = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::parallax_factor);
    p.text = fan::format("{:.2f}", parallax_factor);
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[3]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      f32_t parallax = fan::string_to<f32_t>(text);

      pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::parallax_factor, parallax);
      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));

    p.text = instance->id;
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[4]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      if (does_id_exist(this, text)) {
        fan::print("already existing id, skipping");
        return 0;
      }

      instance->id = text;

      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));

    
    #if defined(fgm_build_model_maker)
    p.text = std::to_string(instance->group_id);
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[5]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      instance->group_id = fan::string_to<uint32_t>(text);
      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));
    #endif
  }
  void push_back(properties_t& p) {
    shape_builder_push_back();
    uint32_t i = instances.size() - 1;
    instances[i]->texturepack_name = p.texturepack_name;

    if (p.id.empty()) {
      static uint32_t id = 0;
      while (does_id_exist(this, std::to_string(id))) { ++id; }
      instances[i]->id = std::to_string(id);
    }
    else {
      instances[i]->id = p.id;
    }

    #if defined(fgm_build_model_maker)
      instances[i]->group_id = p.group_id;
    #endif


    loco_t::vfi_t::properties_t vfip;
    vfip.mouse_button_cb = [this, instance = instances[i]](const loco_t::mouse_button_data_t& ii_d) -> int {
      switch (ii_d.button) {
        case fan::mouse_scroll_up: {
          if (get_fgm()->action_flag & action::move) {
            fan::vec3 ps = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
            ps.z += 1;
            pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, ps);
            pile->loco.sprite.sb_set_depth(&instance->cid, ps.z);
            pile->loco.vfi.shape_list[instance->vfi_id].shape_data.depth = ps.z;
            open_properties(instance);
          }
          return 0;
        }
        case fan::mouse_scroll_down: {
          if (get_fgm()->action_flag & action::move) {
            fan::vec3 ps = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
            ps.z -= 1;
            ps.z = fan::clamp((f32_t)ps.z, (f32_t)0.f, (f32_t)ps.z);
            pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, ps);
            pile->loco.sprite.sb_set_depth(&instance->cid, ps.z);
            pile->loco.vfi.shape_list[instance->vfi_id].shape_data.depth = ps.z;
            open_properties(instance);
          }
          return 0;
        }
        case fan::mouse_left: {
          break;
        }
        default: {
          return 0;
        }
      }
      if (ii_d.button_state == fan::mouse_state::press) {
        ii_d.flag->ignore_move_focus_check = true;
        ii_d.vfi->set_focus_keyboard(ii_d.vfi->get_focus_mouse());
      }
      if (ii_d.button_state == fan::mouse_state::release) {
        ii_d.flag->ignore_move_focus_check = false;
      }

      if (ii_d.button_state == fan::mouse_state::release) {
        release();
        // TODO FIX, erases in near bottom
        if (!get_fgm()->viewport[viewport_area::editor].inside(pile->loco.get_mouse_position()) && !holding_special_key) {
          erase(instance);
        }
        return 0;
      }
      if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
        return 0;
      }
      get_fgm()->action_flag |= action::move;
      auto viewport = pile->loco.sprite.get_viewport(&instance->cid);
      get_fgm()->click_position = ii_d.position;
      get_fgm()->move_offset = fan::vec2(pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position)) - get_fgm()->click_position;
      get_fgm()->resize_offset = get_fgm()->click_position;
      fan::vec3 rp = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
      fan::vec3 rs = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::size);
      get_fgm()->resize_side = fan_2d::collision::rectangle::get_side_collision(ii_d.position, rp, rs);
      open_properties(instance);
      return 0;
    };
    vfip.mouse_move_cb = [this, instance = instances[i]](const loco_t::mouse_move_data_t& ii_d) -> int {
      create_shape_move_resize(instance, ii_d);
      return 0;
    };
    vfip.keyboard_cb = keyboard_cb(instances[i]);
    vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
    vfip.shape.rectangle.position = p.position;
    vfip.shape.rectangle.size = p.size;
    vfip.shape.rectangle.camera = p.camera;
    vfip.shape.rectangle.viewport = p.viewport;
    pile->loco.push_back_input_hitbox(&instances[i]->vfi_id, vfip);
    pile->loco.sprite.push_back(&instances[i]->cid, p);
  }
  void erase(instance_t* instance) {
    close_properties();
    pile->loco.sprite.erase(&instance->cid);
    for (uint32_t i = 0; i < instances.size(); i++) {
      if (&instances[i]->cid == &instance->cid) {
        instances.erase(instances.begin() + i);
        break;
      }
    }
    release();
  }

  fgm_make_clear_f(
    pile->loco.sprite.erase(&it->cid);
  pile->loco.vfi.erase(&it->vfi_id);
  );

  fan::vec3 get_position(instance_t* instance) {
    return pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
  }
  void set_position(instance_t* instance, const fan::vec3& position) {
    pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, position);
    pile->loco.vfi.set_rectangle(instance->vfi_id, &loco_t::vfi_t::shape_data_rectangle_t::position, position);
  }

  fan::vec2 get_size(instance_t* instance) {
    return pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::size);
  }
  void set_size(instance_t* instance, const fan::vec2& size) {
    pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::size, size);
    pile->loco.vfi.set_rectangle(instance->vfi_id, &loco_t::vfi_t::shape_data_rectangle_t::size, size);
  }

  fan::string to_string() const {
    fan::string f;
    uint32_t instances_count = instances.size();
    fan::write_to_string(f, loco_t::shape_type_t::sprite);
    fan::write_to_string(f, instances_count);
    for (auto it : instances) {
      stage_maker_shape_format::shape_sprite_t data;
      data.position = pile->loco.sprite.get(&it->cid, &loco_t::sprite_t::vi_t::position);
      data.size = pile->loco.sprite.get(&it->cid, &loco_t::sprite_t::vi_t::size);
      data.parallax_factor = pile->loco.sprite.get(&it->cid, &loco_t::sprite_t::vi_t::parallax_factor);
      data.texturepack_name = it->texturepack_name;
      data.id = it->id;
      #if defined(fgm_build_model_maker)
      data.group_id = it->group_id;
      #endif
      f += shape_to_string(data);
    }
    return f;
  }

  uint64_t from_string(const fan::string& f) {
    uint64_t off = 0;
    loco_t::shape_type_t::_t shape_type = fan::read_data<loco_t::shape_type_t::_t>(f, off);
    if (shape_type != loco_t::shape_type_t::fgm_shape_loco_name) {
      return 0;
    }
    uint32_t instance_count = fan::read_data<uint32_t>(f, off);
    for (uint32_t i = 0; i < instance_count; ++i) {
      stage_maker_shape_format::shape_sprite_t data;
      data.iterate_masterpiece([&f, &off](auto& o) {
        o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, off);
      });
      loco_t::unlit_sprite_t::properties_t sp;
      sp.position = data.position;
      sp.size = data.size;
      sp.parallax_factor = data.parallax_factor;
      loco_t::texturepack_t::ti_t ti;
      if (get_fgm()->texturepack.qti(data.texturepack_name, &ti)) {
        sp.image = &pile->loco.default_texture;
      }
      else {
        auto& pd = get_fgm()->texturepack.get_pixel_data(ti.pack_id);
        sp.image = &pd.image;
        sp.tc_position = ti.position / pd.image.size;
        sp.tc_size = ti.size / pd.image.size;
      }
      sp.camera = &get_fgm()->camera[viewport_area::editor];
      sp.viewport = &get_fgm()->viewport[viewport_area::editor];
      sp.texturepack_name = data.texturepack_name;
      sp.id = data.id;
      #if defined(fgm_build_model_maker)
      sp.group_id = data.group_id;
      #endif
      push_back(sp);
    }
    return off;
  }

  #include "shape_builder_close.h"
};
#else
struct sprite_t : empty_t {

};
#endif

#if defined(fgm_text)
struct text_t {
  uint8_t holding_special_key = 0;

  struct properties_t : loco_t::text_t::properties_t {
    fan::string id;
  };

  #define fgm_shape_name text
  #define fgm_shape_manual_properties
  #define fgm_shape_instance_data \
    fan::graphics::cid_t cid; \
    loco_t::vfi_t::shape_id_t vfi_id; \
    uint16_t shape; \
    fan::string texturepack_name; \
    fan::string id;
  #include "shape_builder.h"

  void close_properties() {
    if (get_fgm()->properties_open) {
      get_fgm()->properties_open = false;
      get_fgm()->properties_nrs.clear();
      get_fgm()->text_box_menu.erase(get_fgm()->properties_nr);
    }
  }

  void open_properties(text_t::instance_t* instance) {
    close_properties();
    get_fgm()->properties_open = true;

    text_box_menu_t::open_properties_t menup;
    menup.camera = &get_fgm()->camera[viewport_area::properties];
    menup.viewport = &get_fgm()->viewport[viewport_area::properties];
    menup.theme = &get_fgm()->theme;
    menup.position = fan::vec2(0, -0.8);
    menup.gui_size = 0.08;
    auto nr = get_fgm()->text_box_menu.push_menu(menup);
    get_fgm()->properties_nr = nr;
    text_box_menu_t::properties_t p;
    auto position = pile->loco.text.get_instance(&instance->cid).position;
    p.text = fan::format("{:.2f}, {:.2f}, {:.2f}", position.x, position.y, position.z);
    p.text_value = "add cbs";
    p.mouse_button_cb = [this, instance](const loco_t::mouse_button_data_t& mb) -> int {
      use_key_lambda(fan::mouse_left, fan::mouse_state::release);

      // open cb here

      return 0;
    };
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[0]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      fan::vec3 position = fan::string_to<fan::vec3>(text);

      pile->loco.text.set_position(&instance->cid, position);
      pile->loco.text.set_depth(&instance->cid, position.z);

      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));

    f32_t size = pile->loco.text.get_font_size(&instance->cid);
    p.text = fan::format("{:.2f}", size);
    p.text_value = "";
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[1]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      f32_t size = fan::string_to<f32_t>(text);

      pile->loco.text.set_font_size(&instance->cid, size);

      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));

    p.text = pile->loco.text.get_instance(&instance->cid).text;
    p.text_value = "";
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[2]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      pile->loco.text.set_text(&instance->cid, text);

      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));

    p.text = instance->id;
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[3]];
      auto text = pile->loco.text_box.get_text(&it.cid);
      instance->id = text;

      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));

    //
    //get_fgm()->button_menu.clear();
    //
    //properties_menu_t::properties_t menup;
    //menup.text = "position";
    //menup.text_value = pile->loco.button.get_button(instance, &loco_t::button_t::instance_t::position).to_string();
    //get_fgm()->button_menu.push_back(menup);
  }

  void push_back(properties_t& p) {
    instances.resize(instances.size() + 1);
    uint32_t i = instances.size() - 1;
    instances[i] = new instance_t;
    instances[i]->shape = loco_t::shape_type_t::text;

    if (p.id.empty()) {
      static uint32_t id = 0;
      while (does_id_exist(this, std::to_string(id))) { ++id; }
      instances[i]->id = std::to_string(id);
    }
    else {
      instances[i]->id = p.id;
    }

    loco_t::vfi_t::properties_t vfip;
    vfip.mouse_button_cb = [this, instance = instances[i]](const loco_t::mouse_button_data_t& ii_d) -> int {
      switch (ii_d.button) {
        case fan::mouse_scroll_up: {
          if (get_fgm()->action_flag & action::move) {
            fan::vec3 ps = pile->loco.text.get_instance(&instance->cid).position;
            ps.z += 1;
            pile->loco.text.set_position(&instance->cid, ps);
            pile->loco.text.set_depth(&instance->cid, ps.z);
            pile->loco.vfi.shape_list[instance->vfi_id].shape_data.depth = ps.z;
            open_properties(instance);
          }
          return 0;
        }
        case fan::mouse_scroll_down: {
          if (get_fgm()->action_flag & action::move) {
            fan::vec3 ps = pile->loco.text.get_instance(&instance->cid).position;
            ps.z -= 1;
            ps.z = fan::clamp((f32_t)ps.z, (f32_t)0.f, (f32_t)ps.z);
            pile->loco.text.set_position(&instance->cid, ps);
            pile->loco.text.set_depth(&instance->cid, ps.z);
            pile->loco.vfi.shape_list[instance->vfi_id].shape_data.depth = ps.z;
            get_fgm()->text.open_properties(instance);
          }
          return 0;
        }
        case fan::mouse_left: {
          break;
        }
        default: {
          return 0;
        }
      }
      if (ii_d.button_state == fan::mouse_state::press) {
        ii_d.flag->ignore_move_focus_check = true;
        ii_d.vfi->set_focus_keyboard(ii_d.vfi->get_focus_mouse());
      }
      if (ii_d.button_state == fan::mouse_state::release) {
        ii_d.flag->ignore_move_focus_check = false;
      }

      if (ii_d.button_state == fan::mouse_state::release) {
        get_fgm()->text.release();
        // TODO FIX, erases in near bottom
        if (!get_fgm()->viewport[viewport_area::editor].inside(pile->loco.get_mouse_position()) && !holding_special_key) {
          get_fgm()->text.erase(instance);
        }
        return 0;
      }
      if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
        return 0;
      }
      get_fgm()->action_flag |= action::move;
      get_fgm()->click_position = ii_d.position;
      get_fgm()->move_offset = fan::vec2(pile->loco.text.get_instance(&instance->cid).position) - get_fgm()->click_position;
      get_fgm()->resize_offset = get_fgm()->click_position;
      fan::vec3 rp = pile->loco.text.get_instance(&instance->cid).position;
      f32_t rs = pile->loco.text.get_font_size(&instance->cid);
      get_fgm()->resize_side = fan_2d::collision::rectangle::get_side_collision(ii_d.position, rp, rs);
      get_fgm()->text.open_properties(instance);
      return 0;
    };
    vfip.mouse_move_cb = [this, instance = instances[i]](const loco_t::mouse_move_data_t& ii_d) -> int {
      create_shape_move_resize(instance, ii_d);
      return 0;
    };
    vfip.keyboard_cb = keyboard_cb(instances[i]);
    pile->loco.text.push_back(&instances[i]->cid, p);
    vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
    vfip.shape.rectangle.position = p.position;
    vfip.shape.rectangle.size = pile->loco.text.get_text_size(&instances[i]->cid);
    vfip.shape.rectangle.camera = p.camera;
    vfip.shape.rectangle.viewport = p.viewport;
    pile->loco.push_back_input_hitbox(&instances[i]->vfi_id, vfip);
  }
  void erase(instance_t* instance) {
    close_properties();
    pile->loco.text.erase(&instance->cid);
    for (uint32_t i = 0; i < instances.size(); i++) {
      if (&instances[i]->cid == &instance->cid) {
        instances.erase(instances.begin() + i);
        break;
      }
    }
    release();
  }

  fgm_make_clear_f(
    pile->loco.text.erase(&it->cid);
  pile->loco.vfi.erase(&it->vfi_id);
  );

  fan::vec3 get_position(instance_t* instance) {
    return pile->loco.text.get_instance(&instance->cid).position;
  }

  fan::vec2 get_size(instance_t* instance) const {
    return pile->loco.text.get_instance(&instance->cid).font_size;
  }
  void set_size(instance_t* instance, const fan::vec2& size) {
    get_fgm()->text.set_font_size(instance, size.x);
    pile->loco.vfi.set_rectangle(instance->vfi_id, &loco_t::vfi_t::shape_data_rectangle_t::size, size);
  }

  void set_position(instance_t* instance, const fan::vec3& position) {
    pile->loco.text.set_position(&instance->cid, position);
    pile->loco.vfi.set_rectangle(instance->vfi_id, &loco_t::vfi_t::shape_data_rectangle_t::position, position);
  }
  void set_font_size(instance_t* instance, f32_t font_size) {
    pile->loco.text.set_font_size(&instance->cid, font_size);
    pile->loco.vfi.set_rectangle(instance->vfi_id, &loco_t::vfi_t::shape_data_rectangle_t::size, pile->loco.text.get_text_size(&instance->cid));
  }

  fan::string to_string() const {
    fan::string f;
    uint32_t instances_count = instances.size();
    fan::write_to_string(f, loco_t::shape_type_t::text);
    fan::write_to_string(f, instances_count);
    for (auto it : instances) {

      stage_maker_shape_format::shape_text_t data;
      data.position = pile->loco.text.get_instance(&it->cid).position;
      data.size = pile->loco.text.get_instance(&it->cid).font_size;
      data.text = pile->loco.text.get_instance(&it->cid).text;
      data.id = it->id;
      f += shape_to_string(data);
    }
    return f;
  }

  uint64_t from_string(const fan::string& f) {
    uint64_t off = 0;
    loco_t::shape_type_t::_t shape_type = fan::read_data<loco_t::shape_type_t::_t>(f, off);
    if (shape_type != loco_t::shape_type_t::fgm_shape_loco_name) {
      return 0;
    }
    uint32_t instance_count = fan::read_data<uint32_t>(f, off);
    for (uint32_t i = 0; i < instance_count; ++i) {
      stage_maker_shape_format::shape_text_t data;
      data.iterate_masterpiece([&f, &off](auto& o) {
        o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, off);
        });
      text_t::properties_t p;
      p.position = data.position;
      p.font_size = data.size;
      p.text = data.text;
      p.camera = &get_fgm()->camera[viewport_area::editor];
      p.viewport = &get_fgm()->viewport[viewport_area::editor];
      p.id = data.id;
      push_back(p);
    }
    return off;
  }

  #include "shape_builder_close.h"
};
#else
struct text_t : empty_t {

};
#endif

#if defined(fgm_hitbox)
struct hitbox_t {

  static constexpr const char* cb_names[] = { "mouse_button","mouse_move", "keyboard", "text" };

  struct properties_t : loco_t::sprite_t::properties_t {
    loco_t::vfi_t::shape_type_t vfi_type;
    fan::string id;
  };

  uint8_t holding_special_key = 0;

  #define fgm_shape_name hitbox
  #define fgm_shape_manual_properties
  #define fgm_shape_instance_data \
    fan::graphics::cid_t cid; \
    loco_t::vfi_t::shape_id_t vfi_id; \
    uint16_t shape; \
    fan::string id; \
    loco_t::vfi_t::shape_type_t shape_type;
  #include "shape_builder.h"

  void close_properties() {
    if (get_fgm()->properties_open) {
      get_fgm()->properties_open = false;
      get_fgm()->properties_nrs.clear();
      get_fgm()->text_box_menu.erase(get_fgm()->properties_nr);
    }
  }

  void open_properties(instance_t* instance) {
    close_properties();
    get_fgm()->properties_open = true;

    text_box_menu_t::open_properties_t menup;
    menup.camera = &get_fgm()->camera[viewport_area::properties];
    menup.viewport = &get_fgm()->viewport[viewport_area::properties];
    menup.theme = &get_fgm()->theme;
    menup.position = fan::vec2(0, -0.8);
    menup.gui_size = 0.08;
    auto nr = get_fgm()->text_box_menu.push_menu(menup);
    get_fgm()->properties_nr = nr;
    text_box_menu_t::properties_t p;
    auto position = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
    p.text = fan::format("{:.2f}, {:.2f}, {:.2f}", position.x, position.y, position.z);
    p.text_value = "add cbs";
    p.mouse_button_cb = [this, instance](const loco_t::mouse_button_data_t& mb) -> int {
      use_key_lambda(fan::mouse_left, fan::mouse_state::release);

      // open cb here

      return 0;
    };
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[0]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      fan::vec3 position = fan::string_to<fan::vec3>(text);

      pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, position);
      pile->loco.sprite.sb_set_depth(&instance->cid, position.z);
      pile->loco.vfi.shape_list[instance->vfi_id].shape_data.depth = position.z;

      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));

    auto size = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::size);
    p.text = fan::format("{:.2f}, {:.2f}", size.x, size.y);
    p.text_value = "";
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[1]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      fan::vec2 size = fan::string_to<fan::vec2>(text);

      pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::size, size);


      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));

    p.text = fan::to_string(instance->shape_type);
    p.text_value = "";
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[2]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      uint32_t shape_type = fan::string_to<uint32_t>(text);

      instance->shape_type = shape_type;

      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));
  }
  void push_back(properties_t& p) {
    shape_builder_push_back();
    uint32_t i = instances.size() - 1;
    instances[i]->shape_type = p.vfi_type;

    if (p.id.empty()) {
      static uint32_t id = 0;
      while (does_id_exist(this, std::to_string(id))) { ++id; }
      instances[i]->id = std::to_string(id);
    }
    else {
      instances[i]->id = p.id;
    }

    loco_t::vfi_t::properties_t vfip;
    vfip.mouse_button_cb = [this, instance = instances[i]](const loco_t::mouse_button_data_t& ii_d) -> int {
      switch (ii_d.button) {
        case fan::mouse_scroll_up: {
          if (get_fgm()->action_flag & action::move) {
            fan::vec3 ps = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
            ps.z += 1;
            pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, ps);
            pile->loco.sprite.sb_set_depth(&instance->cid, ps.z);
            pile->loco.vfi.shape_list[instance->vfi_id].shape_data.depth = ps.z;
            open_properties(instance);
          }
          return 0;
        }
        case fan::mouse_scroll_down: {
          if (get_fgm()->action_flag & action::move) {
            fan::vec3 ps = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
            ps.z -= 1;
            ps.z = fan::clamp((f32_t)ps.z, (f32_t)0.f, (f32_t)ps.z);
            pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, ps);
            pile->loco.sprite.sb_set_depth(&instance->cid, ps.z);
            pile->loco.vfi.shape_list[instance->vfi_id].shape_data.depth = ps.z;
            open_properties(instance);
          }
          return 0;
        }
        case fan::mouse_left: {
          break;
        }
        default: {
          return 0;
        }
      }
      if (ii_d.button_state == fan::mouse_state::press) {
        ii_d.flag->ignore_move_focus_check = true;
        ii_d.vfi->set_focus_keyboard(ii_d.vfi->get_focus_mouse());
      }
      if (ii_d.button_state == fan::mouse_state::release) {
        ii_d.flag->ignore_move_focus_check = false;
      }

      if (ii_d.button_state == fan::mouse_state::release) {
        release();
        // TODO FIX, erases in near bottom
        if (!get_fgm()->viewport[viewport_area::editor].inside(pile->loco.get_mouse_position()) && !holding_special_key) {
          erase(instance);
        }
        return 0;
      }
      if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
        return 0;
      }
      get_fgm()->action_flag |= action::move;
      auto viewport = pile->loco.sprite.get_viewport(&instance->cid);
      get_fgm()->click_position = ii_d.position;
      get_fgm()->move_offset = fan::vec2(pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position)) - get_fgm()->click_position;
      get_fgm()->resize_offset = get_fgm()->click_position;
      fan::vec3 rp = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
      fan::vec3 rs = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::size);
      get_fgm()->resize_side = fan_2d::collision::rectangle::get_side_collision(ii_d.position, rp, rs);
      open_properties(instance);
      return 0;
    };
    vfip.mouse_move_cb = [this, instance = instances[i]](const loco_t::mouse_move_data_t& ii_d) -> int {
      create_shape_move_resize(instance, ii_d);
      return 0;
    };
    vfip.keyboard_cb = keyboard_cb(instances[i]);
    vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
    vfip.shape.rectangle.position = p.position;
    vfip.shape.rectangle.size = p.size;
    vfip.shape.rectangle.camera = p.camera;
    vfip.shape.rectangle.viewport = p.viewport;
    pile->loco.push_back_input_hitbox(&instances[i]->vfi_id, vfip);
    pile->loco.sprite.push_back(&instances[i]->cid, p);
  }
  // erases even the code generated by fgm
  void erase(instance_t* instance) {
    close_properties();

    #if defined(fgm_build_stage_maker)
    auto stage_name = get_fgm()->get_stage_maker()->get_selected_name(
      get_fgm()->get_stage_maker()->instances[stage_maker_t::stage_t::stage_instance].menu_id
    );
    auto file_name = get_fgm()->get_stage_maker()->get_file_fullpath(stage_name);

    get_fgm()->erase_cbs(file_name, stage_name, "hitbox", instance, cb_names);
    #endif

    pile->loco.sprite.erase(&instance->cid);

    for (uint32_t i = 0; i < instances.size(); i++) {
      if (&instances[i]->cid == &instance->cid) {
        instances.erase(instances.begin() + i);
        break;
      }
    }
    release();
  }

  fgm_make_clear_f(
    pile->loco.sprite.erase(&it->cid);
  pile->loco.vfi.erase(&it->vfi_id);
  );

  fan::vec3 get_position(instance_t* instance) {
    return pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
  }
  void set_position(instance_t* instance, const fan::vec3& position) {
    pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, position);
    pile->loco.vfi.set_rectangle(instance->vfi_id, &loco_t::vfi_t::shape_data_rectangle_t::position, position);
  }

  fan::vec2 get_size(instance_t* instance) {
    return pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::size);
  }
  void set_size(instance_t* instance, const fan::vec2& size) {
    pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::size, size);
    pile->loco.vfi.set_rectangle(instance->vfi_id, &loco_t::vfi_t::shape_data_rectangle_t::size, size);
  }

  fan::string to_string() const {
    fan::string f;
    uint32_t instances_count = instances.size();
    fan::write_to_string(f, loco_t::shape_type_t::hitbox);
    fan::write_to_string(f, instances_count);
    for (auto it : instances) {
      stage_maker_shape_format::shape_hitbox_t data;
      auto& shape = pile->loco.vfi.shape_list[it->vfi_id];
      auto& shape_data = shape.shape_data;
      switch (it->shape_type) {
        case loco_t::vfi_t::shape_t::always: {
          data.position = fan::vec3(fan::vec2(pile->loco.sprite.get(&it->cid, &loco_t::sprite_t::vi_t::position)), shape_data.depth);
          data.size = pile->loco.sprite.get(&it->cid, &loco_t::sprite_t::vi_t::size);
          break;
        }
        case loco_t::vfi_t::shape_t::rectangle: {
          data.position = pile->loco.sprite.get(&it->cid, &loco_t::sprite_t::vi_t::position);
          data.size = pile->loco.sprite.get(&it->cid, &loco_t::sprite_t::vi_t::size);
        }
      }
      data.id = it->id;
      data.vfi_type = it->shape_type;
      f += shape_to_string(data);
    }
    return f;
  }

  uint64_t from_string(const fan::string& f) {
    uint64_t off = 0;
    loco_t::shape_type_t::_t shape_type = fan::read_data<loco_t::shape_type_t::_t>(f, off);
    if (shape_type != loco_t::shape_type_t::fgm_shape_loco_name) {
      return 0;
    }
    uint32_t instance_count = fan::read_data<uint32_t>(f, off);
    for (uint32_t i = 0; i < instance_count; ++i) {
      stage_maker_shape_format::shape_hitbox_t data;
      data.iterate_masterpiece([&f, &off](auto& o) {
        o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, off);
        });
      hitbox_t::properties_t sp;
      sp.position = data.position;
      sp.size = data.size;
      sp.image = &get_fgm()->hitbox_image;
      sp.camera = &get_fgm()->camera[viewport_area::editor];
      sp.viewport = &get_fgm()->viewport[viewport_area::editor];
      sp.vfi_type = data.vfi_type;
      sp.id = data.id;
      push_back(sp);
    }
    return off;
  }

  #include "shape_builder_close.h"
};

#else
struct hitbox_t : empty_t {

};
#endif

#if defined(fgm_mark)
struct mark_t {

  static constexpr const char* cb_names[] = { "mouse_button","mouse_move", "keyboard", "text" };

  struct properties_t : loco_t::sprite_t::properties_t {
    loco_t::vfi_t::shape_type_t vfi_type;
    fan::string id;
    #if defined(fgm_build_model_maker)
    uint32_t group_id = 0;
    #endif
  };

  uint8_t holding_special_key = 0;

  #define fgm_shape_name mark
  #define fgm_shape_manual_properties
  #if defined(fgm_build_model_maker)
    #define fgm_shape_instance_data \
        fan::graphics::cid_t cid; \
        loco_t::vfi_t::shape_id_t vfi_id; \
        uint16_t shape; \
        fan::string id; \
        loco_t::vfi_t::shape_type_t shape_type; \
        uint32_t group_id;
  #else
  // make inherit sometime from shape exports
  #define fgm_shape_instance_data \
        fan::graphics::cid_t cid; \
        loco_t::vfi_t::shape_id_t vfi_id; \
        uint16_t shape; \
        fan::string id; \
        loco_t::vfi_t::shape_type_t shape_type;
  #endif
  #include "shape_builder.h"

  void close_properties() {
    if (get_fgm()->properties_open) {
      get_fgm()->properties_open = false;
      get_fgm()->properties_nrs.clear();
      get_fgm()->text_box_menu.erase(get_fgm()->properties_nr);
    }
  }

  void open_properties(instance_t* instance) {
    close_properties();
    get_fgm()->properties_open = true;

    text_box_menu_t::open_properties_t menup;
    menup.camera = &get_fgm()->camera[viewport_area::properties];
    menup.viewport = &get_fgm()->viewport[viewport_area::properties];
    menup.theme = &get_fgm()->theme;
    menup.position = fan::vec2(0, -0.8);
    menup.gui_size = 0.08;
    auto nr = get_fgm()->text_box_menu.push_menu(menup);
    get_fgm()->properties_nr = nr;
    text_box_menu_t::properties_t p;
    auto position = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
    p.text = fan::format("{:.2f}, {:.2f}, {:.2f}", position.x, position.y, position.z);
    p.text_value = "add cbs";
    p.mouse_button_cb = [this, instance](const loco_t::mouse_button_data_t& mb) -> int {
      use_key_lambda(fan::mouse_left, fan::mouse_state::release);

      // open cb here

      return 0;
    };
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[0]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      fan::vec3 position = fan::string_to<fan::vec3>(text);

      pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, position);
      pile->loco.sprite.sb_set_depth(&instance->cid, position.z);
      pile->loco.vfi.shape_list[instance->vfi_id].shape_data.depth = position.z;

      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));

    auto size = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::size);
    p.text = fan::format("{:.2f}, {:.2f}", size.x, size.y);
    p.text_value = "";
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[1]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      fan::vec2 size = fan::string_to<fan::vec2>(text);

      pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::size, size);

      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));

    p.text = instance->id;
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[2]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      if (does_id_exist(this, text)) {
        fan::print("already existing id, skipping");
        return 0;
      }

      instance->id = text;
      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));

    p.text = std::to_string(instance->group_id);
    p.keyboard_cb = [this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[get_fgm()->properties_nrs[3]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      instance->group_id = fan::string_to<uint32_t>(text);
      return 0;
    };
    get_fgm()->properties_nrs.push_back(get_fgm()->text_box_menu.push_back(nr, p));
  }
  void push_back(properties_t& p) {
    shape_builder_push_back();
    uint32_t i = instances.size() - 1;
    instances[i]->shape_type = p.vfi_type;

    if (p.id.empty()) {
      static uint32_t id = 0;
      while (does_id_exist(this, std::to_string(id))) { ++id; }
      instances[i]->id = std::to_string(id);
    }
    else {
      instances[i]->id = p.id;
    }

    #if defined(fgm_build_model_maker)
    instances[i]->group_id = p.group_id;
    #endif

    loco_t::vfi_t::properties_t vfip;
    vfip.mouse_button_cb = [this, instance = instances[i]](const loco_t::mouse_button_data_t& ii_d) -> int {
      switch (ii_d.button) {
        case fan::mouse_scroll_up: {
          if (get_fgm()->action_flag & action::move) {
            fan::vec3 ps = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
            ps.z += 1;
            pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, ps);
            pile->loco.sprite.sb_set_depth(&instance->cid, ps.z);
            pile->loco.vfi.shape_list[instance->vfi_id].shape_data.depth = ps.z;
            open_properties(instance);
          }
          return 0;
        }
        case fan::mouse_scroll_down: {
          if (get_fgm()->action_flag & action::move) {
            fan::vec3 ps = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
            ps.z -= 1;
            ps.z = fan::clamp((f32_t)ps.z, (f32_t)0.f, (f32_t)ps.z);
            pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, ps);
            pile->loco.sprite.sb_set_depth(&instance->cid, ps.z);
            pile->loco.vfi.shape_list[instance->vfi_id].shape_data.depth = ps.z;
            open_properties(instance);
          }
          return 0;
        }
        case fan::mouse_left: {
          break;
        }
        default: {
          return 0;
        }
      }
      if (ii_d.button_state == fan::mouse_state::press) {
        ii_d.flag->ignore_move_focus_check = true;
        ii_d.vfi->set_focus_keyboard(ii_d.vfi->get_focus_mouse());
      }
      if (ii_d.button_state == fan::mouse_state::release) {
        ii_d.flag->ignore_move_focus_check = false;
      }

      if (ii_d.button_state == fan::mouse_state::release) {
        release();
        // TODO FIX, erases in near bottom
        if (!get_fgm()->viewport[viewport_area::editor].inside(pile->loco.get_mouse_position()) && !holding_special_key) {
          erase(instance);
        }
        return 0;
      }
      if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
        return 0;
      }
      get_fgm()->action_flag |= action::move;
      auto viewport = pile->loco.sprite.get_viewport(&instance->cid);
      get_fgm()->click_position = ii_d.position;
      get_fgm()->move_offset = fan::vec2(pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position)) - get_fgm()->click_position;
      get_fgm()->resize_offset = get_fgm()->click_position;
      fan::vec3 rp = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
      fan::vec3 rs = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::size);
      get_fgm()->resize_side = fan_2d::collision::rectangle::get_side_collision(ii_d.position, rp, rs);
      open_properties(instance);
      return 0;
    };
    vfip.mouse_move_cb = [this, instance = instances[i]](const loco_t::mouse_move_data_t& ii_d) -> int {
      create_shape_move_resize(instance, ii_d);
      return 0;
    };
    vfip.keyboard_cb = keyboard_cb(instances[i]);
    vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
    vfip.shape.rectangle.position = p.position;
    vfip.shape.rectangle.size = p.size;
    vfip.shape.rectangle.camera = p.camera;
    vfip.shape.rectangle.viewport = p.viewport;
    pile->loco.push_back_input_hitbox(&instances[i]->vfi_id, vfip);
    pile->loco.sprite.push_back(&instances[i]->cid, p);
  }
  // erases even the code generated by fgm
  void erase(instance_t* instance) {
    close_properties();

    pile->loco.sprite.erase(&instance->cid);

    for (uint32_t i = 0; i < instances.size(); i++) {
      if (&instances[i]->cid == &instance->cid) {
        instances.erase(instances.begin() + i);
        break;
      }
    }
    release();
  }

  fgm_make_clear_f(
    pile->loco.sprite.erase(&it->cid);
  pile->loco.vfi.erase(&it->vfi_id);
  );

  fan::vec3 get_position(instance_t* instance) {
    return pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
  }
  void set_position(instance_t* instance, const fan::vec3& position) {
    pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, position);
    pile->loco.vfi.set_rectangle(instance->vfi_id, &loco_t::vfi_t::shape_data_rectangle_t::position, position);
  }

  fan::vec2 get_size(instance_t* instance) {
    return pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::size);
  }
  void set_size(instance_t* instance, const fan::vec2& size) {
    pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::size, size);
    pile->loco.vfi.set_rectangle(instance->vfi_id, &loco_t::vfi_t::shape_data_rectangle_t::size, size);
  }

  fan::string to_string() const {
    fan::string f;
    uint32_t instances_count = instances.size();
    fan::write_to_string(f, loco_t::shape_type_t::mark);
    fan::write_to_string(f, instances_count);
    for (auto it : instances) {
      stage_maker_shape_format::shape_mark_t data;
      data.position = pile->loco.sprite.get(&it->cid, &loco_t::sprite_t::vi_t::position);
      data.id = it->id;
      #if defined(fgm_build_model_maker)
      data.group_id = it->group_id;
      #endif
      f += shape_to_string(data);
    }
    return f;
  }

  uint64_t from_string(const fan::string& f) {
    uint64_t off = 0;
    loco_t::shape_type_t::_t shape_type = fan::read_data<loco_t::shape_type_t::_t>(f, off);
    if (shape_type != loco_t::shape_type_t::fgm_shape_loco_name) {
      return 0;
    }
    uint32_t instance_count = fan::read_data<uint32_t>(f, off);
    for (uint32_t i = 0; i < instance_count; ++i) {
      stage_maker_shape_format::shape_mark_t data;
      data.iterate_masterpiece([&f, &off](auto& o) {
        o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, off);
        });
      properties_t sp;
      sp.position = data.position;
      sp.size = fan::vec2(0.05, 0.05);
      sp.image = &get_fgm()->mark_image;
      sp.camera = &get_fgm()->camera[viewport_area::editor];
      sp.viewport = &get_fgm()->viewport[viewport_area::editor];
      sp.id = data.id;
      #if defined(fgm_build_model_maker)
      sp.group_id = data.group_id;
      #endif
      push_back(sp);
    }
    return off;
  }

  #include "shape_builder_close.h"
};

#else
struct mark_t : empty_t {

};
#endif

fan_masterpiece_make(
  (button_t)button,
  (sprite_t)sprite,
  (text_t)text,
  (hitbox_t)hitbox,
  (mark_t)mark
);