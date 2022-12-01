struct sb_menu_maker_type_name {

  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_SafeNext 1
  #define BLL_set_StoreFormat 1
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix instance
  #define BLL_set_type_node uint16_t
  #define BLL_set_node_data \
        fan::graphics::cid_t cid; \
        uint32_t text_id; \
        fan::wstring text; \
        fan::vec3 position; \
        fan_2d::graphics::gui::theme_t theme; \
        loco_t::mouse_button_cb_t mouse_button_cb; \
        loco_t::mouse_move_cb_t mouse_move_cb; \
        loco_t::keyboard_cb_t keyboard_cb;
  #define BLL_set_Link 1
  #include _FAN_PATH(BLL/BLL.h)

  instance_t instances;

  struct properties_t {
    fan::vec3 offset = 0;

    fan::wstring text_value;
    fan::wstring text;

    fan_2d::graphics::gui::theme_t* theme = 0;

    loco_t::mouse_button_cb_t mouse_button_cb = [](const loco_t::mouse_button_data_t&) -> int { return 0; };
    loco_t::mouse_move_cb_t mouse_move_cb = [](const loco_t::mouse_move_data_t&) -> int { return 0; };
    loco_t::keyboard_cb_t keyboard_cb = [](const loco_t::keyboard_data_t&) -> int { return 0; };
  };

  struct select_data_t {
    loco_t::vfi_t* vfi;
    bool selected;
  };

  struct open_properties_t {
    fan::vec3 position;
    f32_t gui_size;

    fan::graphics::theme_list_NodeReference_t theme;

    fan::graphics::viewport_list_NodeReference_t viewport;
    loco_t::matrices_list_NodeReference_t matrices;

    fan::function_t<int(const select_data_t&)> select_cb = [](const select_data_t&) -> int { return 0; };
  };

  auto get_selected_text(loco_t* loco) {
    return instances[selected_id].text;
  }

  void set_selected(loco_t* loco, fan::graphics::cid_t* cid) {
    selected = cid;
    if (selected == nullptr) {
      selected_id.NRI = fan::uninitialized;
      return;
    }
    loco->button.set_theme(selected, loco->button.get_theme(selected), loco_t::button_t::press);
  }
  void set_selected(loco_t* loco, instance_NodeReference_t id) {
    selected = &instances[id].cid;
    selected_id = id;
    //loco->button.set_theme(selected, loco->button.get_theme(selected), loco_t::button_t::press);
  }

  void open(loco_t* loco, const open_properties_t& op) {
    instances.Open();
    global = op;
    global.offset = 0;
    selected = nullptr;
    //loco_t::vfi_t::properties_t vfip;
    //vfip.shape_type = loco_t::vfi_t::shape_t::always;
    //vfip.shape.always.z = op.position.z;

    //vfip.mouse_move_cb = [](const loco_t::vfi_t::mouse_move_data_t& ii_d) -> int { return 0; };
    //vfip.mouse_button_cb = [this, cb = op.select_cb](const loco_t::vfi_t::mouse_button_data_t& mb) -> int {

    //  if (selected == nullptr) {
    //    return 0;
    //  }

    //  if (mb.button != fan::mouse_left) {
    //    return 0;
    //  }

    //  if (mb.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
    //    if (selected && mb.button_state != fan::keyboard_state::press) {
    //      select_data_t sd;
    //      sd.vfi = mb.vfi;
    //      sd.selected = false;
    //      if (cb(sd)) {
    //        return 1;
    //      }
    //      selected = nullptr;
    //      //loco->button.set_theme(selected, loco->button.get_theme(selected), loco_t::button_t::inactive);
    //    }
    //  }

    //  return 0;
    //};
    //vfip.keyboard_cb = [](const loco_t::vfi_t::keyboard_data_t& kd) -> int {
    //  return 0;
    //};

    //empty_click_id = loco->vfi.push_shape(vfip);
  }
  void soft_close(loco_t* loco) {

    auto it = instances.GetNodeFirst();
    while (it != instances.dst) {
      instances.StartSafeNext(it);
      loco->button.erase(&instances[it].cid);
      ((loco_t::button_t::cid_t*)&instances[it].cid)->block_id.NRI = fan::uninitialized;
      it = instances.EndSafeNext();
    }
    //loco->vfi.erase(empty_click_id);
  }
  void close(loco_t* loco) {

    auto it = instances.GetNodeFirst();
    while (it != instances.dst) {
      instances.StartSafeNext(it);
      loco->button.erase(&instances[it].cid);
      it = instances.EndSafeNext();
    }
    instances.Close();
  }
  fan::vec2 get_button_measurements() const {
    return fan::vec2(global.gui_size * 5, global.gui_size);
  }
  auto push_initialized(loco_t* loco, instance_NodeReference_t id, auto nr) {
    loco_t::button_t::properties_t bp;
    bp.position = instances[id].position;
    bp.theme = &instances[id].theme;
    bp.size = fan::vec2(global.gui_size * 5, global.gui_size);
    bp.viewport = global.viewport;
    bp.matrices = global.matrices;
    bp.font_size = global.gui_size;
    bp.text.resize(sizeof(instances[id].text));
    bp.text = instances[id].text;

    bp.mouse_move_cb = [this, loco, id](const loco_t::mouse_move_data_t& d) -> int {
      if (selected == d.cid) {
        loco->button.set_theme(d.cid, loco->button.get_theme(d.cid), loco_t::button_t::press);
      }
      else {
        return instances[id].mouse_move_cb(d);
      }
      return 0;
    };
    bp.mouse_button_cb = [loco, this, id, nr](const loco_t::mouse_button_data_t& d) -> int {

      if (d.button != fan::mouse_left) {
        return 0;
      }

      if (selected == &instances[id].cid && d.button_state == fan::mouse_state::release) {
        goto g_mb_skip;
      }

      if (d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside && d.button_state == fan::mouse_state::release) {
        if (selected != nullptr) {
          loco->button.set_theme(selected, loco->button.get_theme(selected), loco_t::button_t::inactive);
        }
        selected = &instances[id].cid;
        selected_id = id;
      }
    g_mb_skip:
      // if this deleted
      if (instances[id].mouse_button_cb(d)) {
        return 1;
      }

      if (d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside && d.button_state == fan::mouse_state::release) {
        select_data_t sd;
        sd.vfi = d.vfi;
        sd.selected = true;
        if (loco->menu_maker.instances[nr].select_cb(sd)) {
          return 1;
        }
      }
      if (selected == d.cid && d.button_state == fan::mouse_state::release) {
        loco->button.set_theme(d.cid, loco->button.get_theme(d.cid), loco_t::button_t::press);
      }

      return 0;
    };
    bp.keyboard_cb = [this, id](const loco_t::keyboard_data_t& d) -> int {
      return instances[id].keyboard_cb(d);
    };

    loco->button.push_back(&instances[id].cid, bp);

    //loco_t::text_t::properties_t textp;
    //textp.position = bp.position;
    //textp.text = bp.
    //instances[id].text_id = loco->text.push_back()

    return id;
  }

  auto push_back(loco_t* loco, const properties_t& p, auto nr) {
    auto id = instances.NewNodeLast();
    auto& instance = instances[id];
    instance.position = global.position + p.offset;
    instance.position.y += global.offset.y;
    instance.position.z += 1;
    instance.text = p.text;
    instance.mouse_move_cb = p.mouse_move_cb;
    instance.mouse_button_cb = p.mouse_button_cb;
    instance.keyboard_cb = p.keyboard_cb;
    if (p.theme == nullptr) {
      instance.theme = *loco->get_context()->theme_list[global.theme].theme_id;
    }
    else {
      instance.theme = *p.theme;
    }

    auto ret = push_initialized(loco, id, nr);
    global.offset.y += global.gui_size * 2;
    return ret;
  }

  void erase_soft(loco_t* loco, instance_NodeReference_t id) {
    loco->button.erase(&instances[id].cid);
    ((loco_t::button_t::cid_t*)&instances[id].cid)->block_id.NRI = fan::uninitialized;
  }
  void erase(loco_t* loco, instance_NodeReference_t id) {
    loco->button.erase(&instances[id].cid);
    instances.Unlink(id);
    instances.Recycle(id);
  }

  bool is_visually_valid(instance_NodeReference_t id) {
    return ((loco_t::button_t::cid_t*)&instances[id].cid)->block_id != decltype(((loco_t::button_t::cid_t*)&instances[id].cid)->block_id){(uint16_t)fan::uninitialized};
  }

  struct global_t : open_properties_t{
    global_t() = default;

    global_t(const open_properties_t& op) : open_properties_t(op) {
      
    }

    fan::vec2 offset;
  }global;

  fan::graphics::cid_t* selected;
  instance_NodeReference_t selected_id;
};

#undef use_key_lambda