struct sb_menu_maker_type_name {

  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_SafeNext 1
  #define BLL_set_StoreFormat 1
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix instance
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeData \
        loco_t::shape_t id; \
        fan::string text; \
        fan::vec3 position; \
        loco_t::theme_t theme; \
        loco_t::mouse_button_cb_t mouse_button_cb; \
        loco_t::mouse_move_cb_t mouse_move_cb; \
        loco_t::keyboard_cb_t keyboard_cb;
  #define BLL_set_Link 1
  #include _FAN_PATH(BLL/BLL.h)

  using nr_t = instance_NodeReference_t;

  instance_t instances;

  struct properties_t {
    fan::vec3 offset = 0;

    fan::string text_value;
    fan::string text;

    loco_t::theme_t* theme = 0;

    loco_t::mouse_button_cb_t mouse_button_cb = [](const loco_t::mouse_button_data_t&) -> int { return 0; };
    loco_t::mouse_move_cb_t mouse_move_cb = [](const loco_t::mouse_move_data_t&) -> int { return 0; };
    loco_t::keyboard_cb_t keyboard_cb = [](const loco_t::keyboard_data_t&) -> int { return 0; };
  };
  
  properties_t get_properties(loco_t::cid_nt_t& id) {
    auto bp = gloco->sb_menu_maker_shape.get_properties(id);
    properties_t p;
    p.theme = gloco->sb_menu_maker_shape.get_theme(id);
    p.text = bp.text;
    //p.mouse_button_cb = bp.mouse_button_cb;
    //p.mouse_button_cb = bp.mouse_button_cb;
    //p.offset = 
    return p;
  }

  struct select_data_t {
    loco_t::vfi_t* vfi;
    bool selected;
  };

  struct open_properties_t {
    fan::vec3 position;
    f32_t gui_size;

    loco_t::theme_t* theme = 0;

    fan::graphics::viewport_t* viewport = 0;
    loco_t::camera_t* camera = 0;

    fan::function_t<int(const select_data_t&)> select_cb = [](const select_data_t&) -> int { return 0; };
  };

  auto get_selected_text(loco_t* loco) {
    return instances[selected_id].text;
  }

  void set_selected(loco_t* loco, nr_t id) {
    selected_id = id;
    loco->sb_menu_maker_shape.set_theme(instances[selected_id].id, loco->sb_menu_maker_shape.get_theme(instances[selected_id].id), loco_t::button_t::pressed);
  }

  void open(loco_t* loco, const open_properties_t& op) {
    instances.Open();
    global = op;
    global.offset = 0;
    selected_id.sic();
  }
  void soft_close(loco_t* loco) {
    fan::print("a");
    auto it = instances.GetNodeFirst();
    while (it != instances.dst) {
      instances.StartSafeNext(it);
      instances[it].id.erase();
      //loco->sb_menu_maker_shape.erase();
      // TODO?
      instances[it].id->block_id = fan::uninitialized;
      //it = it.Next(&instances);
      it = instances.EndSafeNext();
    }
    //loco->vfi.erase(empty_click_id);
  }
  void close(loco_t* loco) {
    soft_close(loco);
    instances.Close();
  }
  static fan::vec2 get_button_measurements(f32_t gui_size) {
    return fan::vec2(gui_size * 5, gui_size);
  }
  fan::vec2 get_button_measurements() const {
    return get_button_measurements(global.gui_size);
  }
  auto push_initialized(loco_t* loco, instance_NodeReference_t id, auto nr) {
    loco_t::CONCAT(sb_menu_maker_shape, _t)::properties_t bp;
    bp.position = instances[id].position;
    bp.theme = &instances[id].theme;
    bp.size = fan::vec2(global.gui_size * 5, global.gui_size);
    bp.viewport = global.viewport;
    bp.camera = global.camera;
    bp.font_size = global.gui_size;
    bp.text.resize(sizeof(instances[id].text));
    bp.text = instances[id].text;

    bp.mouse_move_cb = [this, loco, id](const loco_t::mouse_move_data_t& d) -> int {

      if (selected_id.iic()) {
        return 0;
      }

      if (instances[selected_id].id == d.id) {
        auto temp = d.id;
        loco->sb_menu_maker_shape.set_theme(temp, loco->sb_menu_maker_shape.get_theme(temp), loco_t::button_t::pressed);
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

      auto temp = d.id;
      if (selected_id == id && d.button_state == fan::mouse_state::release) {
        goto g_mb_skip;
      }

      if (d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside && d.button_state == fan::mouse_state::release) {
        if (!selected_id.iic()) {
          loco->sb_menu_maker_shape.set_theme(instances[selected_id].id, loco->sb_menu_maker_shape.get_theme(instances[selected_id].id), loco_t::button_t::released);
        }
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
        if (loco->sb_menu_maker_var_name.instances[nr].select_cb(sd)) {
          return 1;
        }
      }

      
      if (selected_id.iic()) {
        return 0;
      }

      if (instances[selected_id].id == d.id && d.button_state == fan::mouse_state::release) {
        loco->sb_menu_maker_shape.set_theme(temp, loco->sb_menu_maker_shape.get_theme(temp), loco_t::button_t::pressed);
      }

      return 0;
    };
    bp.keyboard_cb = [this, id](const loco_t::keyboard_data_t& d) -> int {
      return instances[id].keyboard_cb(d);
    };

    instances[id].id = bp;

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
      instance.theme = *(loco_t::theme_t*)loco->get_context()->theme_list[global.theme].theme_id;
    }
    else {
      instance.theme = *p.theme;
    }

    auto ret = push_initialized(loco, id, nr);
    global.offset.y += global.gui_size * 2;
    return ret;
  }

  void erase_soft(loco_t* loco, instance_NodeReference_t id) {
    loco->sb_menu_maker_shape.erase(instances[id].id);
    //TODO?
    instances[id].id->block_id= fan::uninitialized;
  }
  void erase(loco_t* loco, instance_NodeReference_t id) {
    loco->sb_menu_maker_shape.erase(instances[id].id);
    instances.Unlink(id);
    instances.Recycle(id);
  }

  bool is_visually_valid(instance_NodeReference_t id) {
    // TODO ?
    return instances[id].id->block_id != decltype(instances[id].id->block_id){(uint16_t)fan::uninitialized};
    //return true;
  }

  struct global_t : open_properties_t{
    global_t() = default;

    global_t(const open_properties_t& op) : open_properties_t(op) {
      
    }

    fan::vec2 offset;
  }global;

  loco_t::cid_nr_t selected;
  nr_t selected_id;
};

#undef use_key_lambda