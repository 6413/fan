struct sb_menu_maker_type_name {

	struct properties_t {
    std::string text_value;
    std::string text;

		loco_t::mouse_button_cb_t mouse_button_cb = [](const loco_t::mouse_button_data_t&) -> void { return; };
    loco_t::mouse_move_cb_t mouse_move_cb = [](const loco_t::mouse_move_data_t&) -> void { return; };
    loco_t::keyboard_cb_t keyboard_cb = [](const loco_t::keyboard_data_t&) -> void { return; };
  };

	struct open_properties_t {
		fan::vec3 position;
		f32_t gui_size;

		fan::opengl::theme_list_NodeReference_t theme;

		fan::opengl::viewport_list_NodeReference_t viewport;
		fan::opengl::matrices_list_NodeReference_t matrices;
	};

	void open(loco_t* loco, const open_properties_t& op) {
		instances.open();
		global = op;
		global.offset = 0;
		selected = nullptr;
		loco_t::vfi_t::properties_t vfip;
    vfip.shape_type = loco_t::vfi_t::shape_t::always;
    vfip.shape.always.z = op.position.z;

    vfip.mouse_move_cb = [](const loco_t::vfi_t::mouse_move_data_t& ii_d) -> void {};
    vfip.mouse_button_cb = [&](const loco_t::vfi_t::mouse_button_data_t& ii_d) -> void {
			if (ii_d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
				selected = nullptr;
			}
		};
    vfip.keyboard_cb = [](const loco_t::vfi_t::keyboard_data_t& kd) -> void {
		
		};

    empty_click_id = loco->vfi.push_shape(vfip);
	}
	void close(loco_t* loco) {
		for (uint32_t i = 0; i < instances.size(); i++) {
			loco->button.erase(&instances[i]->cid);
			delete instances[i];
		}
		loco->vfi.erase(empty_click_id);
		instances.clear();
	}
	void push_back(loco_t* loco, const properties_t& p) {
		loco_t::button_t::properties_t bp;
		bp.position = global.position;
		bp.position.y += global.offset.y;
		bp.position.z += 0.01;
		bp.theme = global.theme;
		bp.text = p.text;
		bp.size = fan::vec2(global.gui_size * 5, global.gui_size);
		bp.get_viewport()  = global.viewport;
		bp.get_matrices() = global.matrices;
		bp.font_size = global.gui_size;
		uint32_t i = instances.resize(instances.size() + 1);
		instances[i] = new instance_t;

		bp.mouse_move_cb = [cb = p.mouse_move_cb](const loco_t::mouse_move_data_t& d) -> void { 
			cb(d);
		};
		bp.mouse_button_cb = [this, cb = p.mouse_button_cb, instance = instances[i]](const loco_t::mouse_button_data_t& d) -> void {
			if (d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
				selected = &instance->cid;
			}
			cb(d);
		};
		bp.keyboard_cb = [cb = p.keyboard_cb](const loco_t::keyboard_data_t& d) -> void { 
			cb(d);
		};

		loco->button.push_back(&instances[instances.size() - 1]->cid, bp);
		global.offset.y += bp.size.y * 2;
	}

	struct instance_t {
		fan::opengl::cid_t cid;
		uint32_t text_id;
	};

	fan::hector_t<instance_t*> instances;

	struct global_t : open_properties_t{
		global_t() = default;

		global_t(const open_properties_t& op) : open_properties_t(op) {
			
		}

		fan::vec2 offset;
	}global;

	fan::opengl::cid_t* selected;
	loco_t::vfi_t::shape_id_t empty_click_id;

	void set_empty_click_mouse_move_cb(loco_t* loco, loco_t::vfi_t::mouse_move_cb_t mouse_move_cb) {
    loco->vfi.set_common_data(empty_click_id, &loco_t::vfi_t::common_shape_data_t::mouse_move_cb, mouse_move_cb);
  }
  void set_empty_click_mouse_button_cb(loco_t* loco, loco_t::vfi_t::mouse_button_cb_t mouse_button_cb) {
    loco->vfi.set_common_data(empty_click_id, &loco_t::vfi_t::common_shape_data_t::mouse_button_cb, mouse_button_cb);
  }
  void set_empty_click_keyboard_cb(loco_t* loco, loco_t::vfi_t::keyboard_cb_t keyboard_cb) {
    loco->vfi.set_common_data(empty_click_id, &loco_t::vfi_t::common_shape_data_t::keyboard_cb, keyboard_cb);
  }
};