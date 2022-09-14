struct sb_menu_maker_type_name {

	struct properties_t {
    std::string text_value;
    std::string text;

		fan_2d::graphics::gui::theme_t* theme = 0;

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
    vfip.mouse_button_cb = [&, loco](const loco_t::vfi_t::mouse_button_data_t& mb) -> void {

			if (mb.button != fan::mouse_left) {
				return;
			}

			if (mb.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
				if (selected) {
					//loco->button.set_theme(selected, loco->button.get_theme(selected), loco_t::button_t::inactive);
				}
				//selected = nullptr;
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
	fan::opengl::cid_t* push_back(loco_t* loco, const properties_t& p) {
		loco_t::button_t::properties_t bp;
		bp.position = global.position;
		bp.position.y += global.offset.y;
		bp.position.z += 0.01;
		if (p.theme == 0) {
			bp.theme = global.theme;
		}
		else {
			bp.theme = p.theme;
		}
		bp.text = p.text;
		bp.size = fan::vec2(global.gui_size * 5, global.gui_size);
		bp.get_viewport()  = global.viewport;
		bp.get_matrices() = global.matrices;
		bp.font_size = global.gui_size;
		uint32_t i = instances.resize(instances.size() + 1);
		instances[i] = new instance_t;

		bp.mouse_move_cb = [this, loco, cb = p.mouse_move_cb](const loco_t::mouse_move_data_t& d) -> void { 
			if (selected == d.cid) {
				loco->button.set_theme(d.cid, loco->button.get_theme(d.cid), loco_t::button_t::press);
			}
			else {
				cb(d);
			}
		};
		bp.mouse_button_cb = [loco, this, cb = p.mouse_button_cb, i](const loco_t::mouse_button_data_t& d) -> void {

			if (d.button != fan::mouse_left) {
				return;
			}

			if (d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside && d.button_state == fan::key_state::release) {
				if (selected) {
					loco->button.set_theme(selected, loco->button.get_theme(selected), loco_t::button_t::inactive);
				}
				selected = &instances[i]->cid;
				selected_id = i;
			}
			cb(d);
			if (selected == d.cid && d.button_state == fan::key_state::release) {
				loco->button.set_theme(d.cid, loco->button.get_theme(d.cid), loco_t::button_t::press);
			}
		};
		bp.keyboard_cb = [cb = p.keyboard_cb](const loco_t::keyboard_data_t& d) -> void { 
			cb(d);
		};

		loco->button.push_back(&instances[instances.size() - 1]->cid, bp);
		global.offset.y += bp.size.y * 2;

		return &instances[instances.size() - 1]->cid;
	}

	void erase(loco_t* loco, uint32_t id) {
		loco->button.erase(&instances[id]->cid);
		instances.erase(id);
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
	uint32_t selected_id;
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

#undef use_key_lambda