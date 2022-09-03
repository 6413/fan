struct sb_menu_maker_type_name {

	struct properties_t {
    std::string text_value;
    std::string text;

		loco_t::mouse_button_cb_t mouse_button_cb = [](const loco_t::mouse_button_data_t&) -> void { return; };
    loco_t::mouse_move_cb_t mouse_move_cb = [](const loco_t::mouse_move_data_t&) -> void { return; };
    loco_t::keyboard_cb_t keyboard_cb = [](const loco_t::keyboard_data_t&) -> void { return; };

		uint64_t userptr;
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
    vfip.udata = (uint64_t)this;

    vfip.mouse_move_cb = [](const loco_t::vfi_t::mouse_move_data_t& mm_d) -> void {

		};
    vfip.mouse_button_cb = [](const loco_t::vfi_t::mouse_button_data_t& ii_d) -> void {
			if (ii_d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
				((sb_menu_maker_type_name*)ii_d.udata)->selected = nullptr;
			}
		};
    vfip.keyboard_cb = [](const loco_t::vfi_t::keyboard_data_t& kd) -> void {
		
		};

    empty_click_id = loco->vfi.push_shape(vfip);
	}
	void close(loco_t* loco) {
		for (uint32_t i = 0; i < instances.size(); i++) {
			// delete allocated vfi udata
			assert(0);
			loco->button.erase(&instances[i]->cid);
			delete instances[i];
		}
		loco->vfi.erase(empty_click_id);
		instances.clear();
	}
	void push_back(loco_t* loco, const properties_t& p) {
		loco_t::button_t::properties_t bp;
		bp.position = global.position + global.offset + fan::vec3(0, 0, global.position.z + 0.01);
		bp.theme = global.theme;
		bp.text = p.text;
		bp.size = fan::vec2(global.gui_size * 5, global.gui_size);
		bp.viewport = global.viewport;
		bp.matrices = global.matrices;
		bp.font_size = global.gui_size;
		bp.mouse_move_cb = [](const loco_t::mouse_move_data_t& d) -> void { 
			loco_t* loco = OFFSETLESS(d.vfi, loco_t, vfi);
			cb_data_t* cb_data = (cb_data_t*)d.udata;
			instance_t* instance = (instance_t*)cb_data->udata2;
			loco_t::mouse_move_data_t dd = d;
			dd.udata = (uint64_t)instance;
			instance->mouse_move_cb(dd);
		};
		bp.mouse_button_cb = [](const loco_t::mouse_button_data_t& d) -> void { 
			loco_t* loco = OFFSETLESS(d.vfi, loco_t, vfi);
			cb_data_t* cb_data = (cb_data_t*)d.udata;
			instance_t* instance = (instance_t*)cb_data->udata2;
			loco_t::mouse_button_data_t dd = d;
			dd.udata = (uint64_t)instance;
			if (d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
				((sb_menu_maker_type_name*)cb_data->udata)->selected = &instance->cid;
			}
			instance->mouse_button_cb(dd);
		};
		bp.keyboard_cb = [](const loco_t::keyboard_data_t& d) -> void { 
			loco_t* loco = OFFSETLESS(d.vfi, loco_t, vfi);
			cb_data_t* cb_data = (cb_data_t*)d.udata;
			instance_t* instance = (instance_t*)cb_data->udata2;
			loco_t::keyboard_data_t dd = d;
			dd.udata = (uint64_t)instance;
			instance->keyboard_cb(dd);
		};
		uint32_t i = instances.resize(instances.size() + 1);
		instances[i] = new instance_t;

		instances[i]->mouse_button_cb = p.mouse_button_cb;
		instances[i]->mouse_move_cb = p.mouse_move_cb;
		instances[i]->keyboard_cb = p.keyboard_cb;

		cb_data_t cb_data;
		cb_data.udata = (uint64_t)this;
		cb_data.udata2 = (uint64_t)instances[i];
		bp.userptr = (uint64_t)new cb_data_t(cb_data);
		loco->button.push_back(&instances[instances.size() - 1]->cid, bp);
		global.offset.y += bp.size.y * 2;
	}

	struct instance_t {
		fan::opengl::cid_t cid;
		uint32_t text_id;
		loco_t::mouse_button_cb_t mouse_button_cb = [](const loco_t::mouse_button_data_t&) -> void { return; };
    loco_t::mouse_move_cb_t mouse_move_cb = [](const loco_t::mouse_move_data_t&) -> void { return; };
    loco_t::keyboard_cb_t keyboard_cb = [](const loco_t::keyboard_data_t&) -> void { return; };
		uint64_t udata;
	};

	fan::hector_t<instance_t*> instances;

	struct global_t : open_properties_t{

		global_t(const open_properties_t& op) : open_properties_t(op) {
			
		}

		fan::vec2 offset;
	}global;

	struct cb_data_t {
		uint64_t udata;
		uint64_t udata2;
	};

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