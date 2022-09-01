struct sb_menu_maker_type_name {

	struct properties_t {
    std::string text_value;
    std::string text;

		loco_t::vfi_t::mouse_button_cb_t mouse_button_cb = [](const loco_t::vfi_t::mouse_button_data_t&) -> void { return; };
    loco_t::vfi_t::mouse_move_cb_t mouse_move_cb = [](const loco_t::vfi_t::mouse_move_data_t&) -> void { return; };
    loco_t::vfi_t::keyboard_cb_t keyboard_cb = [](const loco_t::vfi_t::keyboard_data_t&) -> void { return; };

		uint64_t userptr;
  };

	struct open_properties_t {
		fan::vec3 position;
		f32_t gui_size;

		fan::opengl::theme_list_NodeReference_t theme;

		fan::opengl::viewport_list_NodeReference_t viewport;
		fan::opengl::matrices_list_NodeReference_t matrices;
	};

	void open(const open_properties_t& op) {
		instances.open();
		global = op;
		global.offset = 0;
	}
	void close() {
		for (uint32_t i = 0; i < instances.size(); i++) {
			delete instances[i];
		}
		instances.clear();
	}
	void push_back(loco_t* loco, const properties_t& p) {
		loco_t::button_t::properties_t bp;
		bp.position = global.position + global.offset;
		bp.theme = global.theme;
		bp.text = p.text;
		bp.size = fan::vec2(global.gui_size * 5, global.gui_size);
		bp.viewport = global.viewport;
		bp.matrices = global.matrices;
		bp.font_size = global.gui_size;
		bp.mouse_move_cb = p.mouse_move_cb;
		bp.mouse_button_cb = p.mouse_button_cb;
		bp.keyboard_cb = p.keyboard_cb;
		bp.userptr = p.userptr;
		uint32_t i = instances.resize(instances.size() + 1);
		instances[i] = new instance_t;
		loco->button.push_back(&instances[instances.size() - 1]->cid, bp);
		global.offset.y += bp.size.y * 2;
	}

	struct instance_t {
		fan::opengl::cid_t cid;
		uint32_t text_id;
	};

	fan::hector_t<instance_t*> instances;

	struct global_t : open_properties_t{

		global_t(const open_properties_t& op) : open_properties_t(op) {
			
		}

		fan::vec2 offset;
	}global;
};