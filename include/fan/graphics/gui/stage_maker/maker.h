static void open_file_gui(const fan::string& path) {
#if defined(fan_platform_windows)
	// thanks microsoft
	fan::string dir_file_path = path;
	dir_file_path.replace_all("/", R"(\\)");

	ShellExecute(
		0,
		"open",
		dir_file_path.c_str(),
		0,
		0,
		SW_SHOWNORMAL
	);
#endif
}

struct stage_maker_t {

  stage_maker_t() = default;
  stage_maker_t(const char* texturepack_name) {
    open(texturepack_name);
  }

	#define use_key_lambda(key, state) \
		if (mb.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) { \
			return 0; \
		} \
		if (mb.button != key) { \
			return 0; \
					} \
		if (mb.button_state != state) { \
			return 0; \
		}

	static constexpr const char* stage_compile_folder_name = "stages_compile";
  static constexpr const char* stage_runtime_folder_name = "stages_runtime";
	static auto get_file_fullpath(const fan::string& stage_name) {
		return fan::string(stage_compile_folder_name) + "/" +
			stage_name + ".h";
	};

  static auto get_file_fullpath_runtime(const fan::string& stage_name) {
    return fan::string(stage_runtime_folder_name) + "/" +
      stage_name + ".fgm";
  };

	static constexpr const char* stage_instance_tempalte_str = R"(void open(auto& loco) {
  
}

void close(auto& loco){
		
}

void window_resize_callback(auto& loco){
		
}

void update(auto& loco){
	
}
)";

	static constexpr f32_t gui_size = 0.05;

	struct stage_t {
		
		static constexpr uint8_t stage_options = 0;
		static constexpr uint8_t stage_instance = 1;

		static constexpr uint8_t main = 0;
		static constexpr uint8_t gui = 1;
		static constexpr uint8_t function = 2;
	};

//	void push_stage_main(const loco_t::menu_maker_button_t::properties_t& p) {
//		pile->loco.menu_maker_button.push_back(stage.get_stage_maker()->instances[stage_t::stage_instance].menu_id, p);
//	}
//	void reopen_stage(uint8_t s) {
//		auto nr = stage.get_stage_maker()->instances[s].menu_id;
//		auto& instances = pile->loco.menu_maker_button.instances[nr].base.instances;
//
//		auto it = instances.GetNodeFirst();
//		while (it != instances.dst) {
//			if (pile->loco.menu_maker_button.is_visually_valid(nr, it)) {
//				it = it.Next(&instances);
//				continue;
//			}
//			pile->loco.menu_maker_button.push_initialized(nr, it);
//			it = it.Next(&instances);
//		}
//	}
//	void close_stage(uint8_t stage) {
//    pile->loco.menu_maker_button.erase_menu_soft(instances[stage].menu_id);
//	}
//
//	void open_stage_function() {
//
//	}
//
//	void reopen_main() {
//		reopen_stage(stage_t::stage_options);
//		reopen_stage(stage_t::stage_instance);
//	}
//
	void open_stage(uint8_t stage_) {
		if (current_stage == stage_) {
			return;
		}
		current_stage = stage_;

		switch (current_stage) {
		case stage_t::main: {

			//loco_t::menu_maker_button_t::open_properties_t op;
			//op.theme = &theme;
			//op.camera = &camera;
			//op.viewport = &viewport;
			//op.gui_size = gui_size * 3;
			//op.position = fan::vec2(op.gui_size * (5.0 / 3), -1.0 + op.gui_size);
			//instances[stage_t::stage_instance].menu_id = pile->loco.menu_maker_button.push_menu(op);
			break;
		}
		case stage_t::gui: {
			//fgm.load();
			break;
		}
		case stage_t::function: {
			//open_stage_function();
			break;
		}
		}
	}

	auto write_stage() {
    stage_h.write(&stage_h_str);
	};
//
//	auto append_stage_to_file(const fan::string& stage_name) {
//		if (stage_h_str.find(stage_name) != fan::string::npos) {
//			return;
//		}
//
//		static constexpr const char* find_end_str("\n  using variant_t = std::variant<");
//    auto struct_stage_end = stage_h_str.find(find_end_str);
//
//		if (struct_stage_end == fan::string::npos) {
//			fan::throw_error("corrupted stage.h");
//		}
//
//    auto append_struct = fmt::format(R"(
//  struct {0}_t : stage_common_t_t<{0}_t> {{
//    using stage_common_t_t::stage_common_t_t;
//    static constexpr auto stage_name = "{0}";
//    #include _PATH_QUOTE(stage_loader_path/{1})
//  }};
//)", 
//    stage_name,
//    get_file_fullpath(stage_name));
//		stage_h_str.insert(struct_stage_end, append_struct);
//
//    static constexpr auto variant_str = "std::variant<";
//    auto found = stage_h_str.find(variant_str);
//    if (found == fan::string::npos) {
//      fan::throw_error("corrupted stage.h");
//    }
//    found += strlen(variant_str);
//    found = stage_h_str.find(">;", found);
//    if (found == fan::string::npos) {
//      fan::throw_error("corrupted stage.h");
//    }
//    if (stage_h_str[found - 1] != '<') {
//      stage_h_str.insert(found, ",");
//      found += 1;
//    }
//    stage_h_str.insert(found, stage_name + "_t*");
//	};
//
//	fan::string get_selected_name(
//		loco_t::menu_maker_button_t::instance_NodeReference_t nr
//	) {
//		return pile->loco.menu_maker_button.instances[nr].base.instances[
//      pile->loco.menu_maker_button.get_selected_id(nr)
//    ].text;
//	}
//
//  void set_selected_name(
//    loco_t::menu_maker_button_t::instance_NodeReference_t nr,
//    const fan::string& text
//  ) {
//    pile->loco.menu_maker_button.set_text(
//      nr,
//      pile->loco.menu_maker_button.get_selected_id(nr),
//      text
//    );
//  }
//
//	fan::string get_selected_name_last() {
//		auto nr = pile->loco.menu_maker_button.instances.GetNodeLast();
//    return pile->loco.menu_maker_button.instances[nr].base.instances[
//      pile->loco.menu_maker_button.instances[nr].base.instances.GetNodeLast()
//    ].text;
//	}
//
//	auto write_stage_instance() {
//		auto stage_name = get_selected_name_last();
//		auto file_name = get_file_fullpath(stage_name);
//		fan::io::file::write(file_name, stage_instance_tempalte_str, std::ios_base::binary);
//    std::ofstream _fgm(get_file_fullpath_runtime(stage_name), std::ios_base::out | std::ios::binary);
//    _fgm.write((const char*)&fgm_t::stage_maker_format_version, sizeof(fgm_t::stage_maker_format_version));
//		append_stage_to_file(stage_name);
//		write_stage();
//	};
//

  void create_stage(const fan::string& stage_name){
    loco_t::dropdown_t::element_properties_t ep;
    ep.text = stage_name;
    ep.mouse_button_cb = [this](const auto& d) -> int {
      if (d.button != fan::mouse_left) {
        return 0;
      }
      if (d.button_state != fan::mouse_state::release) {
        return 0;
      }
      stage_str = stage_menu.get_element_properties().text;
      return 0;
    };
    stage_menu.add(ep);
  };

  void open_stage_menu() {
    loco_t::dropdown_t::open_properties_t op;
    op.gui_size = gui_size;
    op.position = fan::vec2(1.0 - op.gui_size.x * 5, -1.0 + op.gui_size.y);
    op.camera = &camera;
    op.viewport = &viewport;
    op.theme = &theme;
    op.title = "stage";
    op.titleable = true;
    op.gui_size.x *= 4;
    stage_menu.open(op);

    fan::io::iterate_directory(stage_compile_folder_name, [this](const fan::string& path) {

      fan::string p = path;
      auto len = strlen(fan::string(fan::string(stage_compile_folder_name) + "/").c_str());
      p = p.substr(len, p.size() - len);

      if (p == "stage.h") {
        return;
      }
      p.pop_back();
      p.pop_back();
      create_stage(p);
      if (stage_str.empty()) {
        stage_str = p;
      }
    });
    
    auto& instance = gloco->dropdown.menu_list[stage_menu];
    auto it = instance.GetNodeFirst();
    while (it != instance.dst) {
      if (instance[it].ep.text == stage_str) {
        stage_menu.set_selected(it);
        break;
      }
      it = it.Next(&instance);
    }
  }

	void open_options_menu() {
    loco_t::dropdown_t::open_properties_t op;
    op.gui_size = gui_size;
    op.position = fan::vec2(1.0 - op.gui_size.x * 5, -1.0 + op.gui_size.y);
    op.camera = &camera;
    op.viewport = &viewport;
    op.theme = &theme;
    op.title = "stage";
    op.titleable = true;
    op.gui_size.x *= 4;
    op.position.x = -op.position.x;
    op.titleable = false;
    options_menu.open(op);

    loco_t::dropdown_t::element_properties_t ep;
    ep.text = "create stage";
    options_menu.add(ep);

    ep.text = "gui stage";
    ep.mouse_button_cb = [this](const auto& d) {
      if (d.button != fan::mouse_left) {
        return 0;
      }
      if (d.button_state != fan::mouse_state::release) {
        return 0;
      }

      fgm.load();
      stage_menu.clear();
      options_menu.clear();

      return 1;
    };
    options_menu.add(ep);

    ep.text = "function stage";
    ep.mouse_button_cb = [this](const auto& d) {
      if (d.button != fan::mouse_left) {
        return 0;
      }
      if (d.button_state != fan::mouse_state::release) {
        return 0;
      }

      open_file_gui(get_file_fullpath(stage_menu.get_element_properties().text));

      return 1;
    };
    options_menu.add(ep);
	}

//  bool does_stage_exist(const fan::string& stage_name) {
//    auto& instances = pile->loco.menu_maker_button.instances[stage.get_stage_maker()->instances[stage_t::stage_instance].menu_id].base.instances;
//    auto it = instances.GetNodeFirst();
//    while (it != instances.dst) {
//      if (instances[it].text == stage_name) {
//        return true;
//      }
//      it = it.Next(&instances);
//    }
//    return stage_name == "stage";
//  }
//
//	void open_without_init() {
//		loco_t::menu_maker_button_t::open_properties_t op;
//		op.theme = &theme;
//		op.camera = &camera;
//		op.viewport = &viewport;
//
//		op.gui_size = gui_size;
//		op.position = fan::vec2(-1.0 + op.gui_size * 5, -1.0 + op.gui_size * 1);
//		instances[0].menu_id = pile->loco.menu_maker_button.push_menu(op);
//
//		loco_t::menu_maker_button_t::properties_t p;
//		p.text = "Create New Stage";
//		p.mouse_button_cb = [this](const loco_t::mouse_button_data_t& mb) -> int {
//
//			use_key_lambda(fan::mouse_left, fan::mouse_state::release);
//
//			pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco_var_name);
//			open_stage(stage_maker_t::stage_t::stage_e::main);
//			instance_t* instance = &instances[stage_maker_t::stage_t::stage_instance];
//
//      static uint32_t x = 0;
//      while (does_stage_exist(fan::string("stage") + fan::to_string(x))) { ++x; }
//
//			create_stage(fan::string("stage") + fan::to_string(x));
//			write_stage_instance();
//			return 0;
//		};
//
//		pile->loco.menu_maker_button.push_back(instances[stage_t::stage_options].menu_id, p);
//		p.text = "Gui stage";
//		p.mouse_button_cb = [this](const loco_t::mouse_button_data_t& mb) {
//
//			use_key_lambda(fan::mouse_left, fan::mouse_state::release);
//
//			pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco_var_name);
//			auto nr = instances[stage_t::stage_instance].menu_id;
//			auto id = pile->loco.menu_maker_button.get_selected_id(nr);
//			if (pile->loco.menu_maker_button.instances[nr].base.instances.inri(id)) {
//				return 0;
//			}
//
//			close_stage(stage_t::stage_options);
//			close_stage(stage_t::stage_instance);
//			open_stage(stage_t::stage_e::gui);
//
//			return 1;
//		};
//		pile->loco.menu_maker_button.push_back(instances[stage_t::stage_options].menu_id, p);
//		p.text = "Function stage";
//
//		p.mouse_button_cb = [this](const loco_t::mouse_button_data_t& mb) -> int {
//
//			use_key_lambda(fan::mouse_left, fan::mouse_state::release);
//
//			pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco_var_name);
//
//			open_file_gui(
//				get_file_fullpath(
//					pile->loco.menu_maker_button.get_selected_text(instances[stage_maker_t::stage_t::stage_instance].menu_id)
//				)
//			);
//			return 0;
//		};
//		pile->loco.menu_maker_button.push_back(instances[stage_t::stage_options].menu_id, p);
//	}
//
	void open(const char* texturepack_name) {
		
    fgm.open(texturepack_name);

    fan::io::create_directory("stages_compile");
    fan::io::create_directory("stages_runtime");

    auto stage_path = fan::string(stage_compile_folder_name) + "/stage.h";
    bool data_exists = fan::io::file::exists(stage_path);

    stage_h.open(stage_path);

  if (!data_exists) {
    stage_h_str = R"(struct stage {

  using variant_t = std::variant<>;
};)";
    write_stage();
  }
  else {
    stage_h.read(&stage_h_str);
  }
		//fgm.open(texturepack_name);

		current_stage = stage_t::main;

		theme = loco_t::themes::gray();
		erase_theme = loco_t::themes::deep_red();

		fan::vec2 window_size = pile->loco.get_window()->get_size();
		fan::vec2 ratio = window_size / window_size.max();
		pile->loco.open_camera(
			&camera,
			fan::vec2(-1, 1) * ratio.x,
			fan::vec2(-1, 1) * ratio.y
		);
    pile->loco.open_viewport(&viewport, 0, window_size);

		open_stage(stage_t::main);

    open_stage_menu();
		open_options_menu();
   
    keys_callback_nr = pile->loco.get_window()->add_keys_callback([this](const auto& d) {
      if (d.state != fan::keyboard_state::press) {
        return;
      }

      switch(d.key) {
        case fan::key_f2: {
          #include "helper_functions/rename_cb.h"
          break;
        }
      }
    });
	}
	void close() {
    pile->loco.get_window()->remove_keys_callback(keys_callback_nr);
	}

  fan::string stage_str;

	uint8_t current_stage;

	loco_t::camera_t camera;
	loco_t::viewport_t viewport;

	loco_t::theme_t theme;
	loco_t::theme_t erase_theme;

	fan::string stage_h_str;

  loco_t::shape_t rename_textbox;

  fan::window_t::keys_callback_NodeReference_t keys_callback_nr;

  fan::io::file::fstream stage_h;
  
  loco_t::dropdown_t::menu_id_t stage_menu;
  loco_t::dropdown_t::menu_id_t options_menu;

  #if defined(fgm_build_stage_maker)
  #include _FAN_PATH(graphics/gui/fgm/fgm.h)
  fgm_t fgm;
  #endif
};

#undef use_key_lambda