static void open_file_gui(const fan::string& path) {
#if defined(fan_platform_windows)
	// thanks microsoft
	static fan::string dir_file_path = path;
	dir_file_path.replace_all("/", R"(\\)");

  // sanitizer or fuzzer can give fake crash xd
	ShellExecute(
		0,
		"open",
		"stages_compile\\\\stage0.h",
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

	static constexpr const char* stage_instance_tempalte_str = R"(void open() {
  
}

void close() {
		
}

void window_resize_callback(){
		
}

void update(){
	
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

	void open_stage(uint8_t stage_) {
		current_stage = stage_;
	}

	auto write_stage() {
    stage_h.write(&stage_h_str);
	};

	auto append_stage_to_file(const fan::string& stage_name) {
		if (stage_h_str.find(stage_name) != fan::string::npos) {
			return;
		}

		static constexpr const char find_end_str[]("\n};");
    auto struct_stage_end = stage_h_str.find_last_of(find_end_str);

		if (struct_stage_end == fan::string::npos) {
			fan::throw_error("corrupted stage.h");
		}

    struct_stage_end -= sizeof(find_end_str) - 2;

    auto append_struct = fmt::format(R"(
  struct lstd_defstruct({0}_t) {{
    #include _FAN_PATH(graphics/gui/stage_maker/preset.h)

    static constexpr auto stage_name = "{0}";
    #include _PATH_QUOTE(stage_loader_path/{1})
  }};)", 
    stage_name,
    get_file_fullpath(stage_name));
		stage_h_str.insert(struct_stage_end, append_struct);
	};

	auto write_stage_instance(const fan::string& stage_name) {
		auto file_name = get_file_fullpath(stage_name);
		fan::io::file::write(file_name, stage_instance_tempalte_str, std::ios_base::binary);
    if (!fan::io::file::exists(get_file_fullpath_runtime(stage_name))) {
      std::ofstream _fgm(get_file_fullpath_runtime(stage_name), std::ios_base::out | std::ios::binary);
      _fgm.write((const char*)&fgm_t::stage_maker_format_version, sizeof(fgm_t::stage_maker_format_version));
    }
		append_stage_to_file(stage_name);
		write_stage();
	};

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
    write_stage_instance(stage_name);
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

  bool does_stage_exist(const fan::string& stage_name) {
    auto& instance = gloco->dropdown.menu_list[stage_menu];
    auto it = instance.GetNodeFirst();
    while (it != instance.dst) {
      if (instance[it].get_text() == stage_name) {
        return true;
      }
      it = it.Next(&instance);
    }
    return stage_name == "stage";
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
    ep.mouse_button_cb = [this](const auto& d) {
      if (d.button != fan::mouse_left) {
        return 0;
      }
      if (d.button_state != fan::mouse_state::release) {
        return 0;
      }

      static uint32_t x = 0;
      while (does_stage_exist(fan::string("stage") + fan::to_string(x))) { ++x; }

			create_stage(fan::string("stage") + fan::to_string(x));

      return 0;
    };
    options_menu.add(ep);

    ep.text = "gui stage";
    ep.mouse_button_cb = [this](const auto& d) {
      if (d.button != fan::mouse_left) {
        return 0;
      }
      if (d.button_state != fan::mouse_state::release) {
        return 0;
      }
      if (stage_str == "stage.h") {
        return 0;
      }

      fgm.load();
      fgm.read_from_file(stage_str);
      stage_menu.clear();
      options_menu.clear();

      return 1;
    };
    options_menu.add(ep);

    /*ep.text = "function stage";
    ep.mouse_button_cb = [this](const auto& d) {
      if (d.button != fan::mouse_left) {
        return 0;
      }
      if (d.button_state != fan::mouse_state::release) {
        return 0;
      }

      return 1;
    };
    options_menu.add(ep);*/
	}

	void open(const char* texturepack_name) {
    fgm.open(texturepack_name);

    fan::io::create_directory("stages_compile");
    fan::io::create_directory("stages_runtime");

    auto stage_path = fan::string(stage_compile_folder_name) + "/stage.h";
    bool data_exists = fan::io::file::exists(stage_path);

    stage_h.open(stage_path);

  if (!data_exists) {
    stage_h_str = R"(struct stage {
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
          if (!pile->loco.vfi.get_focus_text().is_invalid()) {
            return;
          }
          loco_t::text_box_t::properties_t tp;
          auto& active_element = stage_menu.get_active_element();
          if (active_element.get_text_shape() == "stage") {
            return;
          }
          tp.camera = &camera;
          tp.viewport = &viewport;
          tp.theme = &theme;
          tp.position = active_element.get_position();
          tp.position.z += 5;
          tp.size = active_element.get_size();
          tp.text = active_element.get_text_shape();
          tp.font_size = active_element.get_font_size();
          tp.keyboard_cb = [this, &active_element](const loco_t::keyboard_data_t& d) -> int {
            if (d.keyboard_state != fan::keyboard_state::press) {

              return 0;
            }
            switch (d.key) {

              case fan::key_enter: {
                auto temp_id = d.id;
                auto new_name = pile->loco.text_box.get_text(temp_id);
                if (!does_stage_exist(new_name)) {
                  do {
                    auto old_name = active_element.get_text_shape();

                    #if defined(fan_platform_windows)
                      static std::regex windows_filename_regex(R"([^\\/:*?"<>|\r\n]+(?!\\)$)");
                    #elif defined(fan_platform_unix)
                      static std::regex unix_filename_regex("[^/]+$");
                    #endif

                    if (new_name.empty() || 
                      #if defined(fan_platform_windows)
                        !std::regex_match(new_name, windows_filename_regex)
                      #elif defined(fan_platform_unix)
                        !std::regex_match(new_name, unix_filename_regex)
                      #endif
                      ) {
                      fan::print("invalid stage name");
                      break;
                    }

                    if (fan::io::file::rename(get_file_fullpath(old_name), get_file_fullpath(new_name)) ||
                        fan::io::file::rename(get_file_fullpath_runtime(old_name), get_file_fullpath_runtime(new_name))) {

                      fan::print_format("failed to rename file from:{} - to:{}", old_name, new_name);
                      break;
                    }
                    if (old_name == new_name) {
                      break;
                    }
                    std::regex rg(fan::format(R"(\b{0}(_t)?\b)", old_name));
                    stage_h_str = std::regex_replace(stage_h_str, rg, new_name + R"($1)");
                    write_stage();
                    active_element.set_text(new_name);
                    stage_str = new_name;
                  } while (0);
                }
                else {
                  fan::print("stage name already exists");
                }
                rename_textbox.erase();
                return 1;
              }
            }
            return 0;
          };

          rename_textbox = tp;
          rename_textbox.set_focus();

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