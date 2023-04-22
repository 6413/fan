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

	struct gui_store_t {
		uint16_t type;
		union shape_t {
			struct button_t{
				fan::vec2 position;
				fan::vec2 size;
				fan::string text;
				f32_t font_size;
				loco_t::theme_t theme;
			};
		}shape;
	};

	std::vector<gui_store_t> gui_store;

	struct stage_t {
		
		static constexpr uint8_t stage_options = 0;
		static constexpr uint8_t stage_instance = 1;

		struct stage_e {
			static constexpr uint8_t main = 0;
			static constexpr uint8_t gui = 1;
			static constexpr uint8_t function = 2;
		};

		stage_maker_t* get_stage_maker() {
			return OFFSETLESS(this, stage_maker_t, stage);
		}
	}stage;

	void push_stage_main(const loco_t::menu_maker_button_t::properties_t& p) {
		pile->loco.menu_maker_button.push_back(stage.get_stage_maker()->instances[stage_t::stage_instance].menu_id, p);
	}
	void reopen_stage(uint8_t s) {
		auto nr = stage.get_stage_maker()->instances[s].menu_id;
		auto& instances = pile->loco.menu_maker_button.instances[nr].base.instances;

		auto it = instances.GetNodeFirst();
		while (it != instances.dst) {
			if (pile->loco.menu_maker_button.is_visually_valid(nr, it)) {
				it = it.Next(&instances);
				continue;
			}
			pile->loco.menu_maker_button.push_initialized(nr, it);
			it = it.Next(&instances);
		}
	}
	void close_stage(uint8_t stage) {
    pile->loco.menu_maker_button.erase_menu_soft(instances[stage].menu_id);
	}

	void open_stage_function() {

	}

	void reopen_main() {
		reopen_stage(stage_t::stage_options);
		reopen_stage(stage_t::stage_instance);
	}

	void open_stage(uint8_t stage_) {
		if (current_stage == stage_) {
			return;
		}
		current_stage = stage_;

		switch (current_stage) {
		case stage_t::stage_e::main: {

			loco_t::menu_maker_button_t::open_properties_t op;
			op.theme = &theme;
			op.camera = &camera;
			op.viewport = &viewport;
			op.gui_size = gui_size * 3;
			op.position = fan::vec2(op.gui_size * (5.0 / 3), -1.0 + op.gui_size);
			instances[stage_t::stage_instance].menu_id = pile->loco.menu_maker_button.push_menu(op);
			break;
		}
		case stage_t::stage_e::gui: {
			fgm.load();
			break;
		}
		case stage_t::stage_e::function: {
			open_stage_function();
			break;
		}
		}
	}

	auto write_stage() {
    stage_h.write(&stage_h_str);
	};

	auto append_stage_to_file(const fan::string& stage_name) {
		if (stage_h_str.find(stage_name) != fan::string::npos) {
			return;
		}

		static constexpr const char* find_end_str("\n  using variant_t = std::variant<");
    auto struct_stage_end = stage_h_str.find(find_end_str);

		if (struct_stage_end == fan::string::npos) {
			fan::throw_error("corrupted stage.h");
		}

    auto append_struct = fmt::format(R"(
  struct {0}_t : stage_common_t_t<{0}_t> {{

    using stage_common_t_t::stage_common_t_t;

    static constexpr auto stage_name = "{0}";

    typedef int({0}_t::* button_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int({0}_t::* button_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int({0}_t::* button_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int({0}_t::* button_text_cb_table_t)(const loco_t::text_data_t& d);

    //button_src
    button_mouse_button_cb_table_t button_mouse_button_cb_table[1] = {{ }};
    button_mouse_move_cb_table_t button_mouse_move_cb_table[1] = {{ }};
    button_keyboard_cb_table_t button_keyboard_cb_table[1] = {{ }};
    button_text_cb_table_t button_text_cb_table[1] = {{ }};
    //button_dst
  
    typedef int({0}_t::* hitbox_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int({0}_t::* hitbox_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int({0}_t::* hitbox_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int({0}_t::* hitbox_text_cb_table_t)(const loco_t::text_data_t& d);

    //hitbox_src
    hitbox_mouse_button_cb_table_t hitbox_mouse_button_cb_table[1] = {{ }};
    hitbox_mouse_move_cb_table_t hitbox_mouse_move_cb_table[1] = {{ }};
    hitbox_keyboard_cb_table_t hitbox_keyboard_cb_table[1] = {{ }};
    hitbox_text_cb_table_t hitbox_text_cb_table[1] = {{ }};
    //hitbox_dst

    #include _PATH_QUOTE(stage_loader_path/{1})
  }};
)", 
    stage_name,
    get_file_fullpath(stage_name));
		stage_h_str.insert(struct_stage_end, append_struct);

    static constexpr auto variant_str = "std::variant<";
    auto found = stage_h_str.find(variant_str);
    if (found == fan::string::npos) {
      fan::throw_error("corrupted stage.h");
    }
    found += strlen(variant_str);
    found = stage_h_str.find(">;", found);
    if (found == fan::string::npos) {
      fan::throw_error("corrupted stage.h");
    }
    if (stage_h_str[found - 1] != '<') {
      stage_h_str.insert(found, ",");
      found += 1;
    }
    stage_h_str.insert(found, stage_name + "_t*");
	};

	fan::string get_selected_name(
		loco_t::menu_maker_button_t::instance_NodeReference_t nr
	) {
		return pile->loco.menu_maker_button.instances[nr].base.instances[
      pile->loco.menu_maker_button.get_selected_id(nr)
    ].text;
	}

  void set_selected_name(
    loco_t::menu_maker_button_t::instance_NodeReference_t nr,
    const fan::string& text
  ) {
    pile->loco.menu_maker_button.set_text(
      nr,
      pile->loco.menu_maker_button.get_selected_id(nr),
      text
    );
  }

	fan::string get_selected_name_last() {
		auto nr = pile->loco.menu_maker_button.instances.GetNodeLast();
    return pile->loco.menu_maker_button.instances[nr].base.instances[
      pile->loco.menu_maker_button.instances[nr].base.instances.GetNodeLast()
    ].text;
	}

	auto write_stage_instance() {
		auto stage_name = get_selected_name_last();
		auto file_name = get_file_fullpath(stage_name);
		fan::io::file::write(file_name, stage_instance_tempalte_str, std::ios_base::binary);
    std::ofstream _fgm(get_file_fullpath_runtime(stage_name), std::ios_base::out | std::ios::binary);
    _fgm.write((const char*)&fgm_t::stage_maker_format_version, sizeof(fgm_t::stage_maker_format_version));
		append_stage_to_file(stage_name);
		write_stage();
	};

	void open_options_menu() {
		loco_t::menu_maker_button_t::properties_t p;
		
		auto& current_y = pile->loco.menu_maker_button.get_offset(instances[stage_t::stage_options].menu_id).y;
		auto old_y = current_y;
		current_y = 1.8;

    p.theme = &theme;

    p.text = "Rename";
    p.mouse_button_cb = [this](const loco_t::mouse_button_data_t& mb) -> int {

      use_key_lambda(fan::mouse_left, fan::mouse_state::release);

      #define return_value 0
      #include "helper_functions/rename_cb.h"

      return 0;
    };

    options_ids.push_back(pile->loco.menu_maker_button.push_back(instances[stage_t::stage_options].menu_id, p));
	}

  void create_stage(const fan::string& stage_name){
    loco_t::menu_maker_button_t::properties_t p;
    p.text = stage_name;
    p.mouse_button_cb = [this](const loco_t::mouse_button_data_t& mb) -> int {

      use_key_lambda(fan::mouse_left, fan::mouse_state::release);

      // if switching stage while renaming
      rename_textbox.erase();

      if (mb.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {

        for (auto& i : options_ids) {
          if (pile->loco.menu_maker_button.is_visually_valid(
            instances[stage_t::stage_options].menu_id,
            i
          )) {
            return 0;
          }
          pile->loco.menu_maker_button.push_initialized(
            instances[stage_t::stage_options].menu_id,
            i
          );
        }
      }
      return 0;
    };
    push_stage_main(p);
  };

  bool does_stage_exist(const fan::string& stage_name) {
    auto& instances = pile->loco.menu_maker_button.instances[stage.get_stage_maker()->instances[stage_t::stage_instance].menu_id].base.instances;
    auto it = instances.GetNodeFirst();
    while (it != instances.dst) {
      if (instances[it].text == stage_name) {
        return true;
      }
      it = it.Next(&instances);
    }
    return stage_name == "stage";
  }

	void open_without_init() {
		loco_t::menu_maker_button_t::open_properties_t op;
		op.theme = &theme;
		op.camera = &camera;
		op.viewport = &viewport;

		op.gui_size = gui_size;
		op.position = fan::vec2(-1.0 + op.gui_size * 5, -1.0 + op.gui_size * 1);
		instances[0].menu_id = pile->loco.menu_maker_button.push_menu(op);

		loco_t::menu_maker_button_t::properties_t p;
		p.text = "Create New Stage";
		p.mouse_button_cb = [this](const loco_t::mouse_button_data_t& mb) -> int {

			use_key_lambda(fan::mouse_left, fan::mouse_state::release);

			pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco_var_name);
			open_stage(stage_maker_t::stage_t::stage_e::main);
			instance_t* instance = &instances[stage_maker_t::stage_t::stage_instance];

      static uint32_t x = 0;
      while (does_stage_exist(fan::string("stage") + fan::to_string(x))) { ++x; }

			create_stage(fan::string("stage") + fan::to_string(x));
			write_stage_instance();
			return 0;
		};

		pile->loco.menu_maker_button.push_back(instances[stage_t::stage_options].menu_id, p);
		p.text = "Gui stage";
		p.mouse_button_cb = [this](const loco_t::mouse_button_data_t& mb) {

			use_key_lambda(fan::mouse_left, fan::mouse_state::release);

			pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco_var_name);
			auto nr = instances[stage_t::stage_instance].menu_id;
			auto id = pile->loco.menu_maker_button.get_selected_id(nr);
			if (pile->loco.menu_maker_button.instances[nr].base.instances.inri(id)) {
				return 0;
			}

			close_stage(stage_t::stage_options);
			close_stage(stage_t::stage_instance);
			open_stage(stage_t::stage_e::gui);

			return 1;
		};
		pile->loco.menu_maker_button.push_back(instances[stage_t::stage_options].menu_id, p);
		p.text = "Function stage";

		p.mouse_button_cb = [this](const loco_t::mouse_button_data_t& mb) -> int {

			use_key_lambda(fan::mouse_left, fan::mouse_state::release);

			pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco_var_name);

			open_file_gui(
				get_file_fullpath(
					pile->loco.menu_maker_button.get_selected_text(instances[stage_maker_t::stage_t::stage_instance].menu_id)
				)
			);
			return 0;
		};
		pile->loco.menu_maker_button.push_back(instances[stage_t::stage_options].menu_id, p);
	}

	void open(const char* texturepack_name) {
		
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
		fgm.open(texturepack_name);
		instances.resize(3);

		current_stage = fan::uninitialized;

		theme = loco_t::themes::gray();
		theme.open(pile->loco.get_context());

		erase_theme = loco_t::themes::deep_red();
		erase_theme.open(pile->loco.get_context());

		fan::vec2 window_size = pile->loco.get_window()->get_size();
		fan::vec2 ratio = window_size / window_size.max();
		pile->loco.open_camera(
			&camera,
			fan::vec2(-1, 1) * ratio.x,
			fan::vec2(-1, 1) * ratio.y
		);

		viewport.open(pile->loco.get_context());
		viewport.set(pile->loco.get_context(), 0, window_size, window_size);

		open_without_init();
		open_stage(stage_t::stage_e::main);
		open_options_menu();

    for (auto& i : options_ids) {
		  pile->loco.menu_maker_button.erase_button_soft(instances[stage_t::stage_options].menu_id, i);
    }

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
    });
   
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
		theme.close(pile->loco.get_context());
    pile->loco.get_window()->remove_keys_callback(keys_callback_nr);
	}

	void reopen_from_fgm() {
		//open_without_init();
		reopen_main();
		pile->loco.menu_maker_button.set_selected(
			instances[stage_t::stage_instance].menu_id,
      pile->loco.menu_maker_button.get_selected_id(instances[stage_t::stage_instance].menu_id)
		);
		current_stage = stage_t::stage_e::main;
		//open_erase_button(OFFSETLESS(loco, pile_t, loco_var_name));
	}

	struct instance_t {
		loco_t::menu_maker_button_t::nr_t menu_id;
	};
	std::vector<instance_t> instances;

	uint8_t current_stage;

	loco_t::camera_t camera;
	fan::graphics::viewport_t viewport;
	loco_t::theme_t theme;
	loco_t::theme_t erase_theme;
	fan::string stage_h_str;

	std::vector<loco_t::menu_maker_button_t::base_type_t::instance_NodeReference_t> options_ids;

  loco_t::shape_t rename_textbox;

  fan::window_t::keys_callback_NodeReference_t keys_callback_nr;

  fan::io::file::fstream stage_h;

  #define fgm_button
  //#define fgm_sprite
 // #define fgm_text
  //#define fgm_hitbox
	#include _FAN_PATH(graphics/gui/fgm/fgm.h)
  fgm_t fgm;
};

#undef use_key_lambda