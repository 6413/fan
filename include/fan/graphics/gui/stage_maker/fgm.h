struct fgm_t {
	struct shapes {
		static constexpr uint16_t line = 0;
		static constexpr uint16_t button = 1;
		static constexpr uint16_t sprite = 2;
	};

	struct viewport_area {
		static constexpr uint32_t global = 0;
		static constexpr uint32_t editor = 1;
		static constexpr uint32_t types = 2;
		static constexpr uint32_t properties = 3;
	};

	struct action {
		static constexpr uint32_t move = 1 << 0;
		static constexpr uint32_t resize = 1 << 1;
	};

	struct corners_t {
		static constexpr uint32_t count = 8;
		fan::vec2 corners[count];
	};

	static constexpr fan::vec2 button_size = fan::vec2(0.3, 0.08);
  static constexpr f32_t line_z_depth = 10;
  static constexpr f32_t right_click_z_depth = 11;

	f32_t line_y_offset_between_types_and_properties;

	loco_t* get_loco() {
		// ?
		return ((stage_maker_t*)OFFSETLESS(this, stage_maker_t, fgm))->get_loco();
	}

	// for -1 - 1 matrix
	fan::vec2 translate_viewport_position(const fan::vec2& value) {
		loco_t& loco = *get_loco();
		fan::vec2 window_size = loco.get_window()->get_size();
		return (value + 1) / 2 * window_size;
	}
	fan::vec2 translate_viewport_position_to_coordinate(fan::graphics::viewport_t* to) {
		loco_t& loco = *get_loco();
		fan::vec2 window_size = loco.get_window()->get_size();
    fan::vec2 p = to->get_position() + to->get_size() / 2;

		return p / window_size * 2 - 1;
	}
	static fan::vec2 scale_object_with_viewport(const fan::vec2& size, fan::graphics::viewport_t* from, fan::graphics::viewport_t* to) {
		fan::vec2 f = from->get_size();
		fan::vec2 t = to->get_size();
		return size / (t / f);
	}
	fan::vec2 scale_object_with_viewport(const fan::vec2& size, fan::graphics::viewport_t* from) {
		fan::vec2 f = from->get_size();
		fan::vec2 t = get_loco()->get_window()->get_size();
		return size / (f / t);
	}
	fan::vec2 translate_to_global(const fan::vec2& position) const {
		return position / viewport[viewport_area::global].get_size() * 2 - 1;
	}
	fan::vec2 get_viewport_dst(fan::graphics::viewport_t* from, fan::graphics::viewport_t* to) {
		return (from->get_size() + from->get_position()) / (to->get_size() / 2) - 1;
	}

  void invalidate_right_click_menu() {
    loco_t& loco = *get_loco();
    if (loco.menu_maker.instances.inric(right_click_menu_nr)) {
      return;
    }
    auto v = loco.menu_maker.instances.gnric();
    loco.menu_maker.erase_menu(right_click_menu_nr);
    right_click_menu_nr = v;
  }

	void invalidate_focus() {
		loco_t& loco = *get_loco();
		loco.vfi.invalidate_focus_mouse();
		loco.vfi.invalidate_focus_keyboard();
    invalidate_right_click_menu();
	}

	corners_t get_corners(const fan::vec2& position, const fan::vec2& size) {
		loco_t& loco = *get_loco();
		fan::vec2 c = position;
		fan::vec2 s = size;
		corners_t corners;
		corners.corners[0] = c - s;
		corners.corners[1] = fan::vec2(c.x, c.y - s.y);
		corners.corners[2] = fan::vec2(c.x + s.x, c.y - s.y);
		corners.corners[3] = fan::vec2(c.x - s.x, c.y);
		corners.corners[4] = fan::vec2(c.x + s.x, c.y);
		corners.corners[5] = fan::vec2(c.x - s.x, c.y + s.y);
		corners.corners[6] = fan::vec2(c.x, c.y + s.y);
		corners.corners[7] = fan::vec2(c.x + s.x, c.y + s.y);
		return corners;
	}

	void open_editor_properties() {
		/*menu.clear();

		menu_t::open_properties_t menup;
		menup.matrices = &matrices[viewport_area::properties];
		menup.viewport = &viewport[viewport_area::properties];
		menup.theme = &theme;
		menup.position = fan::vec2(0, -0.8);
		auto nr = menu.push_menu(menup);
		menu_t::properties_t p;
		p.text = "ratio";
		p.text_value = "1, 1";
		menu.push_back(nr, p);*/
	}

	#include "fgm_resize_cb.h"

	void open_from_stage_maker(const fan::wstring& stage_name) {

		properties_nr.NRI = -1;

		line_t::properties_t lp;
		lp.viewport = &viewport[viewport_area::global];
		lp.matrices = &matrices[viewport_area::global];
		lp.color = fan::colors::white;

		// editor window
		line.push_back(lp);
		line.push_back(lp);
		line.push_back(lp);
		line.push_back(lp);

		// properties
		line.push_back(lp);
		line.push_back(lp);

		resize_cb();

		menu_t::open_properties_t op;
		op.matrices = &matrices[viewport_area::types];
		op.viewport = &viewport[viewport_area::types];
		op.theme = &theme;
		op.position = fan::vec2(0, -0.9);
		op.gui_size = 0.08;
		//auto nr = menu.push_menu(op);

		//menu_t::properties_t mp;
		//mp.text = L"button";
		//mp.mouse_button_cb = [nr](const loco_t::mouse_data_t& ii_d) -> int {
		//	pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi), pile_t, loco);
		//	if (ii_d.button != fan::button_left) {
		//		return 0;
		//	}
		//	if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
		//		return 0;
		//	}
		//	pile->stage_maker.fgm.action_flag |= action::move;
		//	builder_button_t::properties_t bbp;
		//	bbp.matrices = &pile->stage_maker.fgm.matrices[viewport_area::editor];
		//	bbp.viewport = &pile->stage_maker.fgm.viewport[viewport_area::editor];
		//	bbp.position = pile->loco.get_mouse_position(
		//		pile->stage_maker.fgm.viewport[viewport_area::editor].get_position(),
		//		pile->stage_maker.fgm.viewport[viewport_area::editor].get_size()
		//	);
		//	
		//	bbp.size = button_size;
		//	//bbp.size = button_size;
		//	bbp.theme = &pile->stage_maker.fgm.theme;
		//	bbp.text = L"button";
		//	bbp.font_size = scale_object_with_viewport(fan::vec2(0.2), &pile->stage_maker.fgm.viewport[viewport_area::types], &pile->stage_maker.fgm.viewport[viewport_area::editor]).x;
		//	
		//	auto& instance = pile->loco.menu_maker.instances[nr].base.instances[pile->loco.menu_maker.instances[nr].base.instances.GetNodeFirst()];
		//	pile->stage_maker.fgm.builder_button.push_back(bbp);
		//	pile->loco.button.set_theme(&instance.cid, loco_t::button_t::inactive);
		//	auto builder_cid = &pile->stage_maker.fgm.builder_button.instance[pile->stage_maker.fgm.builder_button.instance.size() - 1]->cid;
		//	auto ri = pile->loco.button.get_ri(builder_cid);
		//	pile->loco.vfi.set_focus_mouse(ri.vfi_id);
		//	pile->loco.vfi.feed_mouse_button(fan::button_left, fan::mouse_state::press);
		//	pile->stage_maker.fgm.builder_button.open_properties(builder_cid);
		//	
		//	auto stage_name = pile->stage_maker.get_selected_name(
		//		pile,
		//		pile->stage_maker.instances[pile_t::stage_maker_t::stage_t::stage_instance].menu_id,
		//		pile->loco.menu_maker.get_selected_id(pile->stage_maker.instances[pile_t::stage_maker_t::stage_t::stage_instance].menu_id)
		//	);
		//	auto file_name = pile->stage_maker.get_file_fullpath(stage_name);
		//	
		//	fan::string str;
		//	fan::io::file::read(file_name, &str);
		//	
		//	std::size_t button_id = -1;
		//	for (std::size_t j = 0; j < pile->stage_maker.fgm.builder_button.instance.size(); ++j) {
		//		if (&pile->stage_maker.fgm.builder_button.instance[j]->cid == builder_cid) {
		//			button_id = j;
		//			break;
		//		}
		//	}
		//	
		//	if (button_id == -1) {
		//		fan::throw_error("some corruption xd");
		//	}
		//	
		//	if (str.find(fan::to_string(button_id) + "(") != fan::string::npos) {
		//		return 0;
		//	}
		//	
		//	str += "\n\nstatic int mouse_button_cb" + fan::to_string(button_id) + "(const loco_t::mouse_data_t& mb){\n  return 0;\n}";
		//	
		//	fan::io::file::write(file_name, str, std::ios_base::binary);
		//	return 0;
		//};
		//menu.push_back(nr, mp);

		//mp.text = L"sprite";
		//mp.mouse_button_cb = [this, nr](const loco_t::mouse_data_t& ii_d) -> int {
		//	pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi), pile_t, loco);
		//	if (ii_d.button != fan::button_left) {
		//		return 0;
		//	}
		//	if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
		//		return 0;
		//	}
		//	pile->stage_maker.fgm.action_flag |= action::move;
		//	sprite_t::properties_t sp;
		//	sp.matrices = &pile->stage_maker.fgm.matrices[viewport_area::editor];
		//	sp.viewport = &pile->stage_maker.fgm.viewport[viewport_area::editor];
		//	sp.position = pile->loco.get_mouse_position(
		//		pile->stage_maker.fgm.viewport[viewport_area::editor].get_position(),
		//		pile->stage_maker.fgm.viewport[viewport_area::editor].get_size()
		//	);

		//	sp.size = button_size;
		//	auto pd = texturepack.get_pixel_data(default_texture.pack_id);
		//	sp.image = &pd.image;
		//	sp.tc_position = default_texture.position / pd.size;
		//	sp.tc_size = default_texture.size / pd.size;

		//	pile->stage_maker.fgm.sprite.push_back(sp);
		//	auto& instance = pile->stage_maker.fgm.sprite.instances[pile->stage_maker.fgm.sprite.instances.size() - 1];
		//	pile->loco.vfi.set_focus_mouse(instance->vfi_id);
		//	pile->loco.vfi.feed_mouse_button(fan::button_left, fan::mouse_state::press);
		//	pile->stage_maker.fgm.sprite.open_properties(&instance->cid);

		//	return 0;
		//};
		//menu.push_back(nr, mp);

		auto loco = get_loco();

		right_click_menu_nr = loco->menu_maker.instances.gnric();
		static auto push_menu = [this, loco](
      auto mb,
      loco_t::menu_maker_t::properties_t p
      ) {
			pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco);
			loco->menu_maker.push_back(right_click_menu_nr, p);
		};


		loco_t::vfi_t::properties_t vfip;
		vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
		vfip.shape.rectangle.position = translate_viewport_position_to_coordinate(&viewport[viewport_area::types]);
		vfip.shape.rectangle.position.z = right_click_z_depth;
		vfip.shape.rectangle.matrices = &matrices[viewport_area::global];
		vfip.shape.rectangle.viewport = &viewport[viewport_area::global];
		vfip.shape.rectangle.size = viewport[viewport_area::types].get_size() / loco->get_window()->get_size();

    vfip.mouse_button_cb = [this, loco, vfi_id = (loco_t::vfi_t::shape_id_t)0](const loco_t::vfi_t::mouse_data_t& mb) mutable -> int {
      loco_t::menu_maker_t::open_properties_t rcm_op;
		  rcm_op.matrices = &matrices[viewport_area::global];
		  rcm_op.viewport = &viewport[viewport_area::global];
		  rcm_op.theme = &theme;
		  rcm_op.gui_size = 0.04;
			if (mb.button != fan::mouse_right) {
				invalidate_right_click_menu();
				return 0;
			}
			if (mb.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
				invalidate_right_click_menu();
				return 0;
			}
			if (mb.mouse_state != fan::mouse_state::release) {
				invalidate_right_click_menu();
				return 0;
			}
			if (loco->menu_maker.instances.inric(right_click_menu_nr)) {
				rcm_op.position = mb.position + loco->menu_maker.get_button_measurements(rcm_op.gui_size);
				rcm_op.position.z = right_click_z_depth;
				right_click_menu_nr = loco->menu_maker.push_menu(rcm_op);
        push_menu(mb, {
          .text = L"button",
          .mouse_button_cb = [this](const loco_t::mouse_data_t& ii_d) -> int {
			      pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi), pile_t, loco);
			      if (ii_d.button != fan::mouse_left) {
				      return 0;
			      }
            if (ii_d.mouse_state != fan::mouse_state::release) {
              return 0;
            }
			      if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
				      return 0;
			      }
			      builder_button_t::properties_t bbp;
			      bbp.matrices = &pile->stage_maker.fgm.matrices[viewport_area::editor];
			      bbp.viewport = &pile->stage_maker.fgm.viewport[viewport_area::editor];
			      bbp.position = fan::vec3(0, 0, 0);
			
			      bbp.size = button_size;
			      //bbp.size = button_size;
			      bbp.theme = &pile->stage_maker.fgm.theme;
			      bbp.text = L"button";
			      bbp.font_size = scale_object_with_viewport(fan::vec2(0.2), &pile->stage_maker.fgm.viewport[viewport_area::types], &pile->stage_maker.fgm.viewport[viewport_area::editor]).x;
			      pile->stage_maker.fgm.builder_button.push_back(bbp);
			
			      auto builder_cid = &pile->stage_maker.fgm.builder_button.instance[pile->stage_maker.fgm.builder_button.instance.size() - 1]->cid;
			      auto ri = pile->loco.button.get_ri(builder_cid);
			      //pile->loco.vfi.set_focus_mouse(ri.vfi_id);
			      //pile->loco.vfi.feed_mouse_button(fan::button_left, fan::mouse_state::press);
			      pile->stage_maker.fgm.builder_button.open_properties(builder_cid);
			
			      auto stage_name = pile->stage_maker.get_selected_name(
				      pile,
				      pile->stage_maker.instances[pile_t::stage_maker_t::stage_t::stage_instance].menu_id,
				      pile->loco.menu_maker.get_selected_id(pile->stage_maker.instances[pile_t::stage_maker_t::stage_t::stage_instance].menu_id)
			      );
			      auto file_name = pile->stage_maker.get_file_fullpath(stage_name);
			
			      fan::string str;
			      fan::io::file::read(file_name, &str);
			
			      std::size_t button_id = -1;
			      for (std::size_t j = 0; j < pile->stage_maker.fgm.builder_button.instance.size(); ++j) {
				      if (&pile->stage_maker.fgm.builder_button.instance[j]->cid == builder_cid) {
					      button_id = j;
					      break;
				      }
			      }
			
			      if (button_id == -1) {
				      fan::throw_error("some corruption xd");
			      }
			
			      if (str.find(fan::to_string(button_id) + "(") != fan::string::npos) {
				      return 0;
			      }
			
			      str += "\n\nstatic int mouse_button_cb" + fan::to_string(button_id) + "(const loco_t::mouse_button_data_t& mb){\n  return 0;\n}";
			
			      fan::io::file::write(file_name, str, std::ios_base::binary);

            invalidate_right_click_menu();

			      return 1;
		      }
        });
        push_menu(mb, {
          .text = L"text"
        });
        push_menu(mb, {
          .text = L"sprite"
        });

        loco_t::vfi_t::properties_t p;
		    p.shape_type = loco_t::vfi_t::shape_t::always;
		    p.shape.always.z = rcm_op.position.z;
		    p.mouse_button_cb = [this, &vfi_id, loco](const loco_t::vfi_t::mouse_data_t& mb) -> int {
			    pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco);
          invalidate_right_click_menu();
          fan::print("debug0", loco->vfi.focus.mouse.NRI, loco->vfi.focus.keyboard.NRI, loco->vfi.focus.text.NRI);
          loco->vfi.erase(vfi_id);
			    return 1;
          fan::print("debug1", loco->vfi.focus.mouse.NRI, loco->vfi.focus.keyboard.NRI, loco->vfi.focus.text.NRI);
		    };
		    vfi_id = loco->vfi.push_shape(p);
			}

			return 0;
		};
		auto shape_id = loco->push_back_input_hitbox(vfip);

		global_button_t::properties_t gbp;
		gbp.matrices = &matrices[viewport_area::global];
		gbp.viewport = &viewport[viewport_area::global];
		gbp.position = fan::vec2(-0.8, matrices[viewport_area::types].coordinates.top * 0.9);
		gbp.size = button_size / fan::vec2(4, 2);
		gbp.theme = &theme;
		gbp.text = L"<-";
		gbp.mouse_button_cb = [this](const loco_t::mouse_data_t& mb) -> int {
			use_key_lambda(fan::mouse_left, fan::mouse_state::release);

			stage_maker_t* stage_maker = OFFSETLESS(this, stage_maker_t, fgm);
			stage_maker->reopen_from_fgm();
			pile_t* pile = OFFSETLESS(stage_maker->get_loco(), pile_t, loco_var_name);
			write_to_file(stage_maker->get_selected_name(
				pile,
				pile->stage_maker.instances[pile_t::stage_maker_t::stage_t::stage_instance].menu_id,
				stage_maker->in_gui_editor_id
			));

			clear();

			return 1;
		};
		global_button.push_back(gbp);

		open_editor_properties();

		load_from_file(stage_name);
	}

	void open(const char* texturepack_name) {
		loco_t& loco = *get_loco();

    right_click_menu_nr = loco.menu_maker.instances.gnric();

		line_y_offset_between_types_and_properties = 0.0;

		editor_ratio = fan::vec2(1, 1);
		move_offset = 0;
		action_flag = 0;
		theme = fan_2d::graphics::gui::themes::deep_red();
		theme.open(loco.get_context());

		texturepack.open_compiled(&loco, texturepack_name);

		loco_t::texturepack::ti_t ti;
		if (texturepack.qti("test.webp", &ti)) {
			fan::throw_error("failed to load default texture");
		}
		default_texture = ti;

		loco.get_window()->add_resize_callback([this](const fan::window_t::resize_cb_data_t& d) {
			resize_cb();
		});

		// half size
		properties_line_position = fan::vec2(0.5, 0);
		editor_position = fan::vec2(-properties_line_position.x / 2, 0);
		editor_size = editor_position.x + 0.9;

		matrices[viewport_area::global].open(&loco);
		matrices[viewport_area::editor].open(&loco);
		matrices[viewport_area::types].open(&loco);
		matrices[viewport_area::properties].open(&loco);

		viewport[viewport_area::global].open(loco.get_context());
		viewport[viewport_area::editor].open(loco.get_context());
		viewport[viewport_area::types].open(loco.get_context());
		viewport[viewport_area::properties].open(loco.get_context());

		//loco_t::vfi_t::properties_t p;
		//p.shape_type = loco_t::vfi_t::shape_t::always;
		//p.shape.always.z = 0;
		//p.mouse_button_cb = [](const loco_t::vfi_t::mouse_data_t& mb) -> int {
		//	pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco);
		//	return 0;
		//};
		//loco.vfi.push_shape(p);

		//menu.open(fan::vec2(1.05, button_size.y * 1.5));
	}
	void close() {
		clear();
	}
	void clear() {
		line.clear();
		global_button.clear();
		builder_button.clear();
		sprite.clear();
		menu.clear();
	}

	fan::string get_fgm_full_path(const fan::string& stage_name) {
		return fan::string(stage_maker_t::stage_folder_name) + "/" + stage_name + ".fgm";
	}

	void load_from_file(const fan::string& stage_name) {
		fan::string path = get_fgm_full_path(stage_name);
		fan::string f;
		if (!fan::io::file::exists(path)) {
			return;
		}
		fan::io::file::read(path, &f);
		uint64_t off = 0;
		uint32_t instance_count = fan::io::file::read_data<uint32_t>(f, off);
		for (uint32_t i = 0; i < instance_count; i++) {
			auto p = fan::io::file::read_data<fan::vec3>(f, off);
			auto s = fan::io::file::read_data<fan::vec2>(f, off);
			auto fs = fan::io::file::read_data<f32_t>(f, off);
			auto text = fan::io::file::read_data<fan::wstring>(f, off);
			fan::io::file::read_data<fan_2d::graphics::gui::theme_t>(f, off);
			//theme.open(loco->get_context());
			builder_button_t::properties_t bp;
			bp.position = p;
			bp.size = s;
			bp.font_size = fs;
			bp.text = text;
			bp.theme = &theme;
			bp.matrices = &matrices[viewport_area::editor];
			bp.viewport = &viewport[viewport_area::editor];
			builder_button.push_back(bp);
		}
		instance_count = fan::io::file::read_data<uint32_t>(f, off);
		for (uint32_t i = 0; i < instance_count; i++) {
			auto p = fan::io::file::read_data<fan::vec3>(f, off);
			auto s = fan::io::file::read_data<fan::vec2>(f, off);
			auto text_hash = fan::io::file::read_data<uint64_t>(f, off);
			//theme.open(loco->get_context());
			sprite_t::properties_t sp;
			sp.position = p;
			sp.size = s;
			auto pd = texturepack.get_pixel_data(default_texture.pack_id);
			sp.image = &pd.image;
			sp.tc_position = default_texture.position / pd.size;
			sp.tc_size = default_texture.size / pd.size;
			sp.matrices = &matrices[viewport_area::editor];
			sp.viewport = &viewport[viewport_area::editor];
			sprite.push_back(sp);
		}
	}

	void write_to_file(const fan::string& stage_name) {
		auto loco = get_loco();

		fan::string f;
		f.resize(f.size() + sizeof(uint32_t));
		uint32_t instances_count = builder_button.instance.size();
		memcpy(&f[0], &instances_count, sizeof(uint32_t));
		for (auto it : builder_button.instance) {
			auto p = loco->button.get(&it->cid, &loco_t::button_t::vi_t::position);
			auto s = loco->button.get(&it->cid, &loco_t::button_t::vi_t::size);
			auto fs = loco->text.get_properties(
				loco->button.get_ri(&it->cid).text_id
			).font_size;
			auto text = it->text;
			auto theme = loco->button.get_theme(&it->cid);
			
			static auto add_to_f = [&f, &it]<typename T>(const T& o) {
				std::size_t off = f.size();
				if constexpr (std::is_same<fan::string, T>::value || 
						std::is_same<fan::wstring, T>::value
					) {
					uint64_t len = o.size()  * sizeof(typename std::remove_reference_t<decltype(o)>::char_type);
					f.resize(off + sizeof(uint64_t));
					memcpy(&f[off], &len, sizeof(uint64_t));
					off += sizeof(uint64_t);

					f.resize(off + len);
					memcpy(&f[off], o.data(), len);
				}
				else {
					f.resize(off + sizeof(o));
					memcpy(&f[off], &o, sizeof(o));
				}
			};

			add_to_f(p);
			add_to_f(s);
			add_to_f(fs);
			add_to_f(text);
			add_to_f(*theme);
		}

		f.resize(f.size() + sizeof(uint32_t));
		instances_count = sprite.instances.size();
		memcpy(&f[f.size() - sizeof(uint32_t)], &instances_count, sizeof(uint32_t));
		for (auto it : sprite.instances) {
			auto p = loco->sprite.get(&it->cid, &loco_t::sprite_t::vi_t::position);
			auto s = loco->sprite.get(&it->cid, &loco_t::sprite_t::vi_t::size);
			uint64_t text_hash = fan::get_hash("images/test.webp");

			static auto add_to_f = [&f, &it]<typename T>(const T & o) {
				std::size_t off = f.size();
				if constexpr (std::is_same<fan::string, T>::value) {
					uint64_t len = o.size();
					f.resize(off + sizeof(uint64_t));
					memcpy(&f[off], &len, sizeof(uint64_t));
					off += sizeof(uint64_t);

					f.resize(off + o.size());
					memcpy(&f[off], o.data(), o.size());
				}
				else {
					f.resize(off + sizeof(o));
					memcpy(&f[off], &o, sizeof(o));
				}
			};

			add_to_f(p);
			add_to_f(s);
			add_to_f(text_hash);
		}

		fan::io::file::write(
			get_fgm_full_path(stage_name),
			f,
			std::ios_base::binary
		);
	}

	#include "fgm_shape_types.h"

	loco_t::matrices_t matrices[4];
	fan::graphics::viewport_t viewport[4];

	fan_2d::graphics::gui::theme_t theme;

	uint32_t action_flag;

	fan::vec2 click_position;
	fan::vec2 move_offset;
	fan::vec2 resize_offset;
	uint8_t resize_side;

	fan::vec2 properties_line_position;

	fan::vec2 editor_position;
	fan::vec2 editor_size;
	fan::vec2 editor_ratio;

	loco_t::texturepack::ti_t default_texture;

	loco_t::texturepack texturepack;

	loco_t::menu_maker_t::nr_t right_click_menu_nr;
}fgm;