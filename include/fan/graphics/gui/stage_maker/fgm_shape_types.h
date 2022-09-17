struct line_t {
	using properties_t = loco_t::line_t::properties_t;

	loco_t* get_loco() {
		return ((stage_maker_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, line), stage_maker_t, fgm))->get_loco();
	}
	pile_t* get_pile() {
		return OFFSETLESS(get_loco(), pile_t, loco_var_name);
	}

	void push_back(properties_t& p) {
		loco_t& loco = *get_loco();
		instance.resize(instance.size() + 1);
		uint32_t i = instance.size() - 1;
		instance[i] = new instance_t;
		instance[i]->shape = shapes::line;
		loco.line.push_back(&instance[i]->cid, p);
	}
	void clear() {
		loco_t& loco = *get_loco();
		for (auto& it : instance) {
			loco.line.erase(&it->cid);
			delete it;
		}
		instance.clear();
	}

	struct instance_t {
		fan::opengl::cid_t cid;
		uint16_t shape;
	};

	std::vector<instance_t*> instance;
}line;

struct global_button_t {
	using properties_t = loco_t::button_t::properties_t;

	loco_t* get_loco() {
		return ((stage_maker_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, global_button), stage_maker_t, fgm))->get_loco();
	}
	pile_t* get_pile() {
		return OFFSETLESS(get_loco(), pile_t, loco_var_name);
	}

	void push_back(properties_t& p) {
		p.position.z = 1;
		loco_t& loco = *get_loco();
		instance.resize(instance.size() + 1);
		uint32_t i = instance.size() - 1;
		instance[i] = new instance_t;
		instance[i]->shape = shapes::button;
		loco.button.push_back(&instance[i]->cid, p);
	}

	void clear() {
		loco_t& loco = *get_loco();
		for (auto& it : instance) {
			loco.button.erase(&it->cid);
			delete it;
		}
		instance.clear();
	}

	struct instance_t {
		fan::opengl::cid_t cid;
		uint16_t shape;
	};
	std::vector<instance_t*> instance;
}global_button;

struct editor_button_t {
	using properties_t = loco_t::button_t::properties_t;

	loco_t* get_loco() {
		return ((stage_maker_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, editor_button), stage_maker_t, fgm))->get_loco();
	}
	pile_t* get_pile() {
		return OFFSETLESS(get_loco(), pile_t, loco_var_name);
	}

	void push_back(properties_t& p) {
		p.position.z = 1;
		loco_t& loco = *get_loco();
		instance.resize(instance.size() + 1);
		uint32_t i = instance.size() - 1;
		instance[i] = new instance_t;
		instance[i]->shape = shapes::button;
		p.mouse_button_cb = [instance = instance[i]](const loco_t::mouse_button_data_t& ii_d) -> int {
			pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi), pile_t, loco);
			if (ii_d.button != fan::mouse_left) {
				return 0;
			}
			if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
				return 0;
			}
			pile->stage_maker.fgm.action_flag |= action::move;
			builder_button_t::properties_t bbp;
			bbp.get_matrices() = &pile->stage_maker.fgm.matrices[viewport_area::editor];
			bbp.get_viewport() = &pile->stage_maker.fgm.viewport[viewport_area::editor];
			bbp.position = pile->loco.get_mouse_position(
				pile->stage_maker.fgm.viewport[viewport_area::editor].get_position(),
				pile->stage_maker.fgm.viewport[viewport_area::editor].get_size()
			);

			bbp.size = button_size;
			//bbp.size = button_size;
			bbp.theme = &pile->stage_maker.fgm.theme;
			bbp.text = "button";
			bbp.font_size = scale_object_with_viewport(fan::vec2(0.2), &pile->stage_maker.fgm.viewport[viewport_area::types], &pile->stage_maker.fgm.viewport[viewport_area::editor]).x;
			//bbp.font_size = 0.2;
			pile->stage_maker.fgm.builder_button.push_back(bbp);
			pile->loco.button.set_theme(&instance->cid, loco_t::button_t::inactive);
			auto builder_cid = &pile->stage_maker.fgm.builder_button.instance[pile->stage_maker.fgm.builder_button.instance.size() - 1]->cid;
			auto block = pile->loco.button.sb_get_block(builder_cid);
			pile->loco.vfi.set_focus_mouse(block->p[builder_cid->instance_id].vfi_id);
			pile->loco.vfi.feed_mouse_button(fan::mouse_left, fan::key_state::press);
			pile->stage_maker.fgm.builder_button.open_properties(builder_cid);
			return 0;
		};
		loco.button.push_back(&instance[i]->cid, p);
	}

	void clear() {
		loco_t& loco = *get_loco();
		for (auto& it : instance) {
			loco.button.erase(&it->cid);
			delete it;
		}
		instance.clear();
	}
	struct instance_t {
		fan::opengl::cid_t cid;
		uint16_t shape;
	};
	std::vector<instance_t*> instance;
}editor_button;

struct builder_button_t {
	using properties_t = loco_t::button_t::properties_t;

	loco_t* get_loco() {
		return ((stage_maker_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, builder_button), stage_maker_t, fgm))->get_loco();
	}
	pile_t* get_pile() {
		return OFFSETLESS(get_loco(), pile_t, loco_var_name);
	}

	void open_properties(fan::opengl::cid_t* instance) {
		auto pile = get_pile();

		pile->stage_maker.fgm.menu.clear();

		menu_t::open_properties_t menup;
		menup.matrices = &pile->stage_maker.fgm.matrices[pile_t::stage_maker_t::fgm_t::viewport_area::properties];
		menup.viewport = &pile->stage_maker.fgm.viewport[pile_t::stage_maker_t::fgm_t::viewport_area::properties];
		menup.theme = &pile->stage_maker.fgm.theme;
		menup.position = fan::vec2(0, -0.8);
		menup.gui_size = 0.08;
		auto nr = pile->stage_maker.fgm.menu.push_menu(menup);
		menu_t::properties_t p;
		p.text = "";
		p.text_value = "add cbs";
		p.mouse_button_cb = [this, instance](const loco_t::mouse_button_data_t& mb) -> int {
			use_key_lambda(fan::mouse_left, fan::key_state::release);

			auto pile = get_pile();

			auto stage_name = pile->stage_maker.get_selected_name(
				pile,
				pile->stage_maker.instances[pile_t::stage_maker_t::stage_t::stage_instance].menu_id,
				pile->loco.menu_maker.get_selected_id(pile->stage_maker.instances[pile_t::stage_maker_t::stage_t::stage_instance].menu_id)
			);
			auto file_name = pile->stage_maker.get_file_fullpath(stage_name);

			std::string str;
			fan::io::file::read(file_name, &str);

			std::size_t button_id = -1;
			for (std::size_t i = 0; i < pile->stage_maker.fgm.builder_button.instance.size(); ++i) {
				if (&pile->stage_maker.fgm.builder_button.instance[i]->cid == instance) {
					button_id = i;
					break;
				}
			}

			if (button_id == -1) {
				fan::throw_error("some corruption xd");
			}

			if (str.find(std::to_string(button_id) + "(") != std::string::npos) {
				return 0;
			}

			str += "\n\nstatic int mouse_button_cb" + std::to_string(button_id) + "(const loco_t::mouse_button_data_t& mb){\n  return 0;\n}";

			fan::io::file::write(file_name, str, std::ios_base::binary);

			return 0;
		};
		pile->stage_maker.fgm.menu.push_back(nr, p);
		//
		//pile->stage_maker.fgm.menu.clear();
		//
		//properties_menu_t::properties_t menup;
		//menup.text = "position";
		//menup.text_value = pile->loco.button.get_button(instance, &loco_t::button_t::instance_t::position).to_string();
		//pile->stage_maker.fgm.menu.push_back(menup);
	}

	corners_t get_corners(fan::opengl::cid_t* cid) {
		loco_t& loco = *get_loco();
		fan::vec2 c = loco.button.get_button(cid, &loco_t::button_t::instance_t::position);
		fan::vec2 s = loco.button.get_button(cid, &loco_t::button_t::instance_t::size);
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
	void release() {
		pile_t* pile = get_pile();
		pile->stage_maker.fgm.move_offset = 0;
		pile->stage_maker.fgm.action_flag &= ~action::move;
	}
	void push_back(properties_t& p) {
		p.position.z = 1;
		pile_t* pile = get_pile();
		instance.resize(instance.size() + 1);
		uint32_t i = instance.size() - 1;
		instance[i] = new instance_t;
		instance[i]->shape = shapes::button;
		instance[i]->z = 0;
		instance[i]->text = p.text;
		instance[i]->theme = *pile->loco.get_context()->theme_list[p.theme].theme_id;
		p.mouse_button_cb = [instance = instance[i]](const loco_t::mouse_button_data_t& ii_d) -> int {
			pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);
			if (ii_d.button != fan::mouse_left) {
				return 0;
			}
			if (ii_d.button_state == fan::key_state::release) {
				pile->stage_maker.fgm.builder_button.release();
				// TODO FIX, erases in near bottom
				if (!pile->stage_maker.fgm.viewport[viewport_area::editor].inside(pile->loco.get_mouse_position())) {
					pile->stage_maker.fgm.builder_button.erase(&instance->cid);
				}
				return 0;
			}
			if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
				return 0;
			}
			pile->stage_maker.fgm.action_flag |= action::move;
			auto viewport = pile->loco.button.get_viewport(&instance->cid);
			pile->stage_maker.fgm.click_position = ii_d.position;
			pile->stage_maker.fgm.move_offset = fan::vec2(pile->loco.button.get_button(&instance->cid, &loco_t::button_t::instance_t::position)) - pile->stage_maker.fgm.click_position;
			pile->stage_maker.fgm.resize_offset = pile->stage_maker.fgm.click_position;
			fan::vec3 rp = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::instance_t::position);
			fan::vec3 rs = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::instance_t::size);
			pile->stage_maker.fgm.resize_side = fan_2d::collision::rectangle::get_side_collision(ii_d.position, rp, rs);
			pile->stage_maker.fgm.builder_button.open_properties(&instance->cid);
			return 0;
		};
		p.mouse_move_cb = [instance = instance[i]](const loco_t::mouse_move_data_t& ii_d) -> int {
			if (ii_d.flag->ignore_move_focus_check == false) {
				return 0;
			}
			pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);

			if (instance->holding_special_key) {
				fan::vec3 ps = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::instance_t::position);
				fan::vec3 rs = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::instance_t::size);

				static constexpr f32_t minimum_rectangle_size = 0.03;
				static constexpr fan::vec2i multiplier[] = { {-1, -1}, {1, -1}, {1, 1}, {-1, 1} };

				rs += (ii_d.position - pile->stage_maker.fgm.resize_offset) * multiplier[pile->stage_maker.fgm.resize_side] / 2;

				bool ret = 0;
				auto set_rectangle_size = [&](auto fan::vec3::* c) {
					if (rs.*c < minimum_rectangle_size) {
						rs.*c = minimum_rectangle_size;
						uint8_t offset = (4 - fan::ofof(c)) / 4;
						if (rs[offset] > minimum_rectangle_size) {
							pile->stage_maker.fgm.resize_offset[offset] = ii_d.position[offset];
						}
						ret = 1;
					}
				};

				set_rectangle_size(&fan::vec3::x);
				set_rectangle_size(&fan::vec3::y);

				pile->loco.button.set_size(&instance->cid, rs);

				ps += (ii_d.position - pile->stage_maker.fgm.resize_offset) / 2;
				pile->loco.button.set_position(&instance->cid, ps);

				if (ret) {
					return 0;
				}

				pile->stage_maker.fgm.resize_offset = ii_d.position;
				pile->stage_maker.fgm.move_offset = fan::vec2(pile->loco.button.get_button(&instance->cid, &loco_t::button_t::instance_t::position)) - ii_d.position;
				return 0;
			}

			fan::vec3 p;
			p.x = ii_d.position.x + pile->stage_maker.fgm.move_offset.x;
			p.y = ii_d.position.y + pile->stage_maker.fgm.move_offset.y;
			p.z = instance->z;
			pile->loco.button.set_position(&instance->cid, p);

			return 0;
		};
		p.keyboard_cb = [instance = instance[i]](const loco_t::keyboard_data_t& kd) -> int {
			pile_t* pile = OFFSETLESS(OFFSETLESS(kd.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);

			switch (kd.key) {
			case fan::key_delete: {
				switch (kd.key_state) {
				case fan::key_state::press: {

					switch (instance->shape) {
					case shapes::button: {
						pile->stage_maker.fgm.builder_button.erase(&instance->cid);
						pile->stage_maker.fgm.invalidate_focus();
						break;
					}
					}
					break;
				}
				}
				break;
			}
			case fan::key_c: {
				instance->holding_special_key = kd.key_state == fan::key_state::release ? 0 : 1;
				break;
			}
			}
			return 0;
		};
		pile->loco.button.push_back(&instance[i]->cid, p);
		pile->loco.button.set_theme(&instance[i]->cid, loco_t::button_t::inactive);
		auto builder_cid = &instance[i]->cid;
		auto block = pile->loco.button.sb_get_block(builder_cid);
		pile->loco.vfi.set_focus_mouse(block->p[builder_cid->instance_id].vfi_id);
	}
	void erase(fan::opengl::cid_t* cid) {
		loco_t& loco = *get_loco();
		loco.button.erase(cid);
		for (uint32_t i = 0; i < instance.size(); i++) {
			if (&instance[i]->cid == cid) {
				instance.erase(instance.begin() + i);
				break;
			}
		}
		release();
	}
	void clear() {
		loco_t& loco = *get_loco();
		for (auto& it : instance) {
			loco.button.erase(&it->cid);
			delete it;
		}
		instance.clear();
	}

	struct instance_t {
		fan::opengl::cid_t cid;
		uint16_t shape;
		uint8_t holding_special_key = 0;
		f32_t z;
		std::string text;
		fan_2d::graphics::gui::theme_t theme;
	};

	std::vector<instance_t*> instance;
}builder_button;

struct menu_t {
	
	loco_t* get_loco() {
		return ((stage_maker_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, menu), stage_maker_t, fgm))->get_loco();
	}
	pile_t* get_pile() {
		return OFFSETLESS(get_loco(), pile_t, loco_var_name);
	}

	using properties_t = loco_t::menu_maker_t::properties_t;
	using open_properties_t = loco_t::menu_maker_t::open_properties_t;

	struct instance_t {
		loco_t::menu_maker_t::instance_NodeReference_t nr;
		//std::vector<loco_t::menu_maker_base_t::instance_t> ids;
	};

	loco_t::menu_maker_t::instance_NodeReference_t push_menu(const open_properties_t& op) {
		auto pile = get_pile();
		instance_t in;
		in.nr = pile->loco.menu_maker.push_menu(op);
		instance.push_back(in);
		return in.nr;
	}
	loco_t::menu_maker_base_t::instance_NodeReference_t push_back(loco_t::menu_maker_t::instance_NodeReference_t id, const properties_t& properties) {
		auto pile = get_pile();
		return pile->loco.menu_maker.instances[id].base.push_back(&pile->loco, properties);
	}

	void clear() {
		auto pile = get_pile();
		for (auto& it : instance) {
			pile->loco.menu_maker.erase_menu(it.nr);
		}
		instance.clear();
	}

	std::vector<instance_t> instance;
}menu;