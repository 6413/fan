struct stage_maker_t {

	static constexpr f32_t gui_size = 0.05;

	loco_t* get_loco() {
		return (loco_t*)((uint8_t*)OFFSETLESS(this, pile_t, stage_maker_var_name) + offsetof(pile_t, loco_var_name));
	}

	struct stage_t {
		
		static constexpr uint8_t state_instance = 1;

		struct stage_e {
			static constexpr uint8_t main = 0;
			static constexpr uint8_t gui = 1;
			static constexpr uint8_t function = 2;
		};

		stage_maker_t* get_stage_maker() {
			return OFFSETLESS(this, stage_maker_t, stage);
		}

		struct main_t{
			struct instance_t {
				std::string text;
			};
			fan::hector_vector_t<instance_t> instances;
		}main;
		struct gui_t{
			struct instance_t {

			};
			fan::hector_vector_t<instance_t> instances;
		}gui;
		struct function_t{
			struct instance_t {

			};
			fan::hector_vector_t<instance_t> instances;
		}function;
	}stage;

	void push_stage_main(const loco_t::menu_maker_t::properties_t& p) {
		auto& loco = *get_loco();
		loco.menu_maker.push_back(stage.get_stage_maker()->instances[stage_t::state_instance].menu_id, p);
		stage_t::main_t::instance_t i;
		i.text = p.text;
		stage.main.instances.push_back(i);
	}
	void open_stage_main() {
		auto& loco = *get_loco();

		loco_t::menu_maker_t::properties_t p;

		for (uint32_t i = 0; i < stage.main.instances.size(); i++) {
			p.text = stage.main.instances[i].text;
			loco.menu_maker.push_back(stage.get_stage_maker()->instances[stage_t::state_instance].menu_id, p);
		}
	}
	
	void close_stage() {
		auto& loco = *get_loco();
		loco.menu_maker.erase_menu(instances[stage_t::state_instance].menu_id);
	}

	void open_stage(uint8_t stage_) {
		if (current_stage == stage_) {
			return;
		}
		close_stage();
		current_stage = stage_;

		auto& loco = *get_loco();

		loco_t::menu_maker_t::open_properties_t op;
		op.theme = &theme;
		op.matrices = &matrices;
		op.viewport = &viewport;
		op.gui_size = gui_size * 3;
		op.position = fan::vec2(op.gui_size * (5.0 / 3), -1.0 + op.gui_size);
		instances[stage_t::state_instance].menu_id = loco.menu_maker.push_menu(op);

		switch (current_stage) {
		case stage_t::stage_e::main: {
			open_stage_main();
			break;
		}
		}
	}

	void open() {
		instances.open();
		stage.main.instances.open();
		stage.gui.instances.open();
		stage.function.instances.open();

		current_stage = fan::uninitialized;

		auto& loco = *get_loco();

		loco_t::menu_maker_t::open_properties_t op;
		theme = fan_2d::graphics::gui::themes::deep_red();
		theme.open(loco.get_context());

		fan::vec2 window_size = loco.get_window()->get_size();
		fan::vec2 ratio = window_size / window_size.max();
		fan::graphics::open_matrices(
			loco.get_context(),
			&matrices,
			fan::vec2(-1, 1) * ratio.x,
			fan::vec2(-1, 1) * ratio.y
		);

		viewport.open(loco.get_context());
		viewport.set(loco.get_context(), 0, window_size, window_size);

		op.theme = &theme;
		op.matrices = &matrices;
		op.viewport = &viewport;

		op.gui_size = gui_size;
		op.position = fan::vec2(-1.0 + op.gui_size * 5, -1.0 + op.gui_size * 1);
		instance_t instance;
		instance.menu_id = loco.menu_maker.push_menu(op);
		instances.push_back(instance);
		instances.push_back(instance);
		open_stage(stage_t::stage_e::main);

		loco_t::menu_maker_t::properties_t p;
		p.text = "Create New Stage";
		p.mouse_button_cb = [](const loco_t::mouse_button_data_t& mb) {
			if (mb.button != fan::mouse_left) {
				return;
			}
			if (mb.button_state != fan::key_state::release) {
				return;
			}
			pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco_var_name);
			pile->stage_maker.open_stage(stage_maker_t::stage_t::stage_e::main);
			instance_t* instance = &pile->stage_maker.instances[stage_maker_t::stage_t::state_instance];

			loco_t::menu_maker_t::properties_t p;
			static uint32_t x = 0;
			p.text = std::string("Stage") + fan::to_string(x++);
			//p.mouse_button_cb = [](const loco_t::mouse_button_data_t& mb) {
			//	if (mb.button != fan::mouse_left) {
			//		return;
			//	}
			//	if (mb.button_state != fan::key_state::release) {
			//		return;
			//	}
			//	pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco_var_name);
			//	fan::opengl::cid_t* cid = mb.cid;
			//	if (mb.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
			//		pile->loco.button.set_theme(cid, pile->loco.button.get_theme(cid), loco_t::button_t::press);
			//	}
			//	else {
			//		pile->loco.button.set_theme(cid, pile->loco.button.get_theme(cid), loco_t::button_t::inactive);
			//	}
			//};
			pile->stage_maker.push_stage_main(p);
		};

		loco.menu_maker.push_back(instances[0].menu_id, p);
		p.mouse_button_cb = [](const loco_t::mouse_button_data_t& mb) {
			if (mb.button != fan::mouse_left) {
				return;
			}
			if (mb.button_state != fan::key_state::release) {
				return;
			}

			pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco_var_name);

			pile->stage_maker.open_stage(stage_t::stage_e::gui);
		};
		p.text = "Gui stage";
		loco.menu_maker.push_back(instances[0].menu_id, p);
		p.text = "Function stage";
		loco.menu_maker.push_back(instances[0].menu_id, p);
	}
	void close() {
		auto& loco = *get_loco();
		instances.close();
		stage.main.instances.close();
		stage.gui.instances.close();
		stage.function.instances.close();
		theme.close(loco.get_context());
	}

	struct instance_t {
		loco_t::menu_maker_t::id_t menu_id;
	};
	fan::hector_t<instance_t> instances;

	uint8_t current_stage;

	fan::opengl::matrices_t matrices;
	fan::opengl::viewport_t viewport;
	fan_2d::graphics::gui::theme_t theme;

	struct fgm_t {

	  struct instance_t {
	    f32_t z = 0;
	    fan::opengl::cid_t cid;
	    uint16_t shape;
	    uint8_t holding_special_key = 0;
	  };

	  struct shapes {
	    static constexpr uint16_t line = 0;
	    static constexpr uint16_t button = 1;
	  };

	  struct viewport_area{
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
	  fan::vec2 position_to_coordinates(const fan::vec2& value) {
	    loco_t& loco = *get_loco();
	    fan::vec2 window_size = loco.get_window()->get_size();
	  
	    fan::vec2 ret = value / window_size;
	    ret -= 2.0 / 2;
	    return fan::vec2(0.5, ret.y);
	  }
	  static fan::vec2 scale_object_with_viewport(const fan::vec2& size, fan::opengl::viewport_t* from, fan::opengl::viewport_t* to) {
	    fan::vec2 f = from->get_size();
	    fan::vec2 t = to->get_size();
	    return size / (t / f);
	  }
	  fan::vec2 scale_object_with_viewport(const fan::vec2& size, fan::opengl::viewport_t* from) {
	    fan::vec2 f = from->get_size();
	    fan::vec2 t = get_loco()->get_window()->get_size();
	    return size / (f / t);
	  }
	  fan::vec2 translate_to_global(const fan::vec2& position) const {
	    return position / viewport[viewport_area::global].get_size() * 2 - 1;
	  }
	  fan::vec2 get_viewport_dst(fan::opengl::viewport_t* from, fan::opengl::viewport_t* to) {
	    return (from->get_size() + from->get_position()) / (to->get_size() / 2) - 1;
	  }


	  void invalidate_focus() {
	    loco_t& loco = *get_loco();
	    loco.vfi.invalidate_focus_mouse();
	    loco.vfi.invalidate_focus_keyboard();
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
	    menu.clear();
	    properties_menu_t::properties_t menup;
	    menup.text = "ratio";
	    menup.text_value = editor_ratio.to_string();
	    menu.push_back(menup);
	  }

	  void open() {
	    loco_t& loco = *get_loco();

	    line_y_offset_between_types_and_properties = 0.0;

	    editor_ratio = fan::vec2(1, 1);
	    move_offset = 0;
	    action_flag = 0;
	    theme = fan_2d::graphics::gui::themes::deep_red();
	    theme.open(loco.get_context());

	    static auto resize_cb = [&] () {
	      fan::vec2 window_size = get_loco()->get_window()->get_size();
	      pile_t* pile = OFFSETLESS(OFFSETLESS(this, stage_maker_t, fgm), pile_t, stage_maker);
	      fan::vec2 viewport_size = pile->stage_maker.fgm.translate_viewport_position(fan::vec2(1, 1));
	      pile->stage_maker.fgm.viewport[viewport_area::global].set(
	        pile->loco.get_context(),
	        0,
	        viewport_size,
	        window_size
	      );
	      fan::vec2 ratio = viewport_size / viewport_size.max();
	      pile->stage_maker.fgm.matrices[viewport_area::global].set_ortho(
	        fan::vec2(-1, 1) * ratio.x,
	        fan::vec2(-1, 1) * ratio.y
	      );

	      fan::vec2 viewport_position = pile->stage_maker.fgm.translate_viewport_position(pile->stage_maker.fgm.editor_position - pile->stage_maker.fgm.editor_size);
	      viewport_size = pile->stage_maker.fgm.translate_viewport_position(pile->stage_maker.fgm.editor_size + fan::vec2(-pile->stage_maker.fgm.properties_line_position.x / 2 - 0.1));
	      pile->stage_maker.fgm.viewport[viewport_area::editor].set(
	        pile->loco.get_context(),
	        viewport_position,
	        viewport_size,
	        pile->loco.get_window()->get_size()
	      );
	      ratio = viewport_size / viewport_size.max();
	      pile->stage_maker.fgm.matrices[viewport_area::editor].set_ortho(
	        fan::vec2(-1, 1) * ratio.x,
	        fan::vec2(-1, 1) * ratio.y
	      );

	      viewport_position = pile->stage_maker.fgm.translate_viewport_position(fan::vec2(pile->stage_maker.fgm.properties_line_position.x, -1));
	      viewport_size = pile->stage_maker.fgm.translate_viewport_position(fan::vec2(1, pile->stage_maker.fgm.line_y_offset_between_types_and_properties)) - viewport_position;
	      pile->stage_maker.fgm.viewport[viewport_area::types].set(
	        pile->loco.get_context(),
	        viewport_position,
	        viewport_size,
	        pile->loco.get_window()->get_size()
	      );

	      ratio = viewport_size / viewport_size.max();
	      pile->stage_maker.fgm.matrices[viewport_area::types].set_ortho(
	        fan::vec2(-1, 1) * ratio.x,
	        fan::vec2(-1, 1) * ratio.y
	      );

	      viewport_position.y += pile->stage_maker.fgm.translate_viewport_position(fan::vec2(0, pile->stage_maker.fgm.line_y_offset_between_types_and_properties)).y;
	      pile->stage_maker.fgm.viewport[viewport_area::properties].set(
	        pile->loco.get_context(),
	        viewport_position,
	        viewport_size,
	        pile->loco.get_window()->get_size()
	      );

	      ratio = viewport_size / viewport_size.max();
	      pile->stage_maker.fgm.matrices[viewport_area::properties].set_ortho(
	        fan::vec2(-1, 1) * ratio.x,
	        fan::vec2(-1, 1) * ratio.y
	      );

	      fan::vec3 src, dst;

	      src = pile->stage_maker.fgm.editor_position - pile->stage_maker.fgm.editor_size;
	      dst.x = pile->stage_maker.fgm.editor_position.x + pile->stage_maker.fgm.editor_size.x;
	      dst.y = src.y;

	      src.z = dst.z = 10;
	      pile->loco.line.set_line(
	        &pile->stage_maker.fgm.line.instance[0]->cid,
	        src,
	        dst
	      );

	      src = dst;
	      dst.y = pile->stage_maker.fgm.editor_position.y + pile->stage_maker.fgm.editor_size.y;

	      src.z = dst.z = 10;
	      pile->loco.line.set_line(
	        &pile->stage_maker.fgm.line.instance[1]->cid,
	        src,
	        dst
	      );

	      src = dst;
	      dst.x = pile->stage_maker.fgm.editor_position.x - pile->stage_maker.fgm.editor_size.x;

	      src.z = dst.z = 10;
	      pile->loco.line.set_line(
	        &pile->stage_maker.fgm.line.instance[2]->cid,
	        src,
	        dst
	      );

	      src = dst;
	      dst.y = pile->stage_maker.fgm.editor_position.y - pile->stage_maker.fgm.editor_size.y;

	      src.z = dst.z = 10;
	      pile->loco.line.set_line(
	        &pile->stage_maker.fgm.line.instance[3]->cid,
	        src,
	        dst
	      );

	      src = pile->stage_maker.fgm.translate_to_global(
	        pile->stage_maker.fgm.viewport[viewport_area::types].get_position()
	      );
	      dst.x = src.x;
	      dst.y = pile->stage_maker.fgm.matrices[viewport_area::global].coordinates.bottom;
	      src.z = dst.z = 10;
	      pile->loco.line.set_line(
	        &pile->stage_maker.fgm.line.instance[4]->cid,
	        src,
	        dst
	      );
	      src = pile->stage_maker.fgm.translate_to_global(
	        pile->stage_maker.fgm.viewport[viewport_area::types].get_position() +
	        fan::vec2(0, pile->stage_maker.fgm.viewport[viewport_area::types].get_size().y)
	      );
	      dst = pile->stage_maker.fgm.translate_to_global(
	        pile->stage_maker.fgm.viewport[viewport_area::types].get_position() +
	        pile->stage_maker.fgm.viewport[viewport_area::types].get_size()
	      );
	      src.z = dst.z = 10;
	      pile->loco.line.set_line(
	        &pile->stage_maker.fgm.line.instance[5]->cid,
	        src,
	        dst
	      );
	    };

	    loco.get_window()->add_resize_callback([&](fan::window_t* window, const fan::vec2i& ws) {
	      resize_cb();
	    });

	    // half size
	    properties_line_position = fan::vec2(0.5, 0);
	    editor_position = fan::vec2(-properties_line_position.x / 2, 0);
	    editor_size = editor_position.x + 0.9;

	    matrices[viewport_area::global].open(loco.get_context());
	    matrices[viewport_area::editor].open(loco.get_context());
	    matrices[viewport_area::types].open(loco.get_context());
	    matrices[viewport_area::properties].open(loco.get_context());

	    viewport[viewport_area::global].open(loco.get_context());
	    viewport[viewport_area::editor].open(loco.get_context());
	    viewport[viewport_area::types].open(loco.get_context());
	    viewport[viewport_area::properties].open(loco.get_context());

	    loco_t::vfi_t::properties_t p;
	    p.shape_type = loco_t::vfi_t::shape_t::always;
	    p.shape.always.z = 0;
	    p.mouse_button_cb = [](const loco_t::vfi_t::mouse_button_data_t& mb) {
	      pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco);
	      pile->stage_maker.fgm.invalidate_focus();
	      pile->stage_maker.fgm.open_editor_properties();
	    };
	    loco.vfi.push_shape(p);

	    line.open();
	    editor_button.open();
	    builder_button.open();
	    menu.open(fan::vec2(1.05, button_size.y * 1.5));

	    line_t::properties_t lp;
	    lp.get_viewport() = &viewport[viewport_area::global];
	    lp.get_matrices() = &matrices[viewport_area::global];
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
	  
	    editor_button_t::properties_t ebp;
	    ebp.get_matrices() = &matrices[viewport_area::types];
	    ebp.get_viewport()  = &viewport[viewport_area::types];
	    ebp.position = fan::vec2(0, matrices[viewport_area::types].coordinates.top * 0.9);
	    ebp.size = button_size;
	    ebp.theme = &theme;
	    ebp.text = "button";
	    editor_button.push_back(ebp);

	    open_editor_properties();
	  }
	  void close() {
	    line.close();
	    editor_button.close();
	    builder_button.close();
	  }

	  struct line_t {
	    using properties_t = loco_t::line_t::properties_t;

	    loco_t* get_loco() {
	      return ((stage_maker_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, line), stage_maker_t, fgm))->get_loco();
	    }
	    pile_t* get_pile() {
	      return OFFSETLESS(get_loco(), pile_t, loco_var_name);
	    }

	    void open() {
	      instance.open();
	    }
	    void close() {
	      instance.close();
	    }
	    void push_back(properties_t& p) {
	      loco_t& loco = *get_loco();
	      uint32_t i = instance.resize(instance.size() + 1);
	      instance[i] = new instance_t;
	      instance[i]->shape = shapes::line;
	      loco.line.push_back(&instance[i]->cid, p);
	    }
	    fan::hector_t<instance_t*> instance;
	  }line;

	  struct editor_button_t {
	    using properties_t = loco_t::button_t::properties_t;

	    loco_t* get_loco() {
				return ((stage_maker_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, editor_button), stage_maker_t, fgm))->get_loco();
	    }
	    pile_t* get_pile() {
	      return OFFSETLESS(get_loco(), pile_t, loco_var_name);
	    }

	    void open() {
	      instance.open();
	    }
	    void close() {
	      instance.close();
	    }
	    void push_back(properties_t& p) {
	      p.position.z = 1;
	      loco_t& loco = *get_loco();
	      uint32_t i = instance.resize(instance.size() + 1);
	      instance[i] = new instance_t;
	      instance[i]->shape = shapes::button;
	      p.mouse_button_cb = [instance = instance[i]](const loco_t::mouse_button_data_t& ii_d) -> void {
	        pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi), pile_t, loco);
	        if (ii_d.button != fan::mouse_left) {
	          return;
	        }
	        if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
	          return;
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
	        return;
	      };
	      loco.button.push_back(&instance[i]->cid, p);
	    }

	    fan::hector_t<instance_t*> instance;
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

	      properties_menu_t::properties_t menup;
	      menup.text = "position";
	      menup.text_value = pile->loco.button.get_button(instance, &loco_t::button_t::instance_t::position).to_string();
	      pile->stage_maker.fgm.menu.push_back(menup);
	    }

	    void open() {
	      instance.open();
	    }
	    void close() {
	      instance.close();
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
	      loco_t& loco = *get_loco();
	      pile_t* pile = get_pile();
	      uint32_t i = instance.resize(instance.size() + 1);
	      instance[i] = new instance_t;
	      instance[i]->shape = shapes::button;
	      instance[i]->z = 0;
	      p.mouse_button_cb = [instance = instance[i]](const loco_t::mouse_button_data_t& ii_d) -> void {
	        pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);
	        if (ii_d.button != fan::mouse_left) {
	          return;
	        }
	        if (ii_d.button_state == fan::key_state::release) {
	          pile->stage_maker.fgm.builder_button.release();
	          if (!pile->stage_maker.fgm.viewport[viewport_area::editor].inside(pile->loco.get_mouse_position())) {
	            pile->stage_maker.fgm.builder_button.erase(&instance->cid);
	          }
	          return;
	        }
	        if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
	          return;
	        }
	        pile->stage_maker.fgm.action_flag |= action::move;
	        auto viewport = pile->loco.button.get_viewport(&instance->cid);
	        pile->stage_maker.fgm.click_position = ii_d.position;
	        pile->stage_maker.fgm.move_offset =  fan::vec2(pile->loco.button.get_button(&instance->cid, &loco_t::button_t::instance_t::position)) - pile->stage_maker.fgm.click_position;
	        pile->stage_maker.fgm.resize_offset = pile->stage_maker.fgm.click_position;
	        fan::vec3 rp = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::instance_t::position);
	        fan::vec3 rs = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::instance_t::size);
	        pile->stage_maker.fgm.resize_side = fan_2d::collision::rectangle::get_side_collision(ii_d.position, rp, rs);
	        return;
	      };
	      p.mouse_move_cb = [instance = instance[i]](const loco_t::mouse_move_data_t& ii_d) -> void {
	        if (ii_d.flag->ignore_move_focus_check == false) {
	          return;
	        }
	        pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);

	        if (instance->holding_special_key) {
	          fan::vec3 ps = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::instance_t::position);
	          fan::vec3 rs = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::instance_t::size);

	          static constexpr f32_t minimum_rectangle_size = 0.03;
	          static constexpr fan::vec2i multiplier[] = { {-1, -1}, {1, -1}, {1, 1}, {-1, 1} };

	          rs += (ii_d.position - pile->stage_maker.fgm.resize_offset) * multiplier[pile->stage_maker.fgm.resize_side] / 2;

	          bool ret = 0;
	          auto set_rectangle_size = [&](auto fan::vec3::*c) {
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
	            return;
	          }

	          pile->stage_maker.fgm.resize_offset = ii_d.position;
	          pile->stage_maker.fgm.move_offset = fan::vec2(pile->loco.button.get_button(&instance->cid, &loco_t::button_t::instance_t::position)) - ii_d.position;
	          return;
	        }

	        fan::vec3 p;
	        p.x = ii_d.position.x + pile->stage_maker.fgm.move_offset.x;
	        p.y = ii_d.position.y + pile->stage_maker.fgm.move_offset.y;
	        p.z = instance->z;
	        pile->loco.button.set_position(&instance->cid, p);
	      };
	      p.keyboard_cb = [instance = instance[i]](const loco_t::keyboard_data_t& kd) -> void {
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
	      };
	      loco.button.push_back(&instance[i]->cid, p);
	      pile->loco.button.set_theme(&instance[i]->cid, loco_t::button_t::inactive);
	      auto builder_cid = &instance[i]->cid;
	      auto block = pile->loco.button.sb_get_block(builder_cid);
	      pile->loco.vfi.set_focus_mouse(block->p[builder_cid->instance_id].vfi_id);
	    }
	    void erase(fan::opengl::cid_t* cid) {
	      loco_t& loco = *get_loco();
	      loco.button.erase(cid);
	      release();
	    }
	    fan::hector_t<instance_t*> instance;
	  }builder_button;

	  struct properties_menu_t {

	    struct properties_t {
	      std::string text_value;
	      std::string text;
	    };

	    loco_t* get_loco() {
				return ((stage_maker_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, menu), stage_maker_t, fgm))->get_loco();
	    }
	    pile_t* get_pile() {
	      return OFFSETLESS(get_loco(), pile_t, loco_var_name);
	    }

	    void open(const fan::vec2& off) {
	      instance.open();
	      offset = off;
	      o_offset = off;
	    }
	    void close() {
	      instance.close();
	    }

	    void push_back(properties_t mp) {
	      loco_t::button_t::properties_t p;
	      pile_t* pile = get_pile();
	      p.size = pile->stage_maker.fgm.button_size;
	      offset += p.size.y;
	      p.position = fan::vec2(-1, -1) + offset;
	      p.position.z = 1;
	      p.theme = &pile->stage_maker.fgm.theme;
	      p.get_matrices() = &pile->stage_maker.fgm.matrices[viewport_area::properties];
	      p.get_viewport() = &pile->stage_maker.fgm.viewport[viewport_area::properties];
	      p.text = mp.text_value;
	     // p.disable_highlight = true;
	      uint32_t i = instance.resize(instance.size() + 1);
	      instance[i] = new i_t;
	      instance[i]->shape = shapes::button;
	      pile->loco.button.push_back(&instance[i]->cid, p);
	      loco_t::text_t::properties_t tp;
	      tp.text = mp.text;
	      fan::vec2 text_size = pile->loco.text.get_text_size(tp.text, p.font_size);
	      tp.position = p.position - fan::vec3(text_size.x / 1.5 + p.size.x, 0, 0);
	      tp.font_size = p.font_size;
	      tp.get_matrices() = p.get_matrices();
	      tp.get_viewport() = p.get_viewport();
	      instance[i]->text_id = pile->loco.text.push_back(tp);
	    }

	    void clear() {
	      auto& loco = *get_loco();
	      for (uint32_t i = 0; i < instance.size(); i++) {
	        loco.text.erase(instance[i]->text_id);
	        loco.button.erase(&instance[i]->cid);
	        delete instance[i];
	      }
	      instance.clear();
	      offset = o_offset;
	    }

	    struct i_t : instance_t {
	      uint32_t text_id;
	    };

	    fan::vec2 offset;
	    fan::vec2 o_offset;
	    fan::hector_t<i_t*> instance;
	  }menu;

	  fan::opengl::matrices_t matrices[4];
	  fan::opengl::viewport_t viewport[4];

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
	}fgm;
};