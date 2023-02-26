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

struct fgm_t {

	#if defined(fgm_build_stage_maker)
		stage_maker_t* get_stage_maker() {
			return OFFSETLESS(this, stage_maker_t, fgm);
		}
	#endif

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

  #include "common.h"

	static constexpr fan::vec2 button_size = fan::vec2(0.3, 0.08);
  static constexpr f32_t line_z_depth = 50;
  static constexpr f32_t right_click_z_depth = 11;

	f32_t line_y_offset_between_types_and_properties;

	// for -1 - 1 matrix
	fan::vec2 translate_viewport_position(const fan::vec2& value) {
		fan::vec2 window_size = pile->loco.get_window()->get_size();
		return (value + 1) / 2 * window_size;
	}
	fan::vec2 translate_viewport_position_to_coordinate(fan::graphics::viewport_t* to) {
		fan::vec2 window_size = pile->loco.get_window()->get_size();
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
		fan::vec2 t = pile->loco.get_window()->get_size();
		return size / (f / t);
	}
	fan::vec2 translate_to_global(const fan::vec2& position) const {
		return position / viewport[viewport_area::global].get_size() * 2 - 1;
	}
	fan::vec2 get_viewport_dst(fan::graphics::viewport_t* from, fan::graphics::viewport_t* to) {
		return (from->get_size() + from->get_position()) / (to->get_size() / 2) - 1;
	}

  void invalidate_right_click_menu() {
    if (pile->loco.menu_maker_button.instances.inric(right_click_menu_nr)) {
      return;
    }
    auto v = pile->loco.menu_maker_button.instances.gnric();
    pile->loco.menu_maker_button.erase_menu(right_click_menu_nr);
    right_click_menu_nr = v;
    pile->loco.vfi.erase(&vfi_id);
  }

	void invalidate_focus() {
    pile->loco.vfi.invalidate_focus_mouse();
    pile->loco.vfi.invalidate_focus_keyboard();
    invalidate_right_click_menu();
	}

	#if defined(fgm_build_stage_maker)
  static void write_stage_functions(
    fgm_t* fgm,
    auto* shape,
    const fan::string& file_name, 
    const fan::string& stage_name,
    const fan::string& shape_name,
    const auto& cb_names
  ) {
    fan::string str;

    const fan::string advance_str = fan::format("struct {0}_t", stage_name);
    auto advance_position = fgm->get_stage_maker()->stage_h_str.find(
      advance_str
    );

    if (advance_position == fan::string::npos) {
      fan::throw_error("corrupted stage.h:advance_position");
    }

    auto src_str = fan::format("//{0}_src\n",
      shape_name      
    );
    auto src = fgm->get_stage_maker()->stage_h_str.find(
      src_str,
      advance_position
    );
    if (src == fan::string::npos) {
      fan::throw_error("corrupted stage.h:src");
    }
    src += src_str.size();
    auto dst = fgm->get_stage_maker()->stage_h_str.find(
      fan::format("\n    //{0}_dst",
        shape_name
      ),
      advance_position
    );

    if (dst == fan::string::npos) {
      fan::throw_error("corrupted stage.h:dst");
    }

		fgm->get_stage_maker()->stage_h_str.erase(src, dst - src);

    fan::io::file::read(file_name, &str);

    fan::string format;

    for (uint32_t k = 0; k < std::size(cb_names); ++k) {
      format += fan::format(R"(    {3}_{0}_cb_table_t {3}_{1}_cb_table[{2}] = {{)", cb_names[k], cb_names[k], shape->instances.size(), shape_name);

      for (std::size_t j = 0; j < shape->instances.size(); ++j) {
        format += fan::format("&{0}_t::{3}{1}_{2}_cb,", stage_name.c_str(), shape->instances[j]->id, cb_names[k], shape_name);
      }

      format += "};";
      if (k + 1 != std::size(cb_names)) {
        format += "\n";
      }
    }

		fgm->get_stage_maker()->stage_h_str.insert(src, format);

    for (std::size_t j = 0; j < shape->instances.size(); ++j) {
      for (uint32_t k = 0; k < std::size(cb_names); ++k) {
        auto cbs_text = fan::format(R"(
int {2}{0}_{1}_cb(const loco_t::{1}_data_t& mb){{
  return 0;
}}
)", fan::to_string(shape->instances[j]->id), cb_names[k], shape_name);
        if (str.find(cbs_text) != fan::string::npos) {
          continue;
        }
        str += cbs_text;
      }
    }

    fan::io::file::write(file_name, str, std::ios_base::binary);

		fgm->get_stage_maker()->write_stage();

  }
	#endif

	void open_editor_properties() {
	/*	button_menu.clear();

		button_menu_t::open_properties_t menup;
		menup.matrices = &matrices[viewport_area::properties];
		menup.viewport = &viewport[viewport_area::properties];
		menup.theme = &theme;
		menup.position = fan::vec2(0, -0.8);
		auto nr = button_menu.push_menu(menup);
		button_menu_t::properties_t p;
		p.text = L"ratio";
		p.text_value = L"1, 1";
		button_menu.push_back(nr, p);*/
	}

	#include "resize_cb.h"

	void load() {

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

		button_menu_t::open_properties_t op;
		op.matrices = &matrices[viewport_area::types];
		op.viewport = &viewport[viewport_area::types];
		op.theme = &theme;
		op.position = fan::vec2(0, -0.9);
		op.gui_size = 0.08;

		right_click_menu_nr = pile->loco.menu_maker_button.instances.gnric();
		static auto push_menu = [this](
      auto mb,
      loco_t::menu_maker_button_t::properties_t p
      ) {
        pile->loco.menu_maker_button.push_back(right_click_menu_nr, p);
		};

		loco_t::vfi_t::properties_t vfip;
		vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
		vfip.shape.rectangle.position = translate_viewport_position_to_coordinate(&viewport[viewport_area::types]);
		vfip.shape.rectangle.position.z = right_click_z_depth;
		vfip.shape.rectangle.matrices = &matrices[viewport_area::global];
		vfip.shape.rectangle.viewport = &viewport[viewport_area::global];
		vfip.shape.rectangle.size = viewport[viewport_area::types].get_size() / pile->loco.get_window()->get_size();

    vfip.mouse_button_cb = [this](const loco_t::vfi_t::mouse_button_data_t& mb) mutable -> int {
      loco_t::menu_maker_button_t::open_properties_t rcm_op;
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
			if (mb.button_state != fan::mouse_state::release) {
				invalidate_right_click_menu();
				return 0;
			}
			if (pile->loco.menu_maker_button.instances.inric(right_click_menu_nr)) {
				rcm_op.position = mb.position + pile->loco.menu_maker_button.get_button_measurements(rcm_op.gui_size);
				rcm_op.position.z = right_click_z_depth;
				right_click_menu_nr = pile->loco.menu_maker_button.push_menu(rcm_op);
        #if defined(fgm_button)
        push_menu(mb, {
          .text = "button",
          .mouse_button_cb = [this](const loco_t::mouse_button_data_t& ii_d) -> int {
			      pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi), pile_t, loco);
			      if (ii_d.button != fan::mouse_left) {
				      return 0;
			      }
            if (ii_d.button_state != fan::mouse_state::release) {
              return 0;
            }
			      if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
				      return 0;
			      }
			      button_t::properties_t bbp;
			      bbp.matrices = &matrices[viewport_area::editor];
			      bbp.viewport = &viewport[viewport_area::editor];
			      bbp.position = fan::vec3(0, 0, 20);
			
			      bbp.size = button_size;
			      //bbp.size = button_size;
			      bbp.theme = &theme;
			      bbp.text = "button";
			      bbp.font_size = scale_object_with_viewport(fan::vec2(0.2), &viewport[viewport_area::types], &viewport[viewport_area::editor]).x;
			      button.push_back(bbp);
			
            auto it = button.instances[button.instances.size() - 1];
			      auto builder_cid = &it->cid;
			      auto ri = pile->loco.button.get_ri(builder_cid);
			      //pile->loco.vfi.set_focus_mouse(ri.vfi_id);
			      //pile->loco.vfi.feed_mouse_button(fan::button_left, fan::button_state::press);
			      button.open_properties(it);
			
						#if defined(fgm_build_stage_maker)
							auto stage_name = get_stage_maker()->get_selected_name(
								get_stage_maker()->instances[stage_maker_t::stage_t::stage_instance].menu_id
							);
							auto file_name = get_stage_maker()->get_file_fullpath(stage_name);

							write_stage_functions(this, &button, file_name, stage_name, "button", button_t::cb_names);
						#endif

            invalidate_right_click_menu();

			      return 1;
		      }
        });
        #endif
        #if defined(fgm_sprite)
        push_menu(mb, {
          .text = "sprite",
          .mouse_button_cb = [this](const loco_t::mouse_button_data_t& ii_d) -> int {
            pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi), pile_t, loco);
            if (ii_d.button != fan::mouse_left) {
              return 0;
            }
            if (ii_d.button_state != fan::mouse_state::release) {
              return 0;
            }
            if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
              return 0;
            }
          	sprite_t::properties_t sp;
          	sp.matrices = &matrices[viewport_area::editor];
          	sp.viewport = &viewport[viewport_area::editor];
          	sp.position = fan::vec3(0, 0, 20);

            fan::vec2 size = fan::vec2(
              matrices[viewport_area::editor].coordinates.right,
              matrices[viewport_area::editor].coordinates.down
            );

          	sp.size = size / 10;
          	sp.image = &pile->loco.default_texture;
          	sprite.push_back(sp);
          	auto& instance = sprite.instances[sprite.instances.size() - 1];
          	sprite.open_properties(instance);

            invalidate_right_click_menu();

          	return 1;
          }
        });
        #endif
        #if defined(fgm_text)
        push_menu(mb, {
           .text = "text",
          .mouse_button_cb = [this](const loco_t::mouse_button_data_t& ii_d) -> int {
            pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi), pile_t, loco);
            if (ii_d.button != fan::mouse_left) {
              return 0;
            }
            if (ii_d.button_state != fan::mouse_state::release) {
              return 0;
            }
            if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
              return 0;
            }
            text_t::properties_t sp;
            sp.matrices = &matrices[viewport_area::editor];
            sp.viewport = &viewport[viewport_area::editor];
            sp.position = fan::vec3(0, 0, 20);

            sp.font_size = 0.1;
            sp.text = "text";
            text.push_back(sp);
            auto& instance = text.instances[text.instances.size() - 1];
            text.open_properties(instance);

            invalidate_right_click_menu();

            return 1;
          }
        });
        #endif
        #if defined(fgm_hitbox)
        push_menu(mb, {
          .text = "hitbox",
          .mouse_button_cb = [this](const loco_t::mouse_button_data_t& ii_d) -> int {
            pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi), pile_t, loco);
            if (ii_d.button != fan::mouse_left) {
              return 0;
            }
            if (ii_d.button_state != fan::mouse_state::release) {
              return 0;
            }
            if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
              return 0;
            }
          	hitbox_t::properties_t sp;
          	sp.matrices = &matrices[viewport_area::editor];
          	sp.viewport = &viewport[viewport_area::editor];
          	sp.position = fan::vec3(0, 0, 20);
            sp.vfi_type = loco_t::vfi_t::shape_t::always;

          	sp.size = fan::vec2(0.1, 0.1);
          	sp.image = &hitbox_image;
          	hitbox.push_back(sp);
          	auto& instance = hitbox.instances[hitbox.instances.size() - 1];
          	hitbox.open_properties(instance);

						#if defined(fgm_build_stage_maker)
							auto stage_name = get_stage_maker()->get_selected_name(
								get_stage_maker()->instances[stage_maker_t::stage_t::stage_instance].menu_id
							);
							auto file_name = get_stage_maker()->get_file_fullpath(stage_name);

							write_stage_functions(this, &hitbox, file_name, stage_name, "hitbox", hitbox_t::cb_names);
						#endif

            invalidate_right_click_menu();

          	return 1;
          }
        });
        #endif

        #if defined(fgm_mark)
        push_menu(mb, {
          .text = "mark",
          .mouse_button_cb = [this](const loco_t::mouse_button_data_t& ii_d) -> int {
            pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi), pile_t, loco);
            if (ii_d.button != fan::mouse_left) {
              return 0;
            }
            if (ii_d.button_state != fan::mouse_state::release) {
              return 0;
            }
            if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
              return 0;
            }
            mark_t::properties_t sp;
            sp.matrices = &matrices[viewport_area::editor];
            sp.viewport = &viewport[viewport_area::editor];
            sp.position = fan::vec3(0, 0, 20);
            sp.vfi_type = loco_t::vfi_t::shape_t::always;

            sp.size = fan::vec2(0.1, 0.1);
            sp.image = &mark_image;
            mark.push_back(sp);
            auto& instance = mark.instances[mark.instances.size() - 1];
            mark.open_properties(instance);

            #if defined(fgm_build_stage_maker)
              auto stage_name = get_stage_maker()->get_selected_name(
                get_stage_maker()->instances[stage_maker_t::stage_t::stage_instance].menu_id
              );
              auto file_name = get_stage_maker()->get_file_fullpath(stage_name);

              write_stage_functions(this, &hitbox, file_name, stage_name, "hitbox", hitbox_t::cb_names);
            #endif

            invalidate_right_click_menu();

            return 1;
          }
          });
        #endif

        loco_t::vfi_t::properties_t p;
		    p.shape_type = loco_t::vfi_t::shape_t::always;
		    p.shape.always.z = rcm_op.position.z;
		    p.mouse_button_cb = [this](const loco_t::vfi_t::mouse_button_data_t& mb) -> int {
          invalidate_right_click_menu();
			    return 1;
		    };
		    pile->loco.vfi.push_back(&vfi_id, p);
			}

			return 0;
		};
    loco_t::vfi_t::shape_id_t shape_id;
    pile->loco.push_back_input_hitbox(&shape_id, vfip);

		#if defined(fgm_build_stage_maker)

		fan::string stage_name = get_stage_maker()->get_selected_name(
			get_stage_maker()->instances[stage_t::stage_instance].menu_id
		);

		global_button_t::properties_t gbp;
		gbp.matrices = &matrices[viewport_area::global];
		gbp.viewport = &viewport[viewport_area::global];
		gbp.position = fan::vec2(-0.8, matrices[viewport_area::types].coordinates.up * 0.9);
		gbp.size = button_size / fan::vec2(4, 2);
		gbp.theme = &theme;
		gbp.text = "<-";
		gbp.mouse_button_cb = [this](const loco_t::mouse_button_data_t& mb) -> int {
			use_key_lambda(fan::mouse_left, fan::mouse_state::release);

				stage_maker_t* stage_maker = get_stage_maker();
				get_stage_maker()->reopen_from_fgm();
				write_to_file(stage_maker->get_selected_name(
					get_stage_maker()->instances[stage_maker_t::stage_t::stage_instance].menu_id
				));

			clear();

			return 1;
		};
		global_button.push_back(gbp);

		#endif

		open_editor_properties();

		#if defined(fgm_build_stage_maker)
			read_from_file(stage_name);
		#endif

    loco_t::menu_maker_text_box_t::open_properties_t rcm_op;
    rcm_op.viewport = &viewport[viewport_area::global];
    rcm_op.matrices = &matrices[viewport_area::global];
    rcm_op.theme = &theme;
    rcm_op.gui_size = 0.08;

    rcm_op.position = fan::vec2(-.9, .8) + pile->loco.menu_maker_button.get_button_measurements(rcm_op.gui_size);
    rcm_op.position.z = right_click_z_depth;
    global_menu_nr = pile->loco.menu_maker_text_box.push_menu(rcm_op);
    global_menu_ids.push_back(pile->loco.menu_maker_text_box.push_back(global_menu_nr, {
      .text = "-1, 1, -1, 1",
      .keyboard_cb = [this](const loco_t::keyboard_data_t& d) -> int {
        if (d.key != fan::key_enter) {
          return 0;
        }
        if (d.keyboard_state != fan::keyboard_state::press) {
          return 0;
        }

        auto& it = pile->loco.menu_maker_text_box.instances[global_menu_nr].base.instances[global_menu_ids[0]];
        auto text = pile->loco.text_box.get_text(&it.cid);

        fan::vec4 size = fan::string_to<fan::vec4>(text);
        matrices[viewport_area::editor].set_ortho(
          &pile->loco,
          fan::vec2(size[0], size[1]),
          fan::vec2(size[2], size[3])
        );
      }
    }));
	}

	void open(const char* texturepack_name) {
    right_click_menu_nr = pile->loco.menu_maker_button.instances.gnric();

		line_y_offset_between_types_and_properties = 0.0;

		editor_ratio = fan::vec2(1, 1);
		move_offset = 0;
		action_flag = 0;
		theme = loco_t::themes::deep_red();
		theme.open(pile->loco.get_context());

		texturepack.open_compiled(&pile->loco, texturepack_name);

    static constexpr fan::color image[3 * 3] = {
      fan::color(0, 1, 0, 0.3),
      fan::color(0, 1, 0, 0.3),
      fan::color(0, 1, 0, 0.3),
      fan::color(0, 1, 0, 0.3),
      fan::color(0, 0, 0, 0.3),
      fan::color(0, 1, 0, 0.3),
      fan::color(0, 1, 0, 0.3),
      fan::color(0, 1, 0, 0.3),
      fan::color(0, 1, 0, 0.3),
    };

    static constexpr fan::color mark_image_pixels[2] = {
      fan::color(1, 1, 1, 1),
      fan::color(1, 1, 1, 1)
    };

    hitbox_image.load(&pile->loco, (fan::color*)image, 3);
    mark_image.load(&pile->loco, (fan::color*)mark_image_pixels, 1);

		pile->loco.get_window()->add_resize_callback([this](const fan::window_t::resize_cb_data_t& d) {
			resize_cb();
		});
		pile->loco.get_window()->add_mouse_move_callback([this](const auto& d) {
      if (view_action_flag & action::move) {
        fan::vec2 size = fan::vec2(
          matrices[viewport_area::editor].coordinates.right,
          matrices[viewport_area::editor].coordinates.down
        );
        *(fan::vec2*)& camera_position -= 
          ((pile->loco.get_mouse_position() - pile->loco.get_window()->get_previous_mouse_position()) * 
          (size / pile->loco.get_window()->get_size())) * 32;
        matrices[viewport_area::editor].set_camera_position(camera_position);
      }
    });
		pile->loco.get_window()->add_buttons_callback([this](const auto& d) {
      switch (d.button) {
        case fan::mouse_middle: {
          if (d.state == fan::mouse_state::press) {
            view_action_flag |= action::move;
          }
          else {
            view_action_flag &= ~action::move;
          }
          break;
        }
      }
    });

		pile->loco.get_window()->add_keys_callback([this](const auto& d) {
      switch (d.key) {
        case fan::key_f: {
          if (d.state != fan::keyboard_state::press) {
            return;
          }
          camera_position = fan::vec3(0, 0, 0);
          matrices[viewport_area::editor].set_camera_position(camera_position);
          break;
        }
      }
    });

    //camera_position = fan::vec3(-512, 0, 0);
    //matrices[viewport_area::editor].set_camera_position(camera_position);
		// half size
		properties_line_position = fan::vec2(0.5, 0);
		editor_position = fan::vec2(-properties_line_position.x / 2, 0);
		editor_size = editor_position.x + 0.9;

		matrices[viewport_area::global].open(&pile->loco);
		matrices[viewport_area::editor].open(&pile->loco);
		matrices[viewport_area::types].open(&pile->loco);
		matrices[viewport_area::properties].open(&pile->loco);

		viewport[viewport_area::global].open(pile->loco.get_context());
		viewport[viewport_area::editor].open(pile->loco.get_context());
		viewport[viewport_area::types].open(pile->loco.get_context());
		viewport[viewport_area::properties].open(pile->loco.get_context());


	}
	void close() {
		clear();
	}
	void clear() {
		line.clear();
		global_button.clear();
    #if defined(fgm_button)
		button.clear();
    #endif
    #if defined(fgm_sprite)
		sprite.clear();
    #endif
    #if defined(fgm_text)
    text.clear();
    #endif
    #if defined(fgm_hitbox)
    hitbox.clear();
    #endif
		button_menu.clear();

		pile->loco.menu_maker_text_box.erase_menu(global_menu_nr);
	}

	fan::string get_fgm_full_path(const fan::string& stage_name) {
		#if defined(fgm_build_stage_maker)
		return fan::string(stage_maker_t::stage_runtime_folder_name) + "/" + stage_name + ".fgm";
		#else
		fan::throw_error("");
		return "";
		#endif
	}
	#if defined(fgm_build_stage_maker)
  void read_from_file(const fan::string& stage_name) {
    fan::string path = get_fgm_full_path(stage_name);
    fan::string f;
    if (!fan::io::file::exists(path)) {
      return;
    }
    fan::io::file::read(path, &f);

    if (f.empty()) {
      return;
    }
    uint64_t off = 0;

    uint32_t file_version = fan::read_data<uint32_t>(f, off);

    switch (file_version) {
      // no need to put other versions than current because it would not compile
      case stage_maker_format_version: {
        while (off < f.size()) {
          iterate_masterpiece([&f, &off](auto& o) {
            off += o.from_string(f.substr(off));
          });
        }
        break;
      }
      default: {
        fan::throw_error("invalid version fgm version number", file_version);
        break;
      }
    }
  }
	void write_to_file(const fan::string& stage_name) {

		fan::string f;
    // header
    fan::write_to_string(f, stage_maker_format_version);

    iterate_masterpiece([&f](const auto& shape) {
      f += shape.to_string();
    });
   
		fan::io::file::write(
			get_fgm_full_path(stage_name),
			f,
			std::ios_base::binary
		);

    auto offset = get_stage_maker()->stage_h_str.find(stage_name);

    if (offset == fan::string::npos) {
      fan::throw_error("corrupted stage.h");
    }
	}
	#endif

  #include "private.h"
	#include "shapes.h"

	loco_t::matrices_t matrices[4];
	fan::graphics::viewport_t viewport[4];

	loco_t::theme_t theme;

	uint32_t action_flag;
  uint32_t view_action_flag = 0;

  fan::vec3 camera_position = 0;

	fan::vec2 click_position;
	fan::vec2 move_offset;
	fan::vec2 resize_offset;
	uint8_t resize_side;

	fan::vec2 properties_line_position;

	fan::vec2 editor_position;
	fan::vec2 editor_size;
	fan::vec2 editor_ratio;

	loco_t::texturepack_t texturepack;

	loco_t::menu_maker_button_t::nr_t right_click_menu_nr;
  loco_t::menu_maker_text_box_t::nr_t global_menu_nr;
  std::vector< loco_t::menu_maker_text_box_t::id_t> global_menu_ids;
  loco_t::vfi_t::shape_id_t vfi_id;

  loco_t::image_t hitbox_image;
  loco_t::image_t mark_image;
};

#undef use_key_lambda

#undef fgm_button
#undef fgm_sprite
#undef fgm_text
#undef fgm_hitbox
#undef fgm_mark