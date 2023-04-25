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
		static constexpr uint32_t sidepanel = 2;
	};

	struct action {
		static constexpr uint32_t move = 1 << 0;
		static constexpr uint32_t resize = 1 << 1;
	};

  #include "common.h"

  static constexpr f32_t gui_size = 0.1;

  static constexpr f32_t line_z_depth = 50;
  static constexpr f32_t right_click_z_depth = 11;

  static inline f32_t z_depth = 1;

  struct interact_hitbox_t {
    loco_t::vfi_t::shape_id_t vfi_id;
    interact_hitbox_t() {
      vfi_id.sic();
    }
    ~interact_hitbox_t() {
      if (vfi_id.iic()) {
        return;
      }
      gloco->vfi.erase(&vfi_id);
      vfi_id.sic();
    }
    void create_hitbox(fgm_t* fgm, auto* shape, const fan::vec3& position) {
      loco_t::vfi_t::properties_t vfip;
      vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
      vfip.shape.rectangle.viewport = &fgm->viewports[viewport_area::editor];
      vfip.shape.rectangle.camera = &fgm->cameras[viewport_area::editor];
      vfip.shape.rectangle.position = position;
      vfip.shape.rectangle.size = shape->get_size();
      // shape pointer might die
      vfip.mouse_button_cb = [fgm](const auto& d) -> int{
        if (fgm->editor_mode != fgm_t::editor_modes_e::mod) {
          return 0;
        }

        if (d.button_state == fan::mouse_state::press) {
          fgm->shape_mode = fgm_t::shape_mode_e::move;
        }
        else if (d.button_state == fan::mouse_state::release) {
          fgm->shape_mode = fgm_t::shape_mode_e::idle;
        }
        return 0;
      };
      // shape pointer might die
      vfip.mouse_move_cb = [fgm, cid_nt = *(loco_t::cid_nt_t*)shape](const auto& d) -> int {
        switch (fgm->shape_mode) {
          case fgm_t::shape_mode_e::move: {
            gloco->vfi.shape_list[gloco->vfi.focus.mouse].shape_data.shape.rectangle.position = d.position;
            loco_t::shape_t* shape = (loco_t::shape_t*)&cid_nt;
            shape->set_position(d.position);
            break;
          }
          case fgm_t::shape_mode_e::resize: {
            // todo
            break;
          }
        }
        return 0;
      };
      gloco->vfi.push_back(&vfi_id, vfip);
    }
  };

  struct sprite_t : loco_t::shape_t, interact_hitbox_t{
    void create_shape(fgm_t* fgm, const fan::vec2& position) {
      loco_t::sprite_t::properties_t p;
      p.viewport = &fgm->viewports[viewport_area::editor];
      p.camera = &fgm->cameras[viewport_area::editor];
      p.position = position;
      p.position.z = z_depth++;
      p.size = 0.1;
      p.image = &gloco->default_texture;
      *(loco_t::shape_t*)this = p;

      create_hitbox(fgm, this, p.position);
    }
  };

  struct text_t : loco_t::shape_t, interact_hitbox_t{
    void create_shape(fgm_t* fgm, const fan::vec2& position) {
      loco_t::text_t::properties_t p;
      p.viewport = &fgm->viewports[viewport_area::editor];
      p.camera = &fgm->cameras[viewport_area::editor];
      p.position = position;
      p.position.z = z_depth++;
      p.font_size = 0.1;
      p.text = "text";
      *(loco_t::shape_t*)this = p;

      create_hitbox(fgm, this, p.position);
    }
  };

  struct hitbox_t : loco_t::shape_t, interact_hitbox_t{
    void create_shape(fgm_t* fgm, const fan::vec2& position) {
      loco_t::sprite_t::properties_t p;
      p.viewport = &fgm->viewports[viewport_area::editor];
      p.camera = &fgm->cameras[viewport_area::editor];
      p.position = position;
      p.position.z = z_depth++;
      p.size = 0.1;
      p.image = &fgm->hitbox_image;
      *(loco_t::shape_t*)this = p;

      create_hitbox(fgm, this, p.position);
    }
  };

  struct mark_t : loco_t::shape_t, interact_hitbox_t{
    void create_shape(fgm_t* fgm, const fan::vec2& position) {
      loco_t::sprite_t::properties_t p;
      p.viewport = &fgm->viewports[viewport_area::editor];
      p.camera = &fgm->cameras[viewport_area::editor];
      p.position = position;
      p.position.z = z_depth++;
      p.size = 0.1;
      p.image = &fgm->mark_image;
      *(loco_t::shape_t*)this = p;

      create_hitbox(fgm, this, p.position);
    }
  };

protected:
  //#define BLL_set_CPP_CopyAtPointerChange
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix shape_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeDataType std::variant<sprite_t, text_t, hitbox_t, mark_t>
  #define BLL_set_Link 1
  #include _FAN_PATH(BLL/BLL.h)
public:

  using shape_list_nr_t = shape_list_NodeReference_t;
  shape_list_t shape_list;

	void load() {
		resize_cb();
	}

	void open(const char* texturepack_name) {

		//editor_ratio = fan::vec2(1, 1);
		//move_offset = 0;
		//action_flag = 0;
		theme = loco_t::themes::deep_red();

		texturepack.open_compiled(gloco, texturepack_name);

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

    hitbox_image.load((fan::color*)image, 3);
    mark_image.load((fan::color*)mark_image_pixels, 1);

		gloco->get_window()->add_resize_callback([this](const fan::window_t::resize_cb_data_t& d) {
			resize_cb();
		});
		//pile->loco.get_window()->add_mouse_move_callback([this](const auto& d) {
  //    if (view_action_flag & action::move) {
  //      fan::vec2 size = fan::vec2(
  //        camera[viewport_area::editor].coordinates.right,
  //        camera[viewport_area::editor].coordinates.down
  //      );
  //      *(fan::vec2*)& camera_position -= 
  //        ((gloco->get_mouse_position() - gloco->get_window()->get_previous_mouse_position()) * 
  //        (size / gloco->get_window()->get_size())) * 32;
  //      camera[viewport_area::editor].set_camera_position(camera_position);
  //    }
  //  });
		gloco->get_window()->add_buttons_callback([this](const auto& d) {
      if (d.button != fan::mouse_left) {
        return;
      }
      if (d.state != fan::mouse_state::release) {
        return;
      }
      if (editor_shape == loco_t::shape_type_t::invalid) {
        return;
      }
      fan::vec2 window_size = gloco->get_window()->get_size();

      fan::vec2 mouse_pos = gloco->get_mouse_position(cameras[viewport_area::editor], viewports[viewport_area::editor]);

      if (mouse_pos.x > 1 || mouse_pos.y > 1) {
        return;
      }
      switch (editor_mode) {
        case editor_modes_e::make: {
          auto nr = shape_list.NewNodeLast();
          // todo remove switch
          switch (editor_shape) {
            case loco_t::shape_type_t::sprite: {
              shape_list[nr] = sprite_t();
              break;
            }
            case loco_t::shape_type_t::text: {
              shape_list[nr] = text_t();
              break;
            }
            case loco_t::shape_type_t::hitbox: {
              shape_list[nr] = hitbox_t();
              break;
            }
            case loco_t::shape_type_t::mark: {
              shape_list[nr] = mark_t();
              break;
            }
          }
          fan::print("push");
          std::visit([&](auto&& x) { 
            x.create_shape(this, gloco->get_mouse_position(cameras[viewport_area::editor], viewports[viewport_area::editor]));
            }, shape_list[nr]);
          break;
        };
        case editor_modes_e::mod: {
          break;
        };
      }
      //switch (d.button) {
      //  case fan::mouse_middle: {
      //    if (d.state == fan::mouse_state::press) {
      //      view_action_flag |= action::move;
      //    }
      //    else {
      //      view_action_flag &= ~action::move;
      //    }
      //    break;
      //  }
      //}
    });

		gloco->get_window()->add_keys_callback([this](const auto& d) {
      switch (d.key) {
        case fan::key_f: {
          if (d.state != fan::keyboard_state::press) {
            return;
          }
          camera_position = fan::vec3(0, 0, 0);
          cameras[viewport_area::editor].set_camera_position(camera_position);
          break;
        }
      }
    });

		//// half size
    sidepanel_line_position = fan::vec2(0.5, 0);
		editor_position = fan::vec2(-sidepanel_line_position.x / 2, 0);
		editor_size = editor_position.x + 0.9;

    for (auto& i : cameras) {
      i.open();
    }
    for (auto& i : viewports) {
      i.open();
    }

    editor_viewport = fan::vec4(-1, 1, -1, 1);

    loco_t::line_t::properties_t lp;
		lp.viewport = &viewports[viewport_area::global];
		lp.camera = &cameras[viewport_area::global];
		lp.color = fan::colors::white;

    for (auto& i : lines) {
      i = lp;
    }
	}
	void close() {
		clear();
	}
	void clear() {

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

    iterate_masterpiece([&f](auto& shape) {
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

  fan::vec2 translate_viewport_position(const fan::vec2& value) {
    fan::vec2 window_size = gloco->get_window()->get_size();
    return (value + 1) / 2 * window_size;
  }
  fan::vec2 translate_to_global(const fan::vec2& position) const {
    return position / viewports[viewport_area::global].get_size() * 2 - 1;
  }

  void set_viewport_and_camera(uint32_t area, const fan::vec2& position, const fan::vec2& size) {
    fan::vec2 window_size = gloco->get_window()->get_size();
    fan::vec2 viewport_position = translate_viewport_position(position);
    fan::vec2 viewport_size = translate_viewport_position(size);
    viewports[area].set(viewport_position, viewport_size, window_size);
    cameras[area].set_ortho(fan::vec2(-1, 1), fan::vec2(-1, 1));
  }

  void create_lines() {
    fan::vec3 src(editor_position - editor_size, line_z_depth);
    fan::vec3 dst(editor_position.x + editor_size.x, src.y, line_z_depth);

    lines[0].set_line(src, dst);

    src = dst;
    dst.y = editor_position.y + editor_size.y;
    lines[1].set_line(src, dst);

    src = dst;
    dst.x = editor_position.x - editor_size.x;
    lines[2].set_line(src, dst);

    src = dst;
    dst.y = editor_position.y - editor_size.y;
    lines[3].set_line(src, dst);

    src = fan::vec3(translate_to_global(viewports[viewport_area::sidepanel].get_position()), line_z_depth);
    dst.x = src.x;
    dst.y = cameras[viewport_area::global].coordinates.down;
    lines[4].set_line(src, dst);
  }

  void resize_cb() {
    fan::vec2 window_size = gloco->get_window()->get_size();
    auto get_m = [&] (f32_t scaler) {
      fan::vec2 n = window_size.square_normalize();
      n *= 0.666666;
      return fan::vec2(n.y, n.x);
    };
    static constexpr f32_t editor_scaler = 2. / 3;
    editor_position.x = editor_scaler - 1;
    editor_position.y = 0;
    editor_size = get_m(editor_scaler);
    fan::print(editor_position, editor_size);


    set_viewport_and_camera(viewport_area::global, fan::vec2(-1), fan::vec2(1));
    set_viewport_and_camera(viewport_area::editor, editor_position - editor_size, editor_size);
    set_viewport_and_camera(viewport_area::sidepanel, fan::vec2(sidepanel_line_position.x, -1),
      fan::vec2(-0.5, 1)
    );

    create_lines();

    loco_t::dropdown_t::open_properties_t op;
    op.gui_size = gui_size;
    op.position = fan::vec2(-0.25 + op.gui_size.x, -0.7);
    op.camera = &cameras[viewport_area::sidepanel];
    op.viewport = &viewports[viewport_area::sidepanel];
    op.theme = &theme;
    op.titleable = false;
    op.direction = fan::vec2(1, 0);

    settings_menu = op;

    loco_t::dropdown_t::element_properties_t ep;
    ep.text = "Mak";
    ep.mouse_button_cb = [this] (const auto& d) -> int{
      if (d.button != fan::mouse_left) {
        return 0;
      }
      if (d.button_state != fan::mouse_state::release) {
        return 0;
      }

      editor_mode = editor_modes_e::make;

      loco_t::dropdown_t::open_properties_t op;
      op.gui_size = gui_size;
      op.position = fan::vec2(-0.25 + op.gui_size.x, -0.5);
      op.camera = &cameras[viewport_area::sidepanel];
      op.viewport = &viewports[viewport_area::sidepanel];
      op.theme = &theme;
      op.titleable = false;
      op.direction = fan::vec2(0, 1);

      sidepanel_menu.clear();
      sidepanel_menu = op;
      loco_t::dropdown_t::element_properties_t ep;
      ep.text = "sprite";
      ep.mouse_button_cb = [this](const auto& d) -> int {
        editor_shape = loco_t::shape_type_t::sprite;
        return 0;
      };
      sidepanel_menu.add(ep);
      ep.mouse_button_cb = [this](const auto& d) -> int {
        editor_shape = loco_t::shape_type_t::text;
        return 0;
      };
      ep.text = "text";
      sidepanel_menu.add(ep);
      ep.mouse_button_cb = [this](const auto& d) -> int {
        editor_shape = loco_t::shape_type_t::hitbox;
        return 0;
      };
      ep.text = "hitbox";
      sidepanel_menu.add(ep);
      ep.mouse_button_cb = [this](const auto& d) -> int {
        editor_shape = loco_t::shape_type_t::mark;
        return 0;
      };
      ep.text = "mark";
      sidepanel_menu.add(ep);

      return 0;
    };
    settings_menu.add(ep);

    ep.text = "Mod";
    ep.mouse_button_cb = [this] (const auto& d) -> int {
      editor_mode = editor_modes_e::mod;

      loco_t::dropdown_t::open_properties_t op;
      op.gui_size = gui_size;
      op.position = fan::vec2(-0.25 + op.gui_size.x, -0.5);
      op.camera = &cameras[viewport_area::sidepanel];
      op.viewport = &viewports[viewport_area::sidepanel];
      op.theme = &theme;
      op.titleable = false;
      op.direction = fan::vec2(0, 1);

      sidepanel_menu.clear();
      sidepanel_menu = op;
      return 0;
    };
    settings_menu.add(ep);
  }

  #include "private.h"

  struct shape_mode_e {
    static constexpr uint8_t idle = 0;
    static constexpr uint8_t move = 1;
    static constexpr uint8_t resize = 2;
  };

  uint8_t shape_mode = shape_mode_e::idle;

  struct editor_modes_e {
    static constexpr uint8_t make = 0;
    static constexpr uint8_t mod = 1;
  };

  uint8_t editor_mode = editor_modes_e::make;
  uint16_t editor_shape = loco_t::shape_type_t::invalid;

  loco_t::dropdown_t::menu_id_t sidepanel_menu;

  loco_t::dropdown_t::menu_id_t settings_menu;

  std::array<loco_t::shape_t, 5> lines;

	loco_t::camera_t cameras[3];
	loco_t::viewport_t viewports[3];

	loco_t::theme_t theme;

  fan::vec3 camera_position = 0;

  fan::vec2 editor_position;
	fan::vec2 editor_size;
	fan::vec2 editor_ratio;

  fan::vec2 sidepanel_line_position;
	f32_t line_y_offset_between_types_and_properties;

	loco_t::texturepack_t texturepack;

  loco_t::image_t hitbox_image;
  loco_t::image_t mark_image;

  fan::vec4 editor_viewport;
};

#undef use_key_lambda

#undef fgm_button
#undef fgm_sprite
#undef fgm_text
#undef fgm_hitbox
#undef fgm_mark