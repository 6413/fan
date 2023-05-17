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

  static constexpr fan::vec2 gui_size = fan::vec2(0.3, 0.1);

  static constexpr f32_t line_z_depth = 0xfff;
  static constexpr f32_t right_click_z_depth = 11;

  static inline f32_t z_depth = 1;

  bool recreate_sidepanel_mod() {
    loco_t::dropdown_t::open_properties_t op;
    op.gui_size = gui_size;
    op.position = fan::vec2(-0.25 + op.gui_size.x, -0.5);
    op.camera = &cameras[viewport_area::sidepanel];
    op.viewport = &viewports[viewport_area::sidepanel];
    op.theme = &theme;
    op.titleable = false;
    op.direction = fan::vec2(0, 1);
    op.text_box = true;


    sidepanel_menu.clear();
    sidepanel_menu.open(op);

    if (selected_shape_nr.iic()) {
      return 1;
    }

    std::visit([&](auto&& x) {
      x.create_properties(this);
    }, shape_list[selected_shape_nr]);
    return 0;
  }

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
    void create_hitbox(fgm_t* fgm, auto* shape, const fan::vec3& position, const fan::vec2& size) {
      loco_t::vfi_t::properties_t vfip;
      vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
      vfip.shape.rectangle->viewport = &fgm->viewports[viewport_area::editor];
      vfip.shape.rectangle->camera = &fgm->cameras[viewport_area::editor];
      vfip.shape.rectangle->position = position;
      vfip.shape.rectangle->size = size;
      vfip.mouse_button_cb = [fgm, snr = shape->nr](const auto& d) -> int{
        if (fgm->editor_mode != fgm_t::editor_modes_e::mod) {
          return 0;
        }

        if (d.button_state == fan::mouse_state::press) {
          d.flag->ignore_move_focus_check = true;
          fgm->shape_mode = fgm_t::shape_mode_e::move;
          fgm->selected_shape_nr = snr;
          fgm->recreate_sidepanel_mod();
        }
        else if (d.button_state == fan::mouse_state::release) {
          d.flag->ignore_move_focus_check = false;
          fgm->shape_mode = fgm_t::shape_mode_e::idle;
        }
        return 0;
      };
      vfip.mouse_move_cb = [fgm, snr = shape->nr](const auto& d) -> int {

        if (fgm->selected_shape_nr != snr) {
          return 0;
        }

        switch (fgm->shape_mode) {
          case fgm_t::shape_mode_e::move: {
            gloco->vfi.shape_list[gloco->vfi.focus.mouse].shape_data.shape.rectangle->position = d.position;
            auto* s = &std::get<std::remove_pointer_t<decltype(shape)>>(fgm->shape_list[snr]);
            s->set_position(d.position);
            break;
          }
          case fgm_t::shape_mode_e::resize: {
            // todo
            break;
          }
          default: {
            return 0;
          }
        }

        fgm->recreate_sidepanel_mod();
        return 0;
      };
      gloco->vfi.push_back(&vfi_id, vfip);
    }
    void create_hitbox(fgm_t* fgm, auto* shape, const fan::vec3& position) {
      create_hitbox(fgm, shape, position, shape->get_size());
    }
    void set_hitbox_position(const fan::vec3& p) {
      auto& rectangle = gloco->vfi.shape_list[vfi_id].shape_data;
      rectangle.shape.rectangle->position = *(fan::vec2*)&p;
      rectangle.depth = p.z;
    }
    void set_hitbox_size(const fan::vec2& p) {
      auto& rectangle = gloco->vfi.shape_list[vfi_id].shape_data;
      rectangle.shape.rectangle->size = p;
    }
  };
//protected:

  #define BLL_set_declare_NodeReference 1
  #define BLL_set_declare_rest 0
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix shape_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeDataType std::variant<button_t, sprite_t, text_t, hitbox_t, mark_t>
  #define BLL_set_Link 1
  #include _FAN_PATH(BLL/BLL.h)
//public:
  using shape_list_nr_t = shape_list_NodeReference_t;

  #define create_keyboard_cb(code) \
      ep.keyboard_cb = [fgm, snr = nr](const loco_t::keyboard_data_t& d) {  \
      if (d.key != fan::key_enter) {  \
        return 0;  \
      }  \
      if (d.keyboard_state != fan::keyboard_state::press) {  \
        return 0;  \
      } \
      auto shape = &std::get<std::remove_reference_t<decltype(*this)>>(fgm->shape_list[snr]); \
      auto text_box = ((loco_t::shape_t*)&d.id); \
      code \
      return 0; \
    };

  
  bool does_id_exist(const fan::string& id) {

    auto it = shape_list.GetNodeFirst();

    bool exists = false;
    while (it != shape_list.dst) {
      std::visit([&](auto&& shape) {
        if (shape.id == id) {
          exists = true;
        }
      }, shape_list[it]);
      if (exists) {
        return true;
      };
      it = it.Next(&shape_list);
    }

    return false;
  }

  inline static uint64_t id_maker = 0;
  fan::string make_unqiue_id() {
    while (does_id_exist(std::to_string(id_maker))) ++id_maker;
    return std::to_string(id_maker);
  }

  struct button_t : loco_t::shape_t, interact_hitbox_t{
    shape_list_nr_t nr;

    void create_shape(fgm_t* fgm, shape_list_nr_t shape_nr, const fan::vec2& position) {
      nr = shape_nr;

      loco_t::button_t::properties_t p;
      p.viewport = &fgm->viewports[viewport_area::editor];
      p.camera = &fgm->cameras[viewport_area::editor];
      p.position = position;
      p.position.z = z_depth++;
      p.font_size = 0.1;
      p.size = fan::vec2(0.2, 0.1);
      p.text = "text";
      p.theme = &fgm->theme;
      *(loco_t::shape_t*)this = p;

      id = fgm->make_unqiue_id();

      create_hitbox(fgm, this, fan::vec3(*(fan::vec2*)&p.position, z_depth++));
    }
    void create_properties(fgm_t* fgm) {
      loco_t::dropdown_t::element_properties_t ep;
      ep.text = fan::format("{}", get_position().to_string().c_str());
      create_keyboard_cb(
        fan::vec3 position = fan::string_to<fan::vec3>(text_box->get_text());
        shape->set_position(position);
        shape->set_hitbox_position(fan::vec3(*(fan::vec2*)&position, position.z + 1));
      );
      fgm->sidepanel_menu.add(ep);
    }

    fan::string to_string() {
      fan::string f;
      fan::write_to_string(f, loco_t::shape_type_t::button);
      stage_maker_shape_format::shape_button_t data;
      data.position = get_position();
      data.size = get_size();
      data.font_size = get_font_size();
      data.text = get_text();
      data.id = id;
      f += shape_to_string(data);
      return f;
    }
    uint64_t from_string(fgm_t* fgm, const fan::string& f, uint64_t off, auto shape_nr) {
      loco_t::shape_type_t::_t shape_type = *(loco_t::shape_type_t::_t*)&f[off];
      if (shape_type != loco_t::shape_type_t::button) {
        return off;
      }

      off += sizeof(loco_t::shape_type_t::_t);

      stage_maker_shape_format::shape_button_t data;
      data.iterate_masterpiece([&f, &off](auto& o) {
        o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, off);
      });

      loco_t::button_t::properties_t p = data.get_properties(
        fgm->viewports[viewport_area::editor],
        fgm->cameras[viewport_area::editor],
        fgm->theme
      );

      *(loco_t::shape_t*)this = p;
      id = data.id;
      nr = shape_nr;
      //                                                           so that vfi doesnt collide with buttons hitbox
      create_hitbox(fgm, this, fan::vec3(*(fan::vec2*)&p.position, p.position.z + 1));
      return off;
    }

    fan::string id;
  };

  struct sprite_t : loco_t::shape_t, interact_hitbox_t{

    shape_list_nr_t nr;

    void create_shape(fgm_t* fgm, shape_list_nr_t shape_nr, const fan::vec2& position) {
      nr = shape_nr;

      loco_t::sprite_t::properties_t p;
      p.viewport = &fgm->viewports[viewport_area::editor];
      p.camera = &fgm->cameras[viewport_area::editor];
      p.position = position;
      p.position.z = z_depth++;
      p.size = 0.1;
      p.image = &gloco->default_texture;
      *(shape_t*)this = p;

      id = fgm->make_unqiue_id();

      create_hitbox(fgm, this, p.position);
    }
    void create_properties(fgm_t* fgm) {
      loco_t::dropdown_t::element_properties_t ep;
      ep.text = fan::format("{}", get_position().to_string().c_str());
      create_keyboard_cb(
        fan::vec3 position = fan::string_to<fan::vec3>(text_box->get_text());
        shape->set_position(position);
        shape->set_hitbox_position(position);
      );
      fgm->sidepanel_menu.add(ep);
      ep.text = fan::format("{}", get_size().to_string().c_str());
      create_keyboard_cb(
        auto size = fan::string_to<fan::vec2>(text_box->get_text());
        shape->set_size(size);
        shape->set_hitbox_size(size);
      );
      fgm->sidepanel_menu.add(ep);
      ep.text = fan::format("{}", texturepack_name);
      create_keyboard_cb(
        loco_t::texturepack_t::ti_t ti;
        if (fgm->texturepack.qti(text_box->get_text(), &ti)) {
          fan::print_no_space("failed to load texture:", text_box->get_text());
          return 0;
        }

        shape->texturepack_name = text_box->get_text();

        auto& data = fgm->texturepack.get_pixel_data(ti.pack_id);
        shape->set_image(&data.image);
        gloco->sprite.set(*shape, &loco_t::sprite_t::vi_t::tc_position, ti.position / data.image.size);
        gloco->sprite.set(*shape, &loco_t::sprite_t::vi_t::tc_size, ti.size / data.image.size);
      );
      fgm->sidepanel_menu.add(ep);

      ep.text = fan::format("{}", id);
      create_keyboard_cb(
        if (fgm->does_id_exist(text_box->get_text())) {
          fan::print("id already exists, skipping...");
          return 0;
        }
        shape->id = text_box->get_text();
      );
      fgm->sidepanel_menu.add(ep);

      #if defined(fgm_build_model_maker)
      ep.text = fan::format("{}", group_id);
      
      create_keyboard_cb(
        shape->group_id = std::stoul(text_box->get_text());
      );
      fgm->sidepanel_menu.add(ep);
      #endif
    }

    fan::string to_string() {
      fan::string f;
      fan::write_to_string(f, loco_t::shape_type_t::sprite);
      stage_maker_shape_format::shape_sprite_t data;
      data.position = get_position();
      data.size = get_size();
      data.texturepack_name = texturepack_name;
      data.parallax_factor = parallax_factor;
      data.id = id;
      #if defined(fgm_build_model_maker)
      data.group_id = group_id;
      #endif
      f += shape_to_string(data);
      return f;
    }
    uint64_t from_string(fgm_t* fgm, const fan::string& f, uint64_t off, auto shape_nr) {
      loco_t::shape_type_t::_t shape_type = *(loco_t::shape_type_t::_t*)&f[off];
      if (shape_type != loco_t::shape_type_t::sprite) {
        return off;
      }

      off += sizeof(loco_t::shape_type_t::_t);

      stage_maker_shape_format::shape_sprite_t data;
      data.iterate_masterpiece([&f, &off](auto& o) {
        o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, off);
      });

      loco_t::sprite_t::properties_t p = data.get_properties(
        fgm->viewports[viewport_area::editor],
        fgm->cameras[viewport_area::editor],
        fgm->texturepack
      );

      *(loco_t::shape_t*)this = p;

      nr = shape_nr;

      id = data.id;
      create_hitbox(fgm, this, p.position);
      texturepack_name = data.texturepack_name;
      #if defined(fgm_build_model_maker)
        group_id = data.group_id;
      #endif

      return off;
    }

    fan::string id;
    fan::string texturepack_name;
    f32_t parallax_factor = 0;

    #if defined(fgm_build_model_maker)
    uint32_t group_id = 0;
    #endif
  };

  struct text_t : loco_t::shape_t, interact_hitbox_t{
    shape_list_nr_t nr;

    void create_shape(fgm_t* fgm, shape_list_nr_t shape_nr,const fan::vec2& position) {
      nr = shape_nr;

      loco_t::text_t::properties_t p;
      p.viewport = &fgm->viewports[viewport_area::editor];
      p.camera = &fgm->cameras[viewport_area::editor];
      p.position = position;
      p.position.z = z_depth++;
      p.font_size = 0.1;
      p.text = "text";
      *(loco_t::shape_t*)this = p;

      id = fgm->make_unqiue_id();

      create_hitbox(fgm, this, p.position, get_text_size());
    }
    void create_properties(fgm_t* fgm) {
      loco_t::dropdown_t::element_properties_t ep;
      ep.text = fan::format("position {}", get_position().to_string().c_str());
      create_keyboard_cb(
        fan::vec3 position = fan::string_to<fan::vec3>(text_box->get_text());
        shape->set_position(position);
        shape->set_hitbox_position(position);
      );
      fgm->sidepanel_menu.add(ep);

      ep.text = fan::format("{}", get_text());
      create_keyboard_cb(
        shape->set_text(text_box->get_text());
      );
      fgm->sidepanel_menu.add(ep);

      {
        fan::color c = get_color();
        ep.text = fan::format("{}", ((fan::vec4*)&c)->to_string().c_str());
        create_keyboard_cb(
          shape->set_color(fan::string_to<fan::color>(text_box->get_text()));
        );
        fgm->sidepanel_menu.add(ep);

      }
      ep.text = fan::format("{}", id);
      create_keyboard_cb(
        if (fgm->does_id_exist(text_box->get_text())) {
          fan::print("id already exists, skipping...");
          return 0;
        }
        shape->id = text_box->get_text();
      );
      fgm->sidepanel_menu.add(ep);
    }


    fan::string to_string() {
      fan::string f;
      fan::write_to_string(f, loco_t::shape_type_t::text);
      stage_maker_shape_format::shape_text_t data;
      data.position = get_position();
      data.size = get_font_size();
      data.text = get_text();
      data.id = id;
      f += shape_to_string(data);
      return f;
    }
    uint64_t from_string(fgm_t* fgm, const fan::string& f, uint64_t off, auto shape_nr) {
      loco_t::shape_type_t::_t shape_type = *(loco_t::shape_type_t::_t*)&f[off];
      if (shape_type != loco_t::shape_type_t::text) {
        return off;
      }

      off += sizeof(loco_t::shape_type_t::_t);

      stage_maker_shape_format::shape_text_t data;
      data.iterate_masterpiece([&f, &off](auto& o) {
        o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, off);
      });

      loco_t::text_t::properties_t p = data.get_properties(
        fgm->viewports[viewport_area::editor],
        fgm->cameras[viewport_area::editor]
      );

      *(loco_t::shape_t*)this = p;
      nr = shape_nr;

      id = data.id;
      create_hitbox(fgm, this, p.position, get_text_size());

      return off;
    }

    fan::string id;
  };

  struct hitbox_t : loco_t::shape_t, interact_hitbox_t{
    shape_list_nr_t nr;

    void create_shape(fgm_t* fgm, shape_list_nr_t shape_nr,const fan::vec2& position) {
      nr = shape_nr;

      loco_t::sprite_t::properties_t p;
      p.viewport = &fgm->viewports[viewport_area::editor];
      p.camera = &fgm->cameras[viewport_area::editor];
      p.position = position;
      p.position.z = z_depth++;
      p.size = 0.1;
      p.image = &fgm->hitbox_image;
      p.blending = true;
      *(loco_t::shape_t*)this = p;

      id = fgm->make_unqiue_id();

      create_hitbox(fgm, this, p.position);
    }
    void create_properties(fgm_t* fgm) {
      loco_t::dropdown_t::element_properties_t ep;
      ep.text = fan::format("position {}", get_position().to_string().c_str());
      create_keyboard_cb(
        fan::vec3 position = fan::string_to<fan::vec3>(text_box->get_text());
        shape->set_position(position);
        shape->set_hitbox_position(position);
      );
      fgm->sidepanel_menu.add(ep);
    }

    fan::string to_string() {
      fan::string f;
      fan::write_to_string(f, loco_t::shape_type_t::hitbox);
      stage_maker_shape_format::shape_hitbox_t data;
      data.position = get_position();
      data.size = get_size();
      data.vfi_type = vfi_type;
      data.id = id;
      f += shape_to_string(data);
      return f;
    }
    uint64_t from_string(fgm_t* fgm, const fan::string& f, uint64_t off, auto shape_nr) {
      loco_t::shape_type_t::_t shape_type = *(loco_t::shape_type_t::_t*)&f[off];
      if (shape_type != loco_t::shape_type_t::hitbox) {
        return off;
      }

      off += sizeof(loco_t::shape_type_t::_t);

      stage_maker_shape_format::shape_hitbox_t data;
      data.iterate_masterpiece([&f, &off](auto& o) {
        o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, off);
      });

      loco_t::sprite_t::properties_t p = data.get_properties(
        fgm->viewports[viewport_area::editor],
        fgm->cameras[viewport_area::editor],
        &fgm->hitbox_image
      );

      *(loco_t::shape_t*)this = p;

      nr = shape_nr;

      id = data.id;
      vfi_type = data.vfi_type;
      create_hitbox(fgm, this, p.position);

      return off;
    }

    fan::string id;
    loco_t::vfi_t::shape_type_t vfi_type = loco_t::vfi_t::shape_t::rectangle;
  };

  struct mark_t : loco_t::shape_t, interact_hitbox_t{

    shape_list_nr_t nr;

    void create_shape(fgm_t* fgm, shape_list_nr_t shape_nr, const fan::vec2& position) {

      nr = shape_nr;

      loco_t::sprite_t::properties_t p;
      p.viewport = &fgm->viewports[viewport_area::editor];
      p.camera = &fgm->cameras[viewport_area::editor];
      p.position = position;
      p.position.z = z_depth++;
      p.size = 0.1;
      p.image = &fgm->mark_image;
      *(shape_t*)this = p;

      id = fgm->make_unqiue_id();

      create_hitbox(fgm, this, p.position);
    }
    void create_properties(fgm_t* fgm) {
      loco_t::dropdown_t::element_properties_t ep;
      ep.text = fan::format("position {}", get_position().c_str());
      create_keyboard_cb(
        fan::vec3 position = fan::string_to<fan::vec3>(text_box->get_text());
        shape->set_position(position);
        shape->set_hitbox_position(position);
      );
      fgm->sidepanel_menu.add(ep);

      ep.text = fan::format("size {}", get_size().c_str());
      create_keyboard_cb(
        shape->set_size(fan::string_to<fan::vec3>(text_box->get_text()));
      );
      fgm->sidepanel_menu.add(ep);

      ep.text = fan::format("{}", id);
      create_keyboard_cb(
        if (fgm->does_id_exist(text_box->get_text())) {
          fan::print("id already exists, skipping...");
          return 0;
        }
        shape->id = text_box->get_text();
      );
      fgm->sidepanel_menu.add(ep);

      #if defined(fgm_build_model_maker)
      ep.text = fan::format("{}", group_id);
      create_keyboard_cb(
        shape->group_id = std::stoul(text_box->get_text());
      );
      fgm->sidepanel_menu.add(ep);
      #endif
    }

    fan::string to_string() {
      fan::string f;
      fan::write_to_string(f, loco_t::shape_type_t::mark);
      stage_maker_shape_format::shape_mark_t data;
      data.position = get_position();
      data.id = id;
      #if defined(fgm_build_model_maker)
      data.group_id = group_id;
      #endif
      f += shape_to_string(data);
      return f;
    }
    uint64_t from_string(fgm_t* fgm, const fan::string& f, uint64_t off, auto shape_nr) {
      loco_t::shape_type_t::_t shape_type = *(loco_t::shape_type_t::_t*)&f[off];
      if (shape_type != loco_t::shape_type_t::mark) {
        return off;
      }

      off += sizeof(loco_t::shape_type_t::_t);

      stage_maker_shape_format::shape_mark_t data;
      data.iterate_masterpiece([&f, &off](auto& o) {
        o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, off);
      });

      loco_t::sprite_t::properties_t p = data.get_properties(
        fgm->viewports[viewport_area::editor],
        fgm->cameras[viewport_area::editor],
        &fgm->mark_image
      );

      *(loco_t::shape_t*)this = p;

      nr = shape_nr;

      #if defined(fgm_build_model_maker)
      group_id = data.group_id;
      #endif
      id = data.id;
      create_hitbox(fgm, this, p.position);

      return off;
    }

    fan::string id;
    //loco_t::vfi_t::shape_type_t vfi_type = loco_t::vfi_t::shape_t::rectangle;
    #if defined(fgm_build_model_maker)
    uint32_t group_id = 0;
    #endif
  };

  #undef create_keyboard_cb

//protected:
  //#define BLL_set_CPP_CopyAtPointerChange
  #define BLL_set_declare_NodeReference 0
  #define BLL_set_declare_rest 1
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix shape_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeDataType std::variant<button_t, sprite_t, text_t, hitbox_t, mark_t>
  #define BLL_set_Link 1
  #include _FAN_PATH(BLL/BLL.h)
//public:

  shape_list_t shape_list;

	void load() {

    selected_shape_nr.sic();

    loaded = true;

    loco_t::line_t::properties_t lp;
		lp.viewport = &viewports[viewport_area::global];
		lp.camera = &cameras[viewport_area::global];
		lp.color = fan::colors::white;

    for (auto& i : lines) {
      i = lp;
    }

		resize_cb();

    #if defined(fgm_build_stage_maker)
    loco_t::button_t::properties_t p;
    p.position = fan::vec3(-0.8, -0.8, line_z_depth);
    p.size = fan::vec2(0.2, 0.1);
    p.camera = &cameras[viewport_area::global];
    p.viewport = &viewports[viewport_area::global];
    p.text = "<-";
    p.mouse_button_cb = [&](const auto& d) {
      if (d.button != fan::mouse_left) {
        return 0;
      }
      if (d.button_state != fan::mouse_state::release) {
        return 0;
      }

      stage_maker_t* sm = OFFSETLESS(this, stage_maker_t, fgm);
      write_to_file(sm->stage_str);

      for (auto& i : lines) {
        i.erase();
      }

      settings_menu.clear();
      sidepanel_menu.clear();
      
      sm->open_stage(stage_t::main);
      sm->open_stage_menu();
		  sm->open_options_menu();
      

      /*auto it = shape_list.GetNodeFirst();
      while (it != shape_list.dst) {
        std::visit([](auto&& shape) {
          shape.erase();
        }, shape_list[it]);
        it = it.Next(&shape_list);
      }*/
      shape_list.Clear();
      return_button.erase();
      close();

      loaded = false;

      return 1;
    };
    p.theme = &theme;
    return_button = p;
    #endif

    add_shape_cb_nr = gloco->get_window()->add_buttons_callback([this](const auto& d) {
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
            case loco_t::shape_type_t::button: {
              shape_list[nr] = button_t();
              break;
            }
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
          std::visit([&](auto&& x) { 
            x.create_shape(this, nr, gloco->get_mouse_position(cameras[viewport_area::editor], viewports[viewport_area::editor]));
          }, shape_list[nr]);
          selected_shape_nr = nr;
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
      if (!loaded) {
        return;
      }
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

    // move to open and erase after close
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
        case fan::key_delete: {
          if (d.state != fan::keyboard_state::press) {
            return;
          }
          if (selected_shape_nr.iic()) {
            return;
          }
          shape_list.unlrec(selected_shape_nr);
          selected_shape_nr.sic();

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
    lp.src = 0;
    lp.dst = 0;

    for (auto& i : lines) {
      i = lp;
    }
	}
	void close() {
    gloco->get_window()->remove_buttons_callback(add_shape_cb_nr);
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

template <typename... Types, typename Func>
static void forEachType(std::variant<Types...>& variant, Func&& func) {
    (func(Types{}), ...);
}
  void read_from_file(const fan::string& stage_name) {
    #if defined(fgm_build_stage_maker)
    fan::string path = get_fgm_full_path(stage_name);
    #else
    fan::string path = stage_name;
    #endif
    if (!fan::io::file::exists(path)) {
      return;
    }
    fan::string f;
    fan::io::file::read(path, &f);

    if (f.empty()) {
      return;
    }
    uint64_t off = 0;

    uint32_t file_version = fan::read_data<uint32_t>(f, off);

    bool wrote = true;
    bool skip = false;
    switch (file_version) {
      // no need to put other versions than current because it would not compile
      case stage_maker_format_version: {
        fgm_t::shape_list_nr_t nr;
        while (off < f.size()) {
          nr = shape_list.NewNodeLast();
          skip = false;
          forEachType(shape_list[nr], [&](auto value) {
            if (skip) {
              return;
            }
            if (off >= f.size()) {
              return;
            }
            shape_list[nr] = decltype(value)();
            uint64_t old = off;
            std::visit([&](auto&& shape) {
              off = shape.from_string(this, f, off, nr);
            }, shape_list[nr]);
            if (off != old) {
              skip = true;
            }
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

    auto it = shape_list.GetNodeFirst();
    while (it != shape_list.dst) {
      std::visit([&](auto&& shape) {
        f += shape.to_string();
      }, shape_list[it]);
      it = it.Next(&shape_list);
    }

    fan::io::file::write(
      #if defined(fgm_build_stage_maker)
      get_fgm_full_path(stage_name),
      #else
      stage_name,
      #endif
      f,
      std::ios_base::binary
    );
  }

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
    src.z = line_z_depth;
    dst.x = src.x;
    dst.y = cameras[viewport_area::global].coordinates.down;
    lines[4].set_line(src, dst);
  }

  void resize_cb() {
    fan::vec2 window_size = gloco->get_window()->get_size();
    auto get_m = [&](f32_t scaler) {
      fan::vec2 n = window_size.square_normalize();
      n *= 0.666666;
      return fan::vec2(n.y, n.x);
    };
    static constexpr f32_t editor_scaler = 2. / 3;
    editor_position.x = editor_scaler - 1;
    editor_position.y = 0;
    editor_size = get_m(editor_scaler);


    set_viewport_and_camera(viewport_area::global, fan::vec2(-1), fan::vec2(1));

    {
      fan::vec2 window_size = gloco->get_window()->get_size();
      fan::vec2 viewport_position = translate_viewport_position(editor_position);
      fan::vec2 ed = editor_size * window_size;
      viewports[viewport_area::editor].set(viewport_position - ed / 2, ed, window_size);
      cameras[viewport_area::editor].set_ortho(fan::vec2(-1, 1), fan::vec2(-1, 1));

    }

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

    settings_menu.clear();
    settings_menu.open(op);

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
      sidepanel_menu.open(op);

      loco_t::dropdown_t::element_properties_t ep;
      ep.text = "button";
      ep.mouse_button_cb = [this](const auto& d) -> int {
        editor_shape = loco_t::shape_type_t::button;
        return 0;
      };
      sidepanel_menu.add(ep);

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

      if (recreate_sidepanel_mod()) {
        return 0;
      }

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

#if defined(fgm_build_stage_maker)
  loco_t::shape_t return_button;
#endif

  // to make code shorter
  shape_list_nr_t selected_shape_nr;

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
  bool loaded = false;

  loco_t::image_t hitbox_image;
  loco_t::image_t mark_image;

  fan::vec4 editor_viewport;
  fan::window_t::buttons_callback_NodeReference_t add_shape_cb_nr;
};

#undef use_key_lambda

#undef fgm_button
#undef fgm_sprite
#undef fgm_text
#undef fgm_hitbox
#undef fgm_mark