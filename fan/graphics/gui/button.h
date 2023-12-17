struct button_t {


  static constexpr typename loco_t::shape_type_t shape_type = loco_t::shape_type_t::button;

  static constexpr f32_t boundary_multiplier = 0.9;
  static constexpr f32_t released = 1.0;
  static constexpr f32_t hovered = 1.2;
  static constexpr f32_t pressed = 1.4;

  struct vi_t {
    loco_t::position3_t position = 0; 
    f32_t pad;
    fan::vec2 size = 0; 
    fan::vec2 rotation_point = 0; 
    fan::vec3 angle = 0; 
    f32_t pad2;
    fan::color color = fan::colors::white; 
    fan::color outline_color; 
    f32_t outline_size;
    f32_t pad3[3];
  };

  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(vi_t) / 4));

  struct context_key_t {
    using parsed_masterpiece_t = fan::masterpiece_t<
      uint16_t,
      loco_t::camera_list_NodeReference_t,
      fan::graphics::viewport_list_NodeReference_t
    >;
    struct key_t : parsed_masterpiece_t {}key;
  };

  struct cid_t;

  struct ri_t {
    uint8_t selected = 0;
    loco_t::theme_t* theme = 0;
    loco_t::shape_t text_id;
    loco_t::shapes_t::vfi_t::shape_id_t vfi_id;
    uint64_t udata;
    loco_t::mouse_button_cb_t mouse_button_cb = [](const loco_t::mouse_button_data_t&) -> int { return 0; };
    loco_t::mouse_move_cb_t mouse_move_cb = [](const loco_t::mouse_move_data_t&) -> int { return 0; };
    loco_t::keyboard_cb_t keyboard_cb = [](const loco_t::keyboard_data_t&) -> int { return 0; }; 
    bool blending = false;
    fan::vec3 original_position;
    fan::vec2 original_size;
  };

  struct properties_t : vi_t, ri_t, context_key_t {
    using type_t = button_t;
     
    fan::string text; 
    f32_t font_size = 0.1; 

    bool disable_highlight = false; 
 
    loco_t::camera_t* camera = &gloco->default_camera->camera; 
    loco_t::viewport_t* viewport = &gloco->default_camera->viewport;
  };

  void push_back(loco_t::cid_nt_t& id, properties_t& p) {
    get_key_value(uint16_t) = p.position.z;
    get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;

    #if defined(loco_vulkan)
      auto& camera = gloco->camera_list[p.camera];
      if (camera.camera_index.button == (decltype(camera.camera_index.button))-1) {
        camera.camera_index.button = m_camera_index++;
        m_shader.set_camera(loco, camera.camera_id, camera.camera_index.button);
      }
    #endif

    auto theme = p.theme;
    loco_t::shapes_t::responsive_text_t::properties_t tp;
    tp.size = p.size * boundary_multiplier; // padding
    tp.color = theme->button.text_color;
    tp.position = p.position;
    tp.text = p.text;
    tp.position.z = p.position.z + 1;
    tp.viewport = p.viewport;
    tp.camera = p.camera;
    tp.line_limit = 1;
    tp.letter_size_y_multipler = 1;

    sb_push_back(id, p);


    // todo remove
    auto& ri = sb_get_ri(id);
    ri.text_id = tp;

    ri.original_position = p.position;
    ri.original_size = p.size;

    set_theme(id, theme, released);

    loco_t::shapes_t::vfi_t::properties_t vfip;
    vfip.shape_type = loco_t::shapes_t::vfi_t::shape_t::rectangle;
    vfip.shape.rectangle->camera = p.camera;
    vfip.shape.rectangle->viewport = p.viewport;
    vfip.shape.rectangle->position = p.position;
    vfip.shape.rectangle->size = p.size;
    if (!p.disable_highlight) {
      vfip.mouse_move_cb = [this, udata = p.udata, id_ = id](const loco_t::shapes_t::vfi_t::mouse_move_data_t& mm_d) mutable -> int {
        loco_t::mouse_move_data_t mmd = mm_d;
        if (mm_d.flag->ignore_move_focus_check == false && !gloco->shapes.button.sb_get_ri(id_).selected) {
          if (mm_d.mouse_stage == loco_t::shapes_t::vfi_t::mouse_stage_e::inside) {
            gloco->shapes.button.set_theme(id_, gloco->shapes.button.get_theme(id_), hovered);
          }
          else {
            gloco->shapes.button.set_theme(id_, gloco->shapes.button.get_theme(id_), released);
          }
        }
        mmd.id = id_;

        auto theme = get_theme(id_);
        loco_t::theme_t::mouse_move_data_t td = (loco_t::theme_t::mouse_move_data_t)mmd;
        td.theme = theme;
        theme->mouse_move_cb(td);

        sb_get_ri(id_).mouse_move_cb(mmd);
        return 0;
      };
      vfip.mouse_button_cb = [this, udata = p.udata, id_ = id](const loco_t::shapes_t::vfi_t::mouse_button_data_t& ii_d) mutable -> int {
        if (ii_d.flag->ignore_move_focus_check == false && !gloco->shapes.button.sb_get_ri(id_).selected) {
          if (ii_d.button == fan::mouse_left && ii_d.button_state == fan::mouse_state::press) {
            gloco->shapes.button.set_theme(id_, gloco->shapes.button.get_theme(id_), pressed);
            ii_d.flag->ignore_move_focus_check = true;
            gloco->shapes.vfi.set_focus_keyboard(gloco->shapes.vfi.get_focus_mouse());
          }
        } 
        else if (!gloco->shapes.button.sb_get_ri(id_).selected) {
          if (ii_d.button == fan::mouse_left && ii_d.button_state == fan::mouse_state::release) {
            if (ii_d.mouse_stage == loco_t::shapes_t::vfi_t::mouse_stage_e::inside) {
              gloco->shapes.button.set_theme(id_, gloco->shapes.button.get_theme(id_), hovered);
            }
            else {
              gloco->shapes.button.set_theme(id_, gloco->shapes.button.get_theme(id_), released);
            }
            ii_d.flag->ignore_move_focus_check = false;
          }
        }

        loco_t::mouse_button_data_t mid = ii_d;
        mid.id = id_;

        auto theme = get_theme(id_);
        loco_t::theme_t::mouse_button_data_t td = (loco_t::theme_t::mouse_button_data_t)mid;
        td.theme = theme;
        theme->mouse_button_cb(td);

        if (sb_get_ri(id_).mouse_button_cb(mid)) {
          return 1;
        }

        return 0;
      };
      vfip.keyboard_cb = [this, udata = p.udata, id_ = id](const loco_t::shapes_t::vfi_t::keyboard_data_t& kd) mutable -> int {
        loco_t::keyboard_data_t kd_ = kd;
        kd_.id = id_;
        auto theme = get_theme(id_);
        loco_t::theme_t::keyboard_data_t td = (loco_t::theme_t::keyboard_data_t)kd_;
        td.theme = theme;
        theme->keyboard_cb(td);
        sb_get_ri(id_).keyboard_cb(kd_);
        return 0;
      };

      // not defined in button
      //vfip.text_cb = [this, udata = p.udata, id_ = cid](const loco_t::shapes_t::vfi_t::text_data_t& kd) -> int {
      //  loco_t* loco = OFFSETLESS(kd.vfi, loco_t, vfi_var_name);
      //  loco_t::text_data_t kd_ = kd;
      //  kd_.cid = id_;
      //  auto theme = get_theme(id_);
      //  loco_t::theme_t::text_data_t td = *(loco_t::theme_t::text_data_t*)&kd_;
      //  td.theme = theme;
      //  theme->text_cb(td);
      //  sb_get_ri(id_).text_cb(kd_);
      //  return 0;
      //};
    }

    gloco->shapes.vfi.push_back(sb_get_ri(id).vfi_id, vfip);
  }
  void erase(loco_t::cid_nt_t& id) {
    
    auto& ri = sb_get_ri(id);
    ri.text_id.erase();
    gloco->shapes.vfi.erase(ri.vfi_id);
    sb_erase(id);
  }

  auto& get_ri(loco_t::cid_nt_t& id) {
    return sb_get_ri(id);
  }

  void draw(const redraw_key_t &redraw_key, loco_bdbt_NodeReference_t key_root) {
    if (redraw_key.blending) {
      m_current_shader = &m_blending_shader;
    }
    else {
      m_current_shader = &m_shader;
    }
    sb_draw(key_root);
  }

  #if defined(loco_opengl)
    #define sb_shader_vertex_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/button.vs)
    #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/button.fs)
  #endif

  button_t() {
    sb_open(sb_shader_vertex_path, sb_shader_fragment_path);
  }
  ~button_t() {
    sb_close();
  }

  #include _FAN_PATH(graphics/shape_builder.h)

  loco_t::theme_t* get_theme(fan::graphics::theme_list_NodeReference_t nr) {
    return (loco_t::theme_t*)gloco->get_context().theme_list[nr].theme_id;
  }
  loco_t::theme_t* get_theme(loco_t::cid_nt_t& id) {
    return get_theme(get_ri(id).theme);
  }
  void set_theme(loco_t::cid_nt_t& id, loco_t::theme_t* theme, f32_t intensity) {
    
    loco_t::theme_t t = *theme;
    t = t * intensity;
    
    set(id, &vi_t::color, t.button.color);
    set(id, &vi_t::outline_color, t.button.outline_color);
    set(id, &vi_t::outline_size, t.button.outline_size);
    auto& ri = get_ri(id);
    ri.theme = theme;
    ri.text_id.set_outline_color(t.button.text_outline_color);
    ri.text_id.set_outline_size(t.button.text_outline_size);
  }

  template <typename T>
  auto get_button(loco_t::cid_nt_t& id, auto T::* member) {
    
    return gloco->shapes.button.get(id, member);
  }
  template <typename T, typename T2>
  void set_button(loco_t::cid_nt_t& id, auto T::*member, const T2& value) {
    gloco->shapes.button.set(id, member, value);
  }

  //template <typename T>
  //T get_text_renderer(loco_t::cid_nt_t& id, auto T::* member) {
  //  
  //  auto block = sb_get_block(id);
  //  return gloco->text.get(block->p[cid->instance_id].text_id, member);
  //}
  //template <typename T, typename T2>
  //void set_text_renderer(loco_t::cid_nt_t& id, auto T::*member, const T2& value) {
  //  
  //  auto block = sb_get_block(id);
  //  gloco->text.set(block->p[id->instance_id].text_id, member, value);
  //}

  void set_position(loco_t::cid_nt_t& id, const fan::vec3& position) {
    
    auto& ri = get_ri(id);
    fan::vec3 temp = position + fan::vec3(0, 0, 1);
    ri.text_id.set_position(temp);
    ri.original_position = position;
    set_button(id, &vi_t::position, position);
    gloco->shapes.vfi.set_rectangle(
      ri.vfi_id,
      &loco_t::shapes_t::vfi_t::set_rectangle_t::position,  
      position
    );
  }
  void set_position_ar(loco_t::cid_nt_t& id, const fan::vec3& position) {

    auto& ri = get_ri(id);
    fan::vec3 temp = position + fan::vec3(0, 0, 1);
    ri.text_id.set_position(temp);
    set_button(id, &vi_t::position, position);
    gloco->shapes.vfi.set_rectangle(
      ri.vfi_id,
      &loco_t::shapes_t::vfi_t::set_rectangle_t::position,
      position
    );
  }
  void set_size(loco_t::cid_nt_t& id, const fan::vec2& size) {
    
    auto& ri = get_ri(id);
    ri.original_size = size;
    set_button(id, &vi_t::size, size);
    gloco->shapes.vfi.set_rectangle(
      ri.vfi_id,
      &loco_t::shapes_t::vfi_t::set_rectangle_t::size,
      size
    );
    ri.text_id.set_size(size * boundary_multiplier); // pad
  }
  void set_size_ar(loco_t::cid_nt_t& id, const fan::vec2& size) {
    auto& ri = get_ri(id);
    //ri.original_size = size;
    set_button(id, &vi_t::size, size);
    gloco->shapes.vfi.set_rectangle(
      ri.vfi_id,
      &loco_t::shapes_t::vfi_t::set_rectangle_t::size,
      size
    );
    ri.text_id.set_size(size * boundary_multiplier); // pad
  }

  //void set_camera(loco_t::cid_nt_t& id, loco_t::camera_list_NodeReference_t n) {
  //  sb_set_key<context_key_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
  //  
  //  auto block = sb_get_block(id);
  //  gloco->text.set_camera(block->p[cid->instance_id].text_id, n);
  //}

  //fan::graphics::viewport_t* get_viewport(loco_t::cid_nt_t& id) {
  //  
  //  auto block = sb_get_block(id);
  //  return gloco->get_context().viewport_list[*block->p[cid->instance_id].key.get_value<1>()].viewport_id;
  //}
  /*void set_viewport(loco_t::cid_nt_t& id, fan::graphics::viewport_list_NodeReference_t n) {
    
    sb_set_key<instance_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
    auto block = sb_get_block(id);
    gloco->text.set_viewport(block->p[cid->instance_id].text_id, n);
  }*/

  void set_theme(loco_t::cid_nt_t& id, f32_t state) {
    
    gloco->shapes.button.set_theme(id, gloco->shapes.button.get_theme(id), state);
  }

    // gets udata from current focus
  /*uint64_t get_id_udata(loco_t::shapes_t::vfi_t::shape_id_t id) {
    
    auto udata = gloco->shapes.vfi.get_id_udata(id);
    fan::opengl::cid_t* cid = (fan::opengl::cid_t*)udata;
    auto block = sb_get_block(id);
    return block->p[cid->instance_id].udata;
  }*/

  void set_selected(loco_t::cid_nt_t& id, bool flag) {
    auto& ri = get_ri(id);
    ri.selected = flag;
  }

  fan::string get_text(loco_t::cid_nt_t& id) {
    
    auto& ri = get_ri(id);
    return gloco->shapes.responsive_text.get_text(ri.text_id);
  }
  void set_text(loco_t::cid_nt_t& id, const fan::string& text) {
    
    auto& ri = get_ri(id);
    gloco->shapes.responsive_text.set_text(ri.text_id, text);
  }

  ri_t* get_instance_properties(loco_t::cid_nt_t& id) {
    return &sb_get_ri(id);
  }

  void set_focus_mouse(loco_t::cid_nt_t& id) {
    gloco->shapes.vfi.set_focus_mouse(get_instance_properties(id)->vfi_id);
  }

  void set_focus_keyboard(loco_t::cid_nt_t& id) {
    gloco->shapes.vfi.set_focus_keyboard(get_instance_properties(id)->vfi_id);
  }

  void set_focus_text(loco_t::cid_nt_t& id) {
    gloco->shapes.vfi.set_focus_text(get_instance_properties(id)->vfi_id);
  }

  void set_focus(loco_t::cid_nt_t& id) {
    set_focus_mouse(id);
    set_focus_keyboard(id);
    set_focus_text(id);
  }

  void set_depth(loco_t::cid_nt_t& id, f32_t depth) {
    auto& vfi_id = get_instance_properties(id)->vfi_id;
    gloco->shapes.vfi.shape_list[loco_t::shapes_t::vfi_t::shape_id_wrap_t(&vfi_id)].shape_data.depth = depth;
    sb_set_depth(id, depth);
  }

  properties_t get_properties(loco_t::cid_nt_t& id) {
    properties_t p = sb_get_properties(id);
    p.theme = get_theme(id);

    p.position = sb_get_vi(id).position;
    p.text = gloco->shapes.responsive_text.get_text(sb_get_ri(id).text_id);
    p.font_size = 1; // TODO
    return p;
  }

  fan::vec3 get_position(loco_t::cid_nt_t& id) {
    auto& ri = get_ri(id);
    return ri.original_position;
  }

  fan::vec3 get_position_ar(loco_t::cid_nt_t& id) {
    return sb_get_vi(id).position;
  }
  fan::vec2 get_size_ar(loco_t::cid_nt_t& id) {
    return sb_get_vi(id).size;
  }

  fan::vec2 get_size(loco_t::cid_nt_t& id) {
    auto& ri = get_ri(id);
    return ri.original_size;
  }

  #if defined(loco_vulkan)
  uint32_t m_camera_index = 0;
  #endif
};