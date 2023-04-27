struct button_t {

  static constexpr typename loco_t::shape_type_t::_t shape_type = loco_t::shape_type_t::button;

  static constexpr f32_t released = 1.0;
  static constexpr f32_t hovered = 1.2;
  static constexpr f32_t pressed = 1.4;

  struct vi_t {
    loco_button_vi_t
  };

  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(vi_t) / 4));

  struct bm_properties_t {
    loco_button_bm_properties_t
  };

  struct cid_t;

  struct ri_t : bm_properties_t {
    loco_button_ri_t
  };

  #define make_key_value(type, name) \
    type& name = *key.get_value<decltype(key)::get_index_with_type<type>()>();

  struct properties_t : vi_t, ri_t {
    using type_t = button_t;
    loco_button_properties_t
  };

  #undef make_key_value

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
    loco_t::text_t::properties_t tp;
    tp.color = theme->button.text_color;
    tp.font_size = p.font_size;
    tp.position = p.position;
    tp.text = p.text;
    tp.position.z = p.position.z + 1;
    tp.viewport = p.viewport;
    tp.camera = p.camera;

    sb_push_back(id, p);

    sb_get_ri(id).text_id = tp;

    set_theme(id, theme, released);

    loco_t::vfi_t::properties_t vfip;
    vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
    vfip.shape.rectangle.camera = p.camera;
    vfip.shape.rectangle.viewport = p.viewport;
    vfip.shape.rectangle.position = p.position;
    vfip.shape.rectangle.size = p.size;
    if (!p.disable_highlight) {
      vfip.mouse_move_cb = [this, udata = p.udata, id_ = id](const loco_t::vfi_t::mouse_move_data_t& mm_d) mutable -> int {
        loco_t::mouse_move_data_t mmd = mm_d;
        if (mm_d.flag->ignore_move_focus_check == false && !gloco->button.sb_get_ri(id_).selected) {
          if (mm_d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
            gloco->button.set_theme(id_, gloco->button.get_theme(id_), hovered);
          }
          else {
            gloco->button.set_theme(id_, gloco->button.get_theme(id_), released);
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
      vfip.mouse_button_cb = [this, udata = p.udata, id_ = id](const loco_t::vfi_t::mouse_button_data_t& ii_d) mutable -> int {
        if (ii_d.flag->ignore_move_focus_check == false && !gloco->button.sb_get_ri(id_).selected) {
          if (ii_d.button == fan::mouse_left && ii_d.button_state == fan::mouse_state::press) {
            gloco->button.set_theme(id_, gloco->button.get_theme(id_), pressed);
            ii_d.flag->ignore_move_focus_check = true;
            gloco->vfi.set_focus_keyboard(gloco->vfi.get_focus_mouse());
          }
        } 
        else if (!gloco->button.sb_get_ri(id_).selected) {
          if (ii_d.button == fan::mouse_left && ii_d.button_state == fan::mouse_state::release) {
            if (ii_d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
              gloco->button.set_theme(id_, gloco->button.get_theme(id_), hovered);
            }
            else {
              gloco->button.set_theme(id_, gloco->button.get_theme(id_), released);
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
      vfip.keyboard_cb = [this, udata = p.udata, id_ = id](const loco_t::vfi_t::keyboard_data_t& kd) mutable -> int {
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
      //vfip.text_cb = [this, udata = p.udata, id_ = cid](const loco_t::vfi_t::text_data_t& kd) -> int {
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

    gloco->vfi.push_back(&sb_get_ri(id).vfi_id, vfip);
  }
  void erase(loco_t::cid_nt_t& id) {
    
    auto& ri = sb_get_ri(id);
    ri.text_id.erase();
    gloco->vfi.erase(&ri.vfi_id);
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
    #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/button.vs)
  #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/button.fs)
  #elif defined(loco_vulkan)
    #define vulkan_buffer_count 4
    #define sb_shader_vertex_path graphics/glsl/vulkan/2D/objects/button.vert
    #define sb_shader_fragment_path graphics/glsl/vulkan/2D/objects/button.frag
  #endif
  #define vk_sb_ssbo
  #define vk_sb_vp
  #include _FAN_PATH(graphics/shape_builder.h)

  button_t() {
    sb_open();
  }
  ~button_t() {
    sb_close();
  }

  loco_t::theme_t* get_theme(fan::graphics::theme_list_NodeReference_t nr) {
    return (loco_t::theme_t*)gloco->get_context()->theme_list[nr].theme_id;
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
    
    return gloco->button.get(id, member);
  }
  template <typename T, typename T2>
  void set_button(loco_t::cid_nt_t& id, auto T::*member, const T2& value) {
    
    gloco->button.set(id, member, value);
  }

  //template <typename T>
  //T get_text_renderer(loco_t::cid_nt_t& id, auto T::* member) {
  //  
  //  auto block = sb_get_block(id);
  //  return gloco->text.get(block->p[cid->instance_id].text_id, member);
  //}
  template <typename T, typename T2>
  void set_text_renderer(loco_t::cid_nt_t& id, auto T::*member, const T2& value) {
    
    auto block = sb_get_block(id);
    gloco->text.set(block->p[id->instance_id].text_id, member, value);
  }

  void set_position(loco_t::cid_nt_t& id, const fan::vec3& position) {
    
    auto& ri = get_ri(id);
    ri.text_id.set_position(position + fan::vec3(0, 0, 1));
    set_button(id, &vi_t::position, position);
    gloco->vfi.set_rectangle(
      ri.vfi_id,
      &loco_t::vfi_t::set_rectangle_t::position,
      position
    );
  }
  void set_size(loco_t::cid_nt_t& id, const fan::vec3& size) {
    
    auto& ri = get_ri(id);
    set_button(id, &vi_t::size, size);
    gloco->vfi.set_rectangle(
      ri.vfi_id,
      &loco_t::vfi_t::set_rectangle_t::size,
      size
    );
  }

  //void set_camera(loco_t::cid_nt_t& id, loco_t::camera_list_NodeReference_t n) {
  //  sb_set_key<bm_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
  //  
  //  auto block = sb_get_block(id);
  //  gloco->text.set_camera(block->p[cid->instance_id].text_id, n);
  //}

  //fan::graphics::viewport_t* get_viewport(loco_t::cid_nt_t& id) {
  //  
  //  auto block = sb_get_block(id);
  //  return gloco->get_context()->viewport_list[*block->p[cid->instance_id].key.get_value<1>()].viewport_id;
  //}
  /*void set_viewport(loco_t::cid_nt_t& id, fan::graphics::viewport_list_NodeReference_t n) {
    
    sb_set_key<instance_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
    auto block = sb_get_block(id);
    gloco->text.set_viewport(block->p[cid->instance_id].text_id, n);
  }*/

  void set_theme(loco_t::cid_nt_t& id, f32_t state) {
    
    gloco->button.set_theme(id, gloco->button.get_theme(id), state);
  }

    // gets udata from current focus
  /*uint64_t get_id_udata(loco_t::vfi_t::shape_id_t id) {
    
    auto udata = gloco->vfi.get_id_udata(id);
    fan::opengl::cid_t* cid = (fan::opengl::cid_t*)udata;
    auto block = sb_get_block(id);
    return block->p[cid->instance_id].udata;
  }*/

  void set_selected(loco_t::cid_nt_t& id, bool flag) {
    auto& ri = get_ri(id);
    ri.selected = flag;
  }

  // dont edit values
  auto get_text_instance(loco_t::cid_nt_t& id) {
    
    auto& ri = get_ri(id);
    return gloco->text.get_instance(ri.text_id);
  }

  fan::string get_text(loco_t::cid_nt_t& id) {
    
    auto& ri = get_ri(id);
    return gloco->text.get_instance(ri.text_id).text;
  }
  void set_text(loco_t::cid_nt_t& id, const fan::string& text) {
    
    auto& ri = get_ri(id);
    gloco->text.set_text(ri.text_id, text);
  }

  ri_t* get_instance_properties(loco_t::cid_nt_t& id) {
    return &sb_get_ri(id);
  }

  void set_focus_mouse(loco_t::cid_nt_t& id) {
    gloco->vfi.set_focus_mouse(get_instance_properties(id)->vfi_id);
  }

  void set_focus_keyboard(loco_t::cid_nt_t& id) {
    gloco->vfi.set_focus_keyboard(get_instance_properties(id)->vfi_id);
  }

  void set_focus_text(loco_t::cid_nt_t& id) {
    gloco->vfi.set_focus_text(get_instance_properties(id)->vfi_id);
  }

  void set_focus(loco_t::cid_nt_t& id) {
    set_focus_mouse(id);
    set_focus_keyboard(id);
    set_focus_text(id);
  }

  void set_depth(loco_t::cid_nt_t& id, f32_t depth) {
    auto& vfi_id = get_instance_properties(id)->vfi_id;
    gloco->vfi.shape_list[vfi_id].shape_data.depth = depth;
    sb_set_depth(id, depth);
  }

  properties_t get_properties(loco_t::cid_nt_t& id) {
    properties_t p = sb_get_properties(id);
    p.camera = gloco->camera_list[*p.key.get_value<loco_t::camera_list_NodeReference_t>()].camera_id;
    p.theme =  get_theme(id);
    p.viewport = gloco->get_context()->viewport_list[*p.key.get_value<fan::graphics::viewport_list_NodeReference_t>()].viewport_id;

    p.position = get_text_instance(id).position;
    p.text = get_text_instance(id).text;
    p.font_size = get_text_instance(id).font_size;
    return p;
  }

  #if defined(loco_vulkan)
  uint32_t m_camera_index = 0;
  #endif
};