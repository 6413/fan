struct button_t {

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

  void push_back(fan::graphics::cid_t* cid, properties_t& p) {
    get_key_value(uint16_t) = p.position.z;
    get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;

    loco_t* loco = get_loco();

    #if defined(loco_vulkan)
      auto& camera = loco->camera_list[p.camera];
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

    sb_push_back(cid, p);

    loco->text.push_back(&sb_get_ri(cid).text_id, tp);

    set_theme(cid, theme, released);

    loco_t::vfi_t::properties_t vfip;
    vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
    vfip.shape.rectangle.camera = p.camera;
    vfip.shape.rectangle.viewport = p.viewport;
    vfip.shape.rectangle.position = p.position;
    vfip.shape.rectangle.size = p.size;
    vfip.flags = p.vfi_flags;
    if (!p.disable_highlight) {
      vfip.mouse_move_cb = [this, cb = p.mouse_move_cb, udata = p.udata, cid_ = cid](const loco_t::vfi_t::mouse_move_data_t& mm_d) -> int {
        loco_t* loco = OFFSETLESS(mm_d.vfi, loco_t, vfi_var_name);
        loco_t::mouse_move_data_t mmd = mm_d;
        if (mm_d.flag->ignore_move_focus_check == false && !loco->button.sb_get_ri(cid_).selected) {
          if (mm_d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
            loco->button.set_theme(cid_, loco->button.get_theme(cid_), hovered);
          }
          else {
            loco->button.set_theme(cid_, loco->button.get_theme(cid_), released);
          }
        }
        mmd.cid = cid_;

        auto theme = get_theme(cid_);
        loco_t::theme_t::mouse_move_data_t td = (loco_t::theme_t::mouse_move_data_t)mmd;
        td.theme = theme;
        theme->mouse_move_cb(td);

        cb(mmd);
        return 0;
      };
      vfip.mouse_button_cb = [this, cb = p.mouse_button_cb, udata = p.udata, cid_ = cid](const loco_t::vfi_t::mouse_button_data_t& ii_d) -> int {
        loco_t* loco = OFFSETLESS(ii_d.vfi, loco_t, vfi_var_name);
        if (ii_d.flag->ignore_move_focus_check == false && !loco->button.sb_get_ri(cid_).selected) {
          if (ii_d.button == fan::mouse_left && ii_d.button_state == fan::mouse_state::press) {
            loco->button.set_theme(cid_, loco->button.get_theme(cid_), pressed);
            ii_d.flag->ignore_move_focus_check = true;
            loco->vfi.set_focus_keyboard(loco->vfi.get_focus_mouse());
          }
        }
        else if (!loco->button.sb_get_ri(cid_).selected) {
          if (ii_d.button == fan::mouse_left && ii_d.button_state == fan::mouse_state::release) {
            if (ii_d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
              loco->button.set_theme(cid_, loco->button.get_theme(cid_), hovered);
            }
            else {
              loco->button.set_theme(cid_, loco->button.get_theme(cid_), released);
            }
            ii_d.flag->ignore_move_focus_check = false;
          }
        }

        loco_t::mouse_button_data_t mid = ii_d;
        mid.cid = cid_;

        auto theme = get_theme(cid_);
        loco_t::theme_t::mouse_button_data_t td = (loco_t::theme_t::mouse_button_data_t)mid;
        td.theme = theme;
        theme->mouse_button_cb(td);

        cb(mid);

        return 0;
      };
      vfip.keyboard_cb = [this, cb = p.keyboard_cb, udata = p.udata, cid_ = cid](const loco_t::vfi_t::keyboard_data_t& kd) -> int {
        loco_t* loco = OFFSETLESS(kd.vfi, loco_t, vfi_var_name);
        loco_t::keyboard_data_t kd_ = kd;
        kd_.cid = cid_;
        auto theme = get_theme(cid_);
        loco_t::theme_t::keyboard_data_t td = (loco_t::theme_t::keyboard_data_t)kd_;
        td.theme = theme;
        theme->keyboard_cb(td);
        cb(kd_);
        return 0;
      };

      // not defined in button
      //vfip.text_cb = [this, cb = p.text_cb, udata = p.udata, cid_ = cid](const loco_t::vfi_t::text_data_t& kd) -> int {
      //  loco_t* loco = OFFSETLESS(kd.vfi, loco_t, vfi_var_name);
      //  loco_t::text_data_t kd_ = kd;
      //  kd_.cid = cid_;
      //  auto theme = get_theme(cid_);
      //  loco_t::theme_t::text_data_t td = *(loco_t::theme_t::text_data_t*)&kd_;
      //  td.theme = theme;
      //  theme->text_cb(td);
      //  cb(kd_);
      //  return 0;
      //};
    }

    loco->vfi.push_back(&sb_get_ri(cid).vfi_id, vfip);
  }
  void erase(fan::graphics::cid_t* cid) {
    loco_t* loco = get_loco();
    auto& ri = sb_get_ri(cid);
    loco->text.erase(&ri.text_id);
    loco->vfi.erase(&ri.vfi_id);
    sb_erase(cid);
  }

  auto& get_ri(fan::graphics::cid_t* cid) {
    return sb_get_ri(cid);
  }

  void draw() {
    sb_draw();
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
    loco_t* loco = get_loco();
    return (loco_t::theme_t*)loco->get_context()->theme_list[nr].theme_id;
  }
  loco_t::theme_t* get_theme(fan::graphics::cid_t* cid) {
    return get_theme(get_ri(cid).theme);
  }
  void set_theme(fan::graphics::cid_t* cid, loco_t::theme_t* theme, f32_t intensity) {
    loco_t* loco = get_loco();
    loco_t::theme_t t = *theme;
    t = t * intensity;
    
    set(cid, &vi_t::color, t.button.color);
    set(cid, &vi_t::outline_color, t.button.outline_color);
    set(cid, &vi_t::outline_size, t.button.outline_size);
    auto& ri = get_ri(cid);
    ri.theme = theme;
    loco->text.set(&ri.text_id, 
      &loco_t::letter_t::vi_t::outline_color, t.button.text_outline_color);
    loco->text.set(&ri.text_id, 
      &loco_t::letter_t::vi_t::outline_size, t.button.text_outline_size);
  }

  template <typename T>
  auto get_button(fan::graphics::cid_t* cid, auto T::* member) {
    loco_t* loco = get_loco();
    return loco->button.get(cid, member);
  }
  template <typename T, typename T2>
  void set_button(fan::graphics::cid_t* cid, auto T::*member, const T2& value) {
    loco_t* loco = get_loco();
    loco->button.set(cid, member, value);
  }

  //template <typename T>
  //T get_text_renderer(fan::graphics::cid_t* cid, auto T::* member) {
  //  loco_t* loco = get_loco();
  //  auto block = sb_get_block(cid);
  //  return loco->text.get(block->p[cid->instance_id].text_id, member);
  //}
  template <typename T, typename T2>
  void set_text_renderer(fan::graphics::cid_t* cid, auto T::*member, const T2& value) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    loco->text.set(block->p[cid->instance_id].text_id, member, value);
  }

  void set_position(fan::graphics::cid_t* cid, const fan::vec3& position) {
    loco_t* loco = get_loco();
    auto& ri = get_ri(cid);
    loco->text.set_position(&ri.text_id, position + fan::vec3(0, 0, 1));
    set_button(cid, &vi_t::position, position);
    loco->vfi.set_rectangle(
      ri.vfi_id,
      &loco_t::vfi_t::set_rectangle_t::position,
      position
    );
  }
  void set_size(fan::graphics::cid_t* cid, const fan::vec3& size) {
    loco_t* loco = get_loco();
    auto& ri = get_ri(cid);
    set_button(cid, &vi_t::size, size);
    loco->vfi.set_rectangle(
      ri.vfi_id,
      &loco_t::vfi_t::set_rectangle_t::size,
      size
    );
  }

  //void set_camera(fan::graphics::cid_t* cid, loco_t::camera_list_NodeReference_t n) {
  //  sb_set_key<bm_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
  //  loco_t* loco = get_loco();
  //  auto block = sb_get_block(cid);
  //  loco->text.set_camera(block->p[cid->instance_id].text_id, n);
  //}

  //fan::graphics::viewport_t* get_viewport(fan::graphics::cid_t* cid) {
  //  loco_t* loco = get_loco();
  //  auto block = sb_get_block(cid);
  //  return loco->get_context()->viewport_list[*block->p[cid->instance_id].key.get_value<1>()].viewport_id;
  //}
  /*void set_viewport(fan::graphics::cid_t* cid, fan::graphics::viewport_list_NodeReference_t n) {
    loco_t* loco = get_loco();
    sb_set_key<instance_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
    auto block = sb_get_block(cid);
    loco->text.set_viewport(block->p[cid->instance_id].text_id, n);
  }*/

  void set_theme(fan::graphics::cid_t* cid, f32_t state) {
    loco_t* loco = get_loco();
    loco->button.set_theme(cid, loco->button.get_theme(cid), state);
  }

    // gets udata from current focus
  /*uint64_t get_id_udata(loco_t::vfi_t::shape_id_t id) {
    loco_t* loco = get_loco();
    auto udata = loco->vfi.get_id_udata(id);
    fan::opengl::cid_t* cid = (fan::opengl::cid_t*)udata;
    auto block = sb_get_block(cid);
    return block->p[cid->instance_id].udata;
  }*/

  void set_selected(fan::graphics::cid_t* cid, bool flag) {
    auto& ri = get_ri(cid);
    ri.selected = flag;
  }

  // dont edit values
  const auto& get_text_instance(fan::graphics::cid_t* cid) {
    loco_t* loco = get_loco();
    auto& ri = get_ri(cid);
    return loco->text.get_instance(&ri.text_id);
  }

  fan::string get_text(fan::graphics::cid_t* cid) {
    loco_t* loco = get_loco();
    auto& ri = get_ri(cid);
    return loco->text.get_instance(&ri.text_id).text;
  }
  void set_text(fan::graphics::cid_t* cid, const fan::string& text) {
    loco_t* loco = get_loco();
    auto& ri = get_ri(cid);
    loco->text.set_text(&ri.text_id, text);
  }

  ri_t* get_instance_properties(fan::graphics::cid_t* cid) {
    return &sb_get_ri(cid);
  }

  void set_focus_mouse(fan::graphics::cid_t* cid) {
    get_loco()->vfi.set_focus_mouse(get_instance_properties(cid)->vfi_id);
  }

  void set_focus_keyboard(fan::graphics::cid_t* cid) {
    get_loco()->vfi.set_focus_keyboard(get_instance_properties(cid)->vfi_id);
  }

  void set_focus_text(fan::graphics::cid_t* cid) {
    get_loco()->vfi.set_focus_text(get_instance_properties(cid)->vfi_id);
  }

  void set_focus(fan::graphics::cid_t* cid) {
    set_focus_mouse(cid);
    set_focus_keyboard(cid);
    set_focus_text(cid);
  }

  void set_depth(fan::opengl::cid_t* cid, f32_t depth) {
    auto& vfi_id = get_instance_properties(cid)->vfi_id;
    get_loco()->vfi.shape_list[vfi_id].shape_data.depth = depth;
    sb_set_depth(cid, depth);
  }

  #if defined(loco_vulkan)
  uint32_t m_camera_index = 0;
  #endif
};