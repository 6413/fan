struct text_box_t {

  static constexpr typename loco_t::shape_type_t::_t shape_type = loco_t::shape_type_t::text_box;

  struct cursor_properties {
    static constexpr uint64_t speed = 5e+8;
    static constexpr fan::vec2 size = fan::vec2(0.0015, 0.015);
    static constexpr fan::color color = fan::color(1, 1, 1, 0.8);
  };

  static constexpr f32_t released = 1.0;
  static constexpr f32_t hovered = 1.2;
  static constexpr f32_t pressed = 1.4;

  struct vi_t {
    loco_text_box_vi_t
  };

  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(vi_t) / 4));

  struct bm_properties_t {
    loco_text_box_bm_properties_t
  };

  struct cid_t;

  struct ri_t : bm_properties_t {
    loco_text_box_ri_t
  };

  #define make_key_value(type, name) \
    type& name = *key.get_value<decltype(key)::get_index_with_type<type>()>();

  struct properties_t : vi_t, ri_t {
    using type_t = text_box_t;
    loco_text_box_properties_t
  };

  #undef make_key_value

  void push_back(loco_t::cid_nt_t& id, properties_t& p) {

    get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;

    auto theme = gloco->text_box.get_theme(p.theme);
    loco_t::text_t::properties_t tp;
    tp.color = theme->button.text_color;
    tp.font_size = p.font_size;
    tp.position = p.position;
    tp.text = p.text;
    tp.position.z += p.position.z + 1;
    tp.camera = p.camera;
    tp.viewport = p.viewport;

    #if defined(loco_vulkan)
      auto& camera = loco->camera_list[p.camera];
      if (camera.camera_index.text_box == (decltype(camera.camera_index.text_box))-1) {
        camera.camera_index.text_box = m_camera_index++;
        m_shader.set_camera(loco, camera.camera_id, camera.camera_index.text_box);
      }
    #endif

    sb_push_back(id, p);
    auto& ri = sb_get_ri(id);

    ri.text_id = tp;


    set_theme(id, theme, released);

    loco_t::vfi_t::properties_t vfip;
    vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
    vfip.shape.rectangle.camera = p.camera;
    vfip.shape.rectangle.viewport = p.viewport;
    vfip.shape.rectangle.position = p.position;
    vfip.shape.rectangle.size = p.size;
    vfip.flags = p.vfi_flags;
    fed_t::properties_t fp;
    fp.font_size = p.font_size;
    ri.fed.open(fp);
    ri.fed.push_text(p.text);
    if (!p.disable_highlight) {
      vfip.mouse_move_cb = [this, cb = p.mouse_move_cb, udata = p.udata, id_ = id](const loco_t::vfi_t::mouse_move_data_t& mm_d) mutable -> int {
        loco_t* loco = OFFSETLESS(mm_d.vfi, loco_t, vfi_var_name);
        loco_t::mouse_move_data_t mmd = mm_d;
        auto& ri = loco->text_box.sb_get_ri(id_);
        if (mm_d.flag->ignore_move_focus_check == false && !ri.selected) {
          if (mm_d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
            loco->text_box.set_theme(id_, loco->text_box.get_theme(id_), hovered);
          }
          else {
            loco->text_box.set_theme(id_, loco->text_box.get_theme(id_), released);
          }
        }

        //if (loco->get_window()->key_pressed(fan::mouse_left) && loco->vfi.get_focus_keyboard()) {
        //  fan::print("a");
        //  // src press
        //  fan::vec2 src = fan::vec2(mm_d.position) - fan::vec2(get_text_left_position(id_));
        //  // dst release
        //  src.x = fan::clamp(src.x, (f32_t)0, src.x);
        //  fan::vec2 dst = src;

        //  pr.fed.set_mouse_position(src, dst);
        //  update_cursor(id_);
        //}

        mmd.id = id_;
        cb(mmd);
        return 0;
      };
      vfip.mouse_button_cb = [this, cb = p.mouse_button_cb, udata = p.udata, id_ = id](const loco_t::vfi_t::mouse_button_data_t& ii_d) mutable -> int {
        loco_t* loco = OFFSETLESS(ii_d.vfi, loco_t, vfi_var_name);
        auto& ri = loco->text_box.sb_get_ri(id_);
        if (ii_d.flag->ignore_move_focus_check == false && !ri.selected) {
          if (ii_d.button == fan::mouse_left && ii_d.button_state == fan::mouse_state::press) {
            loco->text_box.set_theme(id_, loco->text_box.get_theme(id_), pressed);
            ii_d.flag->ignore_move_focus_check = true;
            loco->vfi.set_focus_keyboard(loco->vfi.get_focus_mouse());
            loco->vfi.set_focus_text(loco->vfi.get_focus_mouse());

            fan::vec2 src = fan::vec2(ii_d.position) - fan::vec2(get_text_left_position(id_));
            // dst release
            src.x = fan::clamp(src.x, (f32_t)0, src.x);
            fan::vec2 dst = src;

            ri.fed.set_mouse_position(src, dst);
            update_cursor(id_);
          }
        }
        else if (!ri.selected) {
          if (loco->ev_timer.time_list.find(&timer) == loco->ev_timer.time_list.end()) {
            loco->ev_timer.start(&timer, cursor_properties::speed);
            update_cursor(id_);
          }

          if (ii_d.button == fan::mouse_left && ii_d.button_state == fan::mouse_state::release) {
            if (ii_d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
              loco->text_box.set_theme(id_, loco->text_box.get_theme(id_), hovered);
            }
            else {
              loco->text_box.set_theme(id_, loco->text_box.get_theme(id_), released);
            }
            ii_d.flag->ignore_move_focus_check = false;
          }
        }

        loco_t::mouse_button_data_t mid = ii_d;
        mid.id = id_;
        cb(mid);

        return 0;
      };
      vfip.keyboard_cb = [this, cb = p.keyboard_cb, udata = p.udata, id_ = id](const loco_t::vfi_t::keyboard_data_t& kd) mutable -> int {
        loco_t* loco = OFFSETLESS(kd.vfi, loco_t, vfi_var_name);
        auto& ri = loco->text_box.sb_get_ri(id_);
        auto update_text = [loco, this, id_]() mutable {
          wed_t::CursorInformation_t ci;
          auto& ri = loco->text_box.sb_get_ri(id_);
          ri.fed.m_wed.GetCursorInformation(ri.fed.m_cr, &ci);
          switch (ci.type) {
            case wed_t::CursorType::FreeStyle: {
              auto text = ri.fed.get_text(ci.FreeStyle.LineReference);
              loco->text_box.set_text(id_, text);

              break;
            }
            case wed_t::CursorType::Selection: {
              fan::assert_test(0);
              break;
            }
          }
        };

        if (kd.keyboard_state != fan::keyboard_state::release) {
          switch (kd.key) {
            case fan::key_backspace: { ri.fed.freestyle_erase_character(); update_text();  break; }
            case fan::key_delete: { ri.fed.freestyle_erase_character_right(); update_text(); break; }
            case fan::key_home:  { ri.fed.freestyle_move_line_begin(); break; }
            case fan::key_end: { ri.fed.freestyle_move_line_end(); break; }
            case fan::key_left: { ri.fed.freestyle_move_left(); break; }
            case fan::key_right: { ri.fed.freestyle_move_right(); break; }
            case fan::key_v: {
              if (loco->get_window()->key_pressed(fan::key_control)) {
                auto pasted_text = fan::io::get_clipboard_text(loco->get_window()->get_handle());

                ri.fed.push_text(pasted_text);

                update_text();
              }
              break;
            }
            case fan::key_enter: {
              loco->vfi.invalidate_focus_mouse();
              loco->vfi.invalidate_focus_keyboard();
              loco->vfi.invalidate_focus_text();
              render_cursor = false;
              fan::ev_timer_t::cb_data_t data;
              data.ev_timer = &loco->ev_timer;
              data.timer = &timer;
              timer.cb(data);
              data.ev_timer->stop(&timer);
              loco_t::keyboard_data_t kd_ = kd;
              kd_.id = id_;
              cb(kd_);
              return 0;
            }
            default: {
              return 0;
            }
          }
        }

        update_cursor(id_);

        loco_t::keyboard_data_t kd_ = kd;
        kd_.id = id_;
        cb(kd_);
        return 0;
      };
      vfip.text_cb = [this, cb = p.text_cb, udata = p.udata, id_ = id](const loco_t::vfi_t::text_data_t& td) mutable -> int {
        loco_t* loco = OFFSETLESS(td.vfi, loco_t, vfi_var_name);
        auto& ri = loco->text_box.sb_get_ri(id_);

        switch (td.key) {
          default: {
            ri.fed.add_character(td.key);
            wed_t::CursorInformation_t ci;
            auto& fed = sb_get_ri(id_).fed;
            fed.m_wed.GetCursorInformation(fed.m_cr, &ci);
            switch (ci.type) {
              case wed_t::CursorType::FreeStyle: {
                loco->text_box.set_text(id_, ri.fed.get_text(ci.FreeStyle.LineReference));
                break;
              }
              case wed_t::CursorType::Selection: {
                fan::assert_test(0);
                break;
              }
            }
            update_cursor(id_);
            break;
          }
        }

        loco_t::text_data_t td_ = td;
        td_.id = id_;
        cb(td_);
        return 0;
      };
    }

    gloco->vfi.push_back(&ri.vfi_id, vfip);

    cursor = fan_init_struct(
      loco_t::rectangle_t::properties_t,
      .position.z = tp.position.z + 1,
      .size = cursor_properties::size * (gloco->get_camera_view_size(*p.camera) / 2),
      .size.y = p.font_size,
      .camera = p.camera,
      .viewport = p.viewport,
      .color = fan::colors::transparent,
      .blending = true
    );
  }
  void erase(loco_t::cid_nt_t& id) {
    // what if you remove other thing whats not focused
    invalidate_cursor();
    auto& ri = sb_get_ri(id);
    ri.text_id.erase();
    gloco->vfi.erase(&ri.vfi_id);
    sb_erase(id);
  }

  void update_cursor(loco_t::cid_nt_t& id) {
    auto& ri = sb_get_ri(id);
    wed_t::CursorInformation_t ci;
    auto& fed = ri.fed;
    cursor.set_camera(get_properties(id).camera);
    cursor.set_viewport(get_properties(id).viewport);
    fed.m_wed.GetCursorInformation(fed.m_cr, &ci);
    fan::vec3 p = get_properties(id).position;
    p.z += 1;
;
    f32_t font_size = sb_get_ri(id).text_id.get_font_size();

    ri.fed.set_font_size(ci.FreeStyle.LineReference, font_size);
    // set_font_size invalidates ci so need to refetch it
    fed.m_wed.GetCursorInformation(fed.m_cr, &ci);
    switch (ci.type) {
      case wed_t::CursorType::FreeStyle: {
        uint32_t line_index = fed.m_wed.GetLineIndexByLineReference(ci.FreeStyle.LineReference);
        uint32_t character_index = fed.m_wed.GetCharacterIndexByCharacterReference(
          ci.FreeStyle.LineReference,
          ci.FreeStyle.CharacterReference
        );
        auto cp = get_character_position(id, line_index, character_index);
        p = fan::vec3(*(fan::vec2*)&cp, p.z);
        break;
      }
      case wed_t::CursorType::Selection: {
        fan::assert_test(0);
        //m_wed.GetLineIndexByLineReference(ci.Selection.LineReference);
        break;
      }
    }
    cursor.set_position(p);
    cursor.set_size(cursor_properties::size * (gloco->get_camera_view_size(*get_properties(id).camera) * 2));
    gloco->ev_timer.stop(&timer);
    render_cursor = true;
    fan::ev_timer_t::cb_data_t d;
    d.ev_timer = &gloco->ev_timer;
    d.timer = &timer;
    timer.cb(d);
  }

  ri_t* get_instance_properties(loco_t::cid_nt_t& id) {
    return &sb_get_ri(id);
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

  void invalidate_cursor() {
    cursor.~id_t();
    gloco->ev_timer.stop(&timer);
  }

  text_box_t() {
    sb_open();
  }
  ~text_box_t() {
    // check erase, need to somehow iterate block
    fan::assert_test(0);
    sb_close();
  }

  loco_t::theme_t* get_theme(fan::graphics::theme_list_NodeReference_t nr) {
    return (loco_t::theme_t*)gloco->get_context()->theme_list[nr].theme_id;
  }
  loco_t::theme_t* get_theme(loco_t::cid_nt_t& id) {
    return get_theme(sb_get_ri(id).theme);
  }
  void set_theme(loco_t::cid_nt_t& id, loco_t::theme_t* theme, f32_t intensity) {
    loco_t::theme_t t = *theme;
    t = t * intensity;

    set(id, &vi_t::color, t.button.color);
    set(id, &vi_t::outline_color, t.button.outline_color);
    set(id, &vi_t::outline_size, t.button.outline_size);
    auto& ri = sb_get_ri(id);
    ri.theme = theme;
    sb_get_ri(id).text_id.set_outline_color(t.button.text_outline_color);
    sb_get_ri(id).text_id.set_outline_size(t.button.text_outline_size);
  }

  template <typename T>
  T get_button(loco_t::cid_nt_t& id, T vi_t::* member) {
    return gloco->text_box.get(id, member);
  }
  template <typename T, typename T2>
  void set_button(loco_t::cid_nt_t& id, T vi_t::* member, const T2& value) {
    gloco->text_box.set(id, member, value);
  }

  //template <typename T>
  //T get_text_renderer(loco_t::cid_nt_t& id, T loco_t::letter_t::vi_t::* member) {
  //  loco_t* loco = get_loco();
  //  return loco->text.get(sb_get_ri(id).text_id, member);
  //}
  template <typename T, typename T2>
  void set_text_renderer(loco_t::cid_nt_t& id, T loco_t::letter_t::vi_t::* member, const T2& value) {
    gloco->text.set(sb_get_ri(id).text_id, member, value);
  }

  void set_position(loco_t::cid_nt_t& id, const fan::vec3& position) {
    auto& ri = sb_get_ri(id);
    ri.text_id.set_position(position + fan::vec3(0, 0, 1));
    set_button(id, &vi_t::position, position);
    gloco->vfi.set_rectangle(
      ri.vfi_id,
      &loco_t::vfi_t::set_rectangle_t::position,
      position
    );
  }
  void set_size(loco_t::cid_nt_t& id, const fan::vec3& size) {
    auto& ri = sb_get_ri(id);
    set_button(id, &vi_t::size, size);
    gloco->vfi.set_rectangle(
      ri.vfi_id,
      &loco_t::vfi_t::set_rectangle_t::size,
      size
    );
  }

 /* loco_t::camera_t* get_camera(loco_t::cid_nt_t& id) {
    loco_t* loco = get_loco();
    return loco->camera_list[*block->p[cid->instance_id].key.get_value<0>()].camera_id;
  }
  void set_camera(loco_t::cid_nt_t& id, loco_t::camera_list_NodeReference_t n) {
    sb_set_key<bm_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
    loco_t* loco = get_loco();
    auto block = sb_get_block(id);
    loco->text.set_camera(block->p[cid->instance_id].text_id, n);
  }

  fan::graphics::viewport_t* get_viewport(loco_t::cid_nt_t& id) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(id);
    return loco->get_context()->viewport_list[*block->p[cid->instance_id].key.get_value<1>()].viewport_id;
  }
  void set_viewport(loco_t::cid_nt_t& id, fan::graphics::viewport_list_NodeReference_t n) {
    loco_t* loco = get_loco();
    sb_set_key<bm_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
    auto block = sb_get_block(id);
    loco->text.set_viewport(block->p[cid->instance_id].text_id, n);
  }*/

  void set_theme(loco_t::cid_nt_t& id, f32_t state) {
    gloco->text_box.set_theme(id, gloco->text_box.get_theme(id), state);
  }

  // gets udata from current focus
/*uint64_t get_id_udata(loco_t::vfi_t::shape_id_t id) {
  loco_t* loco = get_loco();
  auto udata = loco->vfi.get_id_udata(id);
  fan::opengl::cid_t* cid = (fan::opengl::cid_t*)udata;
  auto block = sb_get_block(id);
  return block->p[cid->instance_id].udata;
}*/

  void set_selected(loco_t::cid_nt_t& id, bool flag) {
    sb_get_ri(id).selected = flag;
  }

  fan::string get_text(loco_t::cid_nt_t& id) {
    return gloco->text.get_instance(sb_get_ri(id).text_id).text;
  }
  void set_text(loco_t::cid_nt_t& id, const fan::string& text) {
    gloco->text.set_text(sb_get_ri(id).text_id, text);
  }

  fan::vec3 get_text_left_position(loco_t::cid_nt_t& id) {
    f32_t text_length = sb_get_ri(id).text_id.get_text_size().x;
    fan::vec3 center = get_button(id, &text_box_t::vi_t::position);
    center.x -= text_length * 0.5;
    return center;
  }

  fan::vec3 get_character_position(loco_t::cid_nt_t& id, uint32_t line, uint32_t width) {
    fan::vec3 center = get_button(id, &text_box_t::vi_t::position);
    if (width == 0) {
      
      if (gloco->text.get_instance(sb_get_ri(id).text_id).text.empty()) {
        return center;
      }
    }
    //fan::print(width);
    fan::vec3 p = get_text_left_position(id);
    const fan::string& text = gloco->text.get_instance(sb_get_ri(id).text_id).text;
    f32_t font_size = sb_get_ri(id).text_id.get_font_size();
    fan::string measured_string;
    for (uint32_t i = 0; i < width; ++i) {
      measured_string += text.get_utf8(i);
    }
    p.x += gloco->text.get_text_size(measured_string, font_size).x;
    p.y = get_button(id, &text_box_t::vi_t::position).y;
    return p;
  }

  fan::ev_timer_t::timer_t timer = fan::function_t<void(const fan::ev_timer_t::cb_data_t&)>([this](const fan::ev_timer_t::cb_data_t& c) {
    if (!render_cursor) {
      cursor.set_color(fan::colors::transparent);
    }
    else {
      cursor.set_color(cursor_properties::color);
    }
    render_cursor = !render_cursor;
    c.ev_timer->start(c.timer, cursor_properties::speed);
  });
  loco_t::id_t cursor;
  bool render_cursor = true;

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
    update_cursor(id);
  }

  void invalidate_focus(loco_t::cid_nt_t& id) {
    auto& ri = sb_get_ri(id);
    if (ri.vfi_id == gloco->vfi.get_focus_mouse()) {
      gloco->vfi.invalidate_focus_mouse();
    }
    if (ri.vfi_id == gloco->vfi.get_focus_keyboard()) {
      gloco->vfi.invalidate_focus_keyboard();
    }
    if (ri.vfi_id == gloco->vfi.get_focus_text()) {
      gloco->vfi.invalidate_focus_text();
    }
  }

  // dont edit values
  loco_t::text_t::properties_t get_text_instance(loco_t::cid_nt_t& id) {
    auto& ri = sb_get_ri(id);
    return gloco->text.get_instance(ri.text_id);
  }

  // can be incomplete
  properties_t get_properties(loco_t::cid_nt_t& id) {
    properties_t p = sb_get_properties(id);
    p.camera = gloco->camera_list[*p.key.get_value<loco_t::camera_list_NodeReference_t>()].camera_id;
    //p.theme =  gloco->get_context()->theme_list[*p.key.get_values<loco_t::t>()].matrices_id;
    p.viewport = gloco->get_context()->viewport_list[*p.key.get_value<fan::graphics::viewport_list_NodeReference_t>()].viewport_id;
    p.position = get_text_instance(id).position;
    p.text = get_text_instance(id).text;
    return p;
  }

  #if defined(loco_vulkan)
    uint32_t m_camera_index = 0;
  #endif
};

#include _FAN_PATH(graphics/opengl/2D/objects/hardcode_close.h)