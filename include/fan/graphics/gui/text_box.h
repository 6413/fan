struct text_box_t {

  struct cursor_properties {
    static constexpr uint64_t speed = 5e+8;
    static constexpr fan::vec2 size = fan::vec2(0.002, 0.015);
    static constexpr fan::color color = fan::colors::white;
  };

  static constexpr f32_t inactive = 1.0;
  static constexpr f32_t hover = 1.2;
  static constexpr f32_t press = 1.4;

  struct instance_t {
    fan::vec3 position = 0;
    f32_t angle = 0;
    fan::vec2 size = 0;
    fan::vec2 rotation_point = 0;
    fan::color color = fan::colors::white;
    fan::color outline_color;
    fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
    f32_t outline_size;
  };

  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(instance_t) / 4));

  #define hardcode0_t fan::opengl::matrices_list_NodeReference_t
  #define hardcode0_n matrices
  #define hardcode1_t fan::opengl::viewport_list_NodeReference_t
  #define hardcode1_n viewport
  #include _FAN_PATH(graphics/opengl/2D/objects/hardcode_open.h)

  struct instance_properties_t {
    struct key_t : parsed_masterpiece_t {}key;

    expand_get_functions

    uint8_t selected;
    fan::opengl::theme_list_NodeReference_t theme;
    loco_t::vfi_t::shape_id_t vfi_id;
    uint64_t udata;

    uint32_t text_id;
    fed_t fed;
  };

  struct properties_t : instance_t, instance_properties_t {
    properties_t() {
      selected = 0;
    }

    fan::wstring text;
    f32_t font_size = 0.1;

    loco_t::vfi_t::iflags_t vfi_flags;

    bool disable_highlight = false;

    loco_t::mouse_button_cb_t mouse_button_cb = [](const loco_t::mouse_button_data_t&) -> int { return 0; };
    loco_t::mouse_move_cb_t mouse_move_cb = [](const loco_t::mouse_move_data_t&) -> int { return 0; };
    loco_t::keyboard_cb_t keyboard_cb = [](const loco_t::keyboard_data_t&) -> int { return 0; };
    loco_t::text_cb_t text_cb = [](const loco_t::text_data_t&) -> int { return 0; };
  };

  void push_back(fan::opengl::cid_t* cid, properties_t& p) {
    loco_t* loco = get_loco();
    auto theme = loco->text_box.get_theme(p.theme);
    loco_t::text_t::properties_t tp;
    tp.color = theme->button.text_color;
    tp.font_size = p.font_size;
    tp.position = p.position;
    tp.text = p.text;
    tp.position.z += p.position.z + 0.5;
    tp.get_viewport() = p.get_viewport();
    tp.get_matrices() = p.get_matrices();
    auto block = sb_push_back(cid, p);
    auto& pr = block->p[cid->instance_id];

    pr.text_id = loco->text.push_back(tp);

    set_theme(cid, theme, inactive);

    loco_t::vfi_t::properties_t vfip;
    vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
    vfip.shape.rectangle.matrices = p.get_matrices();
    vfip.shape.rectangle.viewport = p.get_viewport();
    vfip.shape.rectangle.position = p.position;
    vfip.shape.rectangle.size = p.size;
    vfip.flags = p.vfi_flags;
    fed_t::properties_t fp;
    fp.font_size = p.font_size;
    fp.loco = loco;
    pr.fed.open(fp);
    pr.fed.push_text(p.text);
    if (!p.disable_highlight) {
      vfip.mouse_move_cb = [this, &pr, cb = p.mouse_move_cb, udata = p.udata, cid_ = cid](const loco_t::vfi_t::mouse_move_data_t& mm_d) -> int {
        loco_t* loco = OFFSETLESS(mm_d.vfi, loco_t, vfi_var_name);
        loco_t::mouse_move_data_t mmd = mm_d;
        auto block = loco->text_box.sb_get_block(cid_);
        if (mm_d.flag->ignore_move_focus_check == false && !block->p[cid_->instance_id].selected) {
          if (mm_d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
            loco->text_box.set_theme(cid_, loco->text_box.get_theme(cid_), hover);
          }
          else {
            loco->text_box.set_theme(cid_, loco->text_box.get_theme(cid_), inactive);
          }
        }

        //if (loco->get_window()->key_pressed(fan::mouse_left) && loco->vfi.get_focus_keyboard()) {
        //  fan::print("a");
        //  // src press
        //  fan::vec2 src = fan::vec2(mm_d.position) - fan::vec2(get_text_left_position(cid_));
        //  // dst release
        //  src.x = fan::clamp(src.x, (f32_t)0, src.x);
        //  fan::vec2 dst = src;

        //  pr.fed.set_mouse_position(src, dst);
        //  update_cursor(cid_);
        //}

        mmd.cid = cid_;
        cb(mmd);
        return 0;
      };
      vfip.mouse_button_cb = [this, &pr, cb = p.mouse_button_cb, udata = p.udata, cid_ = cid](const loco_t::vfi_t::mouse_button_data_t& ii_d) -> int {
        loco_t* loco = OFFSETLESS(ii_d.vfi, loco_t, vfi_var_name);
        auto block = loco->text_box.sb_get_block(cid_);
        if (ii_d.flag->ignore_move_focus_check == false && !block->p[cid_->instance_id].selected) {
          if (ii_d.button == fan::mouse_left && ii_d.button_state == fan::mouse_state::press) {
            loco->text_box.set_theme(cid_, loco->text_box.get_theme(cid_), press);
            ii_d.flag->ignore_move_focus_check = true;
            loco->vfi.set_focus_keyboard(loco->vfi.get_focus_mouse());
            loco->vfi.set_focus_text(loco->vfi.get_focus_mouse());

            fan::vec2 src = fan::vec2(ii_d.position) - fan::vec2(get_text_left_position(cid_));
            // dst release
            src.x = fan::clamp(src.x, (f32_t)0, src.x);
            fan::vec2 dst = src;

            pr.fed.set_mouse_position(src, dst);
            update_cursor(cid_);
          }
        }
        else if (!block->p[cid_->instance_id].selected) {
          if (loco->ev_timer.time_list.find(&timer) == loco->ev_timer.time_list.end()) {
            loco->ev_timer.start(&timer, cursor_properties::speed);
            update_cursor(cid_);
          }

          if (ii_d.button == fan::mouse_left && ii_d.button_state == fan::mouse_state::release) {
            if (ii_d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
              loco->text_box.set_theme(cid_, loco->text_box.get_theme(cid_), hover);
            }
            else {
              loco->text_box.set_theme(cid_, loco->text_box.get_theme(cid_), inactive);
            }
            ii_d.flag->ignore_move_focus_check = false;
          }
        }

        loco_t::mouse_button_data_t mid = ii_d;
        mid.cid = cid_;
        cb(mid);

        return 0;
      };
      vfip.keyboard_cb = [this, &pr, cb = p.keyboard_cb, udata = p.udata, cid_ = cid](const loco_t::vfi_t::keyboard_data_t& kd) -> int {
        loco_t* loco = OFFSETLESS(kd.vfi, loco_t, vfi_var_name);

        auto update_text = [loco, &pr, this, cid_] {
          wed_t::CursorInformation_t ci;
          auto& fed = sb_get_block(cid_)->p[cid_->instance_id].fed;
          fed.m_wed.GetCursorInformation(fed.m_cr, &ci);
          switch (ci.type) {
            case wed_t::CursorType::FreeStyle: {
              loco->text_box.set_text(cid_, pr.fed.get_text(ci.FreeStyle.LineReference));
              break;
            }
            case wed_t::CursorType::Selection: {
              assert(0);
              break;
            }
          }
        };

        if (kd.keyboard_state != fan::keyboard_state::release) {
          switch (kd.key) {
            case fan::key_backspace: { pr.fed.freestyle_erase_character(); update_text();  break; }
            case fan::key_delete: { pr.fed.freestyle_erase_character_right(); update_text(); break; }
            case fan::key_home:  { pr.fed.freestyle_move_line_begin(); break; }
            case fan::key_end: { pr.fed.freestyle_move_line_end(); break; }
            case fan::key_left: { pr.fed.freestyle_move_left(); break; }
            case fan::key_right: { pr.fed.freestyle_move_right(); break; }
            case fan::key_v: {
              if (loco->get_window()->key_pressed(fan::key_control)) {
                auto pasted_text = fan::io::get_clipboard_text(loco->get_window()->get_handle());

                pr.fed.push_text(pasted_text);

                update_text();
              }
            }
            default: {
              return 0;
            }
          }
        }

        update_cursor(cid_);

        loco_t::keyboard_data_t kd_ = kd;
        auto block = loco->text_box.sb_get_block(cid_);
        kd_.cid = cid_;
        cb(kd_);
        return 0;
      };
      vfip.text_cb = [this, &pr, cb = p.text_cb, udata = p.udata, cid_ = cid](const loco_t::vfi_t::text_data_t& td) -> int {
        loco_t* loco = OFFSETLESS(td.vfi, loco_t, vfi_var_name);

        switch (td.key) {
          default: {
            pr.fed.add_character(td.key);
            wed_t::CursorInformation_t ci;
            auto& fed = sb_get_block(cid_)->p[cid_->instance_id].fed;
            fed.m_wed.GetCursorInformation(fed.m_cr, &ci);
            switch (ci.type) {
              case wed_t::CursorType::FreeStyle: {
                loco->text_box.set_text(cid_, pr.fed.get_text(ci.FreeStyle.LineReference));
                break;
              }
              case wed_t::CursorType::Selection: {
                assert(0);
                break;
              }
            }
            update_cursor(cid_);
            break;
          }
        }

        loco_t::text_data_t td_ = td;
        auto block = loco->text_box.sb_get_block(cid_);
        td_.cid = cid_;
        cb(td_);
        return 0;
      };
    }

    pr.vfi_id = loco->vfi.push_shape(vfip);

    loco_t::rectangle_t::properties_t rp;
    rp.position.z = tp.position.z;
    rp.size = cursor_properties::size;
    rp.size.y = p.font_size;
    rp.get_matrices() = p.get_matrices();
    rp.get_viewport() = p.get_viewport();
    rp.color = fan::colors::transparent;
    loco->rectangle.push_back(&cursor_id, rp);
  }
  void erase(fan::opengl::cid_t* cid) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    instance_properties_t* p = &block->p[cid->instance_id];
    loco->text.erase(p->text_id);
    loco->vfi.erase(block->p[cid->instance_id].vfi_id);
    sb_erase(cid);
  }

  void update_cursor(fan::opengl::cid_t* cid) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    wed_t::CursorInformation_t ci;
    auto& fed = block->p[cid->instance_id].fed;
    fed.m_wed.GetCursorInformation(fed.m_cr, &ci);
    fan::vec3 p = loco->rectangle.get(&cursor_id, &loco_t::rectangle_t::instance_t::position);
    switch (ci.type) {
      case wed_t::CursorType::FreeStyle: {
        uint32_t line_index = fed.m_wed.GetLineIndexByLineReference(ci.FreeStyle.LineReference);
        uint32_t character_index = fed.m_wed.GetCharacterIndexByCharacterReference(
          ci.FreeStyle.LineReference,
          ci.FreeStyle.CharacterReference
        );
        p = get_character_position(cid, line_index, character_index);
        break;
      }
      case wed_t::CursorType::Selection: {
        assert(0);
        //m_wed.GetLineIndexByLineReference(ci.Selection.LineReference);
        break;
      }
    }
    loco->rectangle.set(&cursor_id, &loco_t::rectangle_t::instance_t::position, p);
    loco->ev_timer.stop(&timer);
    render_cursor = true;
    fan::ev_timer_t::cb_data_t d;
    d.ev_timer = &loco->ev_timer;
    d.timer = &timer;
    timer.cb(d);
  }

  instance_properties_t* get_instance_properties(fan::opengl::cid_t* cid) {
    return &sb_get_block(cid)->p[cid->instance_id];
  }

  void draw() {
    sb_draw();
  }

  #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle_box.vs)
  #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle_box.fs)
  #include _FAN_PATH(graphics/opengl/2D/objects/shape_builder.h)

  void open() {
    sb_open();
  }
  void close() {
    // check erase, need to somehow iterate block
    assert(0);
    sb_close();
  }

  fan_2d::graphics::gui::theme_t* get_theme(fan::opengl::theme_list_NodeReference_t nr) {
    loco_t* loco = get_loco();
    return loco->get_context()->theme_list[nr].theme_id;
  }
  fan_2d::graphics::gui::theme_t* get_theme(fan::opengl::cid_t* cid) {
    return get_theme(blocks[*(bll_block_NodeReference_t*)&cid->block_id].block.p[cid->instance_id].theme);
  }
  void set_theme(fan::opengl::cid_t* cid, fan_2d::graphics::gui::theme_t* theme, f32_t intensity) {
    loco_t* loco = get_loco();
    fan_2d::graphics::gui::theme_t t = *theme;
    t = t * intensity;
    set(cid, &instance_t::color, t.button.color);
    set(cid, &instance_t::outline_color, t.button.outline_color);
    set(cid, &instance_t::outline_size, t.button.outline_size);
    auto block = sb_get_block(cid);
    block->p[cid->instance_id].theme = theme;
    loco->text.set(block->p[cid->instance_id].text_id,
      &loco_t::letter_t::instance_t::outline_color, t.button.text_outline_color);
    loco->text.set(block->p[cid->instance_id].text_id,
      &loco_t::letter_t::instance_t::outline_size, t.button.text_outline_size);
  }

  template <typename T>
  T get_button(fan::opengl::cid_t* cid, T instance_t::* member) {
    loco_t* loco = get_loco();
    return loco->text_box.get(cid, member);
  }
  template <typename T, typename T2>
  void set_button(fan::opengl::cid_t* cid, T instance_t::* member, const T2& value) {
    loco_t* loco = get_loco();
    loco->text_box.set(cid, member, value);
  }

  template <typename T>
  T get_text_renderer(fan::opengl::cid_t* cid, T loco_t::letter_t::instance_t::* member) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    return loco->text.get(block->p[cid->instance_id].text_id, member);
  }
  template <typename T, typename T2>
  void set_text_renderer(fan::opengl::cid_t* cid, T loco_t::letter_t::instance_t::* member, const T2& value) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    loco->text.set(block->p[cid->instance_id].text_id, member, value);
  }

  void set_position(fan::opengl::cid_t* cid, const fan::vec3& position) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    loco->text.set_position(&block->p[cid->instance_id].text_id, position + fan::vec3(0, 0, 0.5));
    set_button(cid, &instance_t::position, position);
    loco->vfi.set_rectangle(
      block->p[cid->instance_id].vfi_id,
      &loco_t::vfi_t::set_rectangle_t::position,
      position
    );
  }
  void set_size(fan::opengl::cid_t* cid, const fan::vec3& size) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    set_button(cid, &instance_t::size, size);
    loco->vfi.set_rectangle(
      block->p[cid->instance_id].vfi_id,
      &loco_t::vfi_t::set_rectangle_t::size,
      size
    );
  }

  fan::opengl::matrices_t* get_matrices(fan::opengl::cid_t* cid) {
    auto block = sb_get_block(cid);
    loco_t* loco = get_loco();
    return loco->get_context()->matrices_list[*block->p[cid->instance_id].key.get_value<0>()].matrices_id;
  }
  void set_matrices(fan::opengl::cid_t* cid, fan::opengl::matrices_list_NodeReference_t n) {
    sb_set_key<instance_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    loco->text.set_matrices(block->p[cid->instance_id].text_id, n);
  }

  fan::opengl::viewport_t* get_viewport(fan::opengl::cid_t* cid) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    return loco->get_context()->viewport_list[*block->p[cid->instance_id].key.get_value<1>()].viewport_id;
  }
  void set_viewport(fan::opengl::cid_t* cid, fan::opengl::viewport_list_NodeReference_t n) {
    loco_t* loco = get_loco();
    sb_set_key<instance_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
    auto block = sb_get_block(cid);
    loco->text.set_viewport(block->p[cid->instance_id].text_id, n);
  }

  void set_theme(fan::opengl::cid_t* cid, f32_t state) {
    loco_t* loco = get_loco();
    loco->text_box.set_theme(cid, loco->text_box.get_theme(cid), state);
  }

  // gets udata from current focus
/*uint64_t get_id_udata(loco_t::vfi_t::shape_id_t id) {
  loco_t* loco = get_loco();
  auto udata = loco->vfi.get_id_udata(id);
  fan::opengl::cid_t* cid = (fan::opengl::cid_t*)udata;
  auto block = sb_get_block(cid);
  return block->p[cid->instance_id].udata;
}*/

  void set_selected(fan::opengl::cid_t* cid, bool flag) {
    auto block = sb_get_block(cid);
    block->p[cid->instance_id].selected = flag;
  }

  fan::wstring get_text(fan::opengl::cid_t* cid) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    return loco->text.get_properties(block->p[cid->instance_id].text_id).text;
  }
  void set_text(fan::opengl::cid_t* cid, const fan::wstring& text) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    loco->text.set_text(&block->p[cid->instance_id].text_id, text);
  }

  fan::vec3 get_text_left_position(fan::opengl::cid_t* cid) {
    auto loco = get_loco();
    uint32_t id = sb_get_block(cid)->p[cid->instance_id].text_id;
    f32_t text_length = loco->text.get_text_size(id).x;
    fan::vec3 center = get_button(cid, &text_box_t::instance_t::position);
    center.x -= text_length * 0.5;
    return center;
  }

  fan::vec3 get_character_position(fan::opengl::cid_t* cid, uint32_t line, uint32_t width) {

    auto loco = get_loco();
    uint32_t id = sb_get_block(cid)->p[cid->instance_id].text_id;
    fan::vec3 center = get_button(cid, &text_box_t::instance_t::position);
    if (width == 0) {
      if (loco->text.get_properties(id).text.empty()) {
        return center;
      }
    }

    fan::vec3 p = get_text_left_position(cid);
    const fan::wstring& text = loco->text.get_properties(id).text;
    f32_t font_size = loco->text.letter_ids[id].p.font_size;
    for (uint32_t i = 0; i < width; ++i) {
      auto letter = loco->font.info.get_letter_info(loco->font.decode_letter(text[i]), font_size);
      p.x += letter.metrics.advance;
    }
    p.y = get_button(cid, &text_box_t::instance_t::position).y;
    return p;
  }

  fan::ev_timer_t::timer_t timer = fan::function_t<void(const fan::ev_timer_t::cb_data_t&)>([this](const fan::ev_timer_t::cb_data_t& c) {
    if (!render_cursor) {
      get_loco()->rectangle.set(&cursor_id, &loco_t::rectangle_t::instance_t::color, fan::colors::transparent);
    }
    else {
      get_loco()->rectangle.set(&cursor_id, &loco_t::rectangle_t::instance_t::color, cursor_properties::color);
    }
    render_cursor = !render_cursor;
    c.ev_timer->start(c.timer, cursor_properties::speed);
  });
  fan::opengl::cid_t cursor_id;
  bool render_cursor = true;
};

#include _FAN_PATH(graphics/opengl/2D/objects/hardcode_close.h)