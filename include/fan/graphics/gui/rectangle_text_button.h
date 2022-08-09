struct rectangle_text_button_t {

  using be_t = fan_2d::graphics::gui::be_t;

  struct properties_t : loco_t::text_box_t::properties_t {
    be_t::on_input_cb_t mouse_input_cb = [](const be_t::mouse_input_data_t&) -> uint8_t {return 1; };
    be_t::on_mouse_move_cb_t mouse_move_cb = [](const be_t::mouse_move_data_t&) -> uint8_t { return 1; };

    void* userptr;
  };

protected:

  static void lib_set_theme(
    loco_t* loco,
    fan::opengl::cid_t* cid,
    f32_t intensity
    ) {
    loco->text_box.set_theme(loco, 0, cid, &(*loco->text_box.get_theme(loco, cid) * intensity));
  }

#define make_code_small_plis(d_n, i) lib_set_theme( \
  (loco_t*)d_n.userptr[0], \
    (fan::opengl::cid_t*)d_n.element_id, \
    i \
  );

  static uint8_t mouse_move_cb(const be_t::mouse_move_data_t& mm_data) {
    switch (mm_data.mouse_stage) {
      case fan_2d::graphics::gui::mouse_stage_e::inside: {
        make_code_small_plis(mm_data, 1.1);
        break;
      }
      case fan_2d::graphics::gui::mouse_stage_e::outside: {
        make_code_small_plis(mm_data, 1.0 / 1.1);
        break;
      }
    }
    return 1;
  }
  static uint8_t mouse_input_cb(const be_t::mouse_input_data_t& ii_data) {
    if (ii_data.key != fan::mouse_left) {
      return 1;
    }
    switch (ii_data.mouse_stage) {
      case fan_2d::graphics::gui::mouse_stage_e::inside: {
        switch (ii_data.key_state) {
          case fan::key_state::press: {
            make_code_small_plis(ii_data, 1.2);
            break;
          }
          case fan::key_state::release: {
            make_code_small_plis(ii_data, 1.1);
            break;
          }
        }
        break;
      }
      case fan_2d::graphics::gui::mouse_stage_e::outside: {
        make_code_small_plis(ii_data, 1.0 / 1.2);
        break;
      }
    }
    return 1;
  }

#undef make_code_small_plis

public:

  void open(loco_t* loco)
  {
    list.open();
    e.amount = 0;
  }

  void close(loco_t* loco)
  {
    list.close();
  }

  void push_back(loco_t* loco, fan::opengl::cid_t* cid, fan_2d::graphics::gui::be_t* button_event, properties_t& p) {
    uint32_t id;
    if (e.amount != 0) {
      id = e.id;
      e.id = *(uint32_t*)&list[e.id];
      e.amount--;
    }
    else {
      id = list.resize(list.size() + 1);
    }
    loco->text_box.push_back(loco, cid, p);

    fan_2d::graphics::gui::be_t::properties_t be_p;
    be_p.hitbox_type = fan_2d::graphics::gui::be_t::hitbox_type_t::rectangle;
    be_p.hitbox_rectangle.position = p.position;
    be_p.hitbox_rectangle.size = p.size;
    be_p.on_input_function = p.mouse_input_cb;
    be_p.on_mouse_event_function = p.mouse_move_cb;
    be_p.userptr[0] = loco;
    be_p.userptr[2] = p.userptr;
    be_p.cid = cid;
    list[id].button_event_id = button_event->push_back(be_p, mouse_input_cb, mouse_move_cb);
  }

  void set_theme(loco_t* loco, fan::opengl::cid_t* cid, fan_2d::graphics::gui::theme_t* theme) {
    // TODO FIX 0
    loco->text_box.set_theme(loco, 0, cid, theme);
  }

  struct{
    uint32_t id;
    uint32_t amount;
  }e;

  struct element_t {
    uint32_t button_event_id;
  };

  fan::hector_t<element_t> list;
};