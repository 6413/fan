struct rectangle_text_box_t {

  using instance_t = loco_t::box_t::instance_t;

  struct properties_t : loco_t::box_t::properties_t {

    fan::utf16_string text;
    f32_t font_size = 0.1;
  };

  static constexpr uint32_t max_instance_size = 256;

  void open(loco_t* loco) {

  }
  void close(loco_t* loco) {

  }

  void push_back(loco_t* loco, fan::opengl::cid_t* cid, properties_t& p) {
    loco->box.push_back(loco, cid, p);
    auto theme = loco->box.get_theme(loco, p.theme);
    loco_t::text_t::properties_t tp;
    tp.color = theme->button.text_color;
    tp.font_size = p.font_size;
    tp.position = p.position;
    tp.text = p.text;
    tp.position.z += p.position.z + 0.001;
    // do something with id xd
    uint32_t id = loco->text.push_back(loco, tp);

    set_theme(loco, id, cid, theme);
  }
  void erase(loco_t* loco, fan::opengl::cid_t* cid) {
    loco->box.erase(loco, cid);
    //  erase text rendere here
  }

  template <typename T>
  T get_box(loco_t* loco, fan::opengl::cid_t* cid, T instance_t::* member) {
    return loco->box.get(loco, cid, member);
  }
  template <typename T, typename T2>
  void set_box(loco_t* loco, fan::opengl::cid_t* cid, T instance_t::*member, const T2& value) {
    loco->box.set(loco, cid, member, value);
  }

  template <typename T>
  T get_text(loco_t* loco, uint32_t id, T loco_t::letter_t::instance_t::* member) {
    return loco->text.get(loco, id, member);
  }
  template <typename T, typename T2>
  void set_text(loco_t* loco, uint32_t id, T loco_t::letter_t::instance_t::*member, const T2& value) {
    loco->text.set(loco, id, member, value);
  }
  fan_2d::graphics::gui::theme_t* get_theme(loco_t* loco, fan::opengl::cid_t* cid) {
    return loco->box.get_theme(loco, cid);
  }

  void set_theme(loco_t* loco, uint32_t text_id, fan::opengl::cid_t* cid, fan_2d::graphics::gui::theme_t* theme) {
    loco->box.set_theme(loco, cid, theme);
    //set(loco, text_id, &loco_t::letter_t::instance_t::outline_color, theme.button.text_outline_color);
    //set(loco, text_id, &loco_t::letter_t::instance_t::outline_size, theme.button.text_outline_color);
  }
};