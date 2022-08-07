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

    loco_t::text_t::properties_t tp;
    tp.color = p.theme.button.text_color;
    tp.font_size = p.font_size;
    tp.position = p.position;
    tp.text = p.text;
    tp.position.z += p.position.z + 0.001;
    // do something with id xd
    uint32_t id = loco->text.push_back(loco, tp);

    set_theme(loco, id, cid, p.theme);
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

  void set_theme(loco_t* loco, uint32_t text_id, fan::opengl::cid_t* cid, const fan_2d::graphics::gui::theme& theme) {
    set_box(loco, cid, &instance_t::color, theme.button.color);
    set_box(loco, cid, &instance_t::outline_color, theme.button.outline_color);
    set_box(loco, cid, &instance_t::outline_size, theme.button.outline_size);
    set_text(loco, text_id, &loco_t::letter_t::instance_t::outline_color, theme.button.text_outline_color);
    set_text(loco, text_id, &loco_t::letter_t::instance_t::outline_size, theme.button.text_outline_color);
  }
};