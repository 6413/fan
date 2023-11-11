#if defined(loco_line)

struct line_properties_t {
  fan::graphics::camera_t* camera = gloco->default_camera;
  fan::vec3 src = fan::vec3(0, 0, 0);
  fan::vec2 dst = fan::vec2(1, 1);
  fan::color color = fan::color(1, 1, 1, 1);
  bool blending = false;
};

struct line_t : loco_t::shape_t {
  line_t(line_properties_t p = line_properties_t()) {
    *(loco_t::shape_t*)this = loco_t::shape_t(
      fan_init_struct(
        typename loco_t::shapes_t::line_t::properties_t,
        .camera = &p.camera->camera,
        .viewport = &p.camera->viewport,
        .src = p.src,
        .dst = p.dst,
        .color = p.color,
        .blending = p.blending
      ));
  }
};
#endif

#if defined(loco_rectangle)
struct rectangle_properties_t {
  fan::graphics::camera_t* camera = gloco->default_camera;
  fan::vec3 position = fan::vec3(0, 0, 0);
  fan::vec2 size = fan::vec2(0.1, 0.1);
  fan::color color = fan::color(1, 1, 1, 1);
  bool blending = false;
};

// make sure you dont do position = vec2
struct rectangle_t : loco_t::shape_t {
  rectangle_t(rectangle_properties_t p = rectangle_properties_t()) {
    *(loco_t::shape_t*)this = loco_t::shape_t(
      fan_init_struct(
        typename loco_t::shapes_t::rectangle_t::properties_t,
        .camera = &p.camera->camera,
        .viewport = &p.camera->viewport,
        .position = p.position,
        .size = p.size,
        .color = p.color,
        .blending = p.blending
      ));
  }
};
#endif

#if defined(loco_circle)
struct circle_properties_t {
  fan::graphics::camera_t* camera = gloco->default_camera;
  fan::vec3 position = fan::vec3(0, 0, 0);
  f32_t radius = 0.1;
  fan::color color = fan::color(1, 1, 1, 1);
  bool blending = false;
};

struct circle_t : loco_t::shape_t {
  circle_t(circle_properties_t p = circle_properties_t()) {
    *(loco_t::shape_t*)this = loco_t::shape_t(
      fan_init_struct(
        typename loco_t::shapes_t::circle_t::properties_t,
        .camera = &p.camera->camera,
        .viewport = &p.camera->viewport,
        .position = p.position,
        .radius = p.radius,
        .color = p.color,
        .blending = p.blending
      ));
  }
};
#endif


#if defined(loco_sprite)
struct unlit_sprite_properties_t {
  fan::graphics::camera_t* camera = gloco->default_camera;
  fan::vec3 position = fan::vec3(0, 0, 0);
  fan::vec2 size = fan::vec2(0.1, 0.1);
  fan::color color = fan::color(1, 1, 1, 1);
  loco_t::image_t* image = &gloco->default_texture;
  bool blending = false;
  fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
};

struct unlit_sprite_t : loco_t::shape_t {
  unlit_sprite_t(unlit_sprite_properties_t p = unlit_sprite_properties_t()) {
    *(loco_t::shape_t*)this = loco_t::shape_t(
      fan_init_struct(
        typename loco_t::shapes_t::unlit_sprite_t::properties_t,
        .camera = &p.camera->camera,
        .viewport = &p.camera->viewport,
        .position = p.position,
        .size = p.size,
        .image = p.image,
        .color = p.color,
        .blending = p.blending,
        .rotation_vector = p.rotation_vector
      ));
  }
};

struct sprite_properties_t {
  fan::graphics::camera_t* camera = gloco->default_camera;
  fan::vec3 position = fan::vec3(0, 0, 0);
  fan::vec2 size = fan::vec2(0.1, 0.1);
  fan::color color = fan::color(1, 1, 1, 1);
  loco_t::image_t* image = &gloco->default_texture;
  bool blending = false;
  fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
};

struct sprite_t : loco_t::shape_t {
  sprite_t(sprite_properties_t p = sprite_properties_t()) {
    *(loco_t::shape_t*)this = loco_t::shape_t(
      fan_init_struct(
        typename loco_t::shapes_t::sprite_t::properties_t,
        .camera = &p.camera->camera,
        .viewport = &p.camera->viewport,
        .position = p.position,
        .size = p.size,
        .image = p.image,
        .color = p.color,
        .blending = p.blending,
        .rotation_vector = p.rotation_vector
      ));
  }
};

#endif

#if defined(loco_text)
struct letter_properties_t {
  loco_t::camera_t* camera = &gloco->default_camera->camera;
  loco_t::viewport_t* viewport = &gloco->default_camera->viewport;
  fan::color color = fan::colors::white;
  fan::vec3 position = fan::vec3(0, 0, 0);
  f32_t font_size = 1;
  uint32_t letter_id;
};

struct letter_t : loco_t::shape_t {
  letter_t(letter_properties_t p = letter_properties_t()) {
    *(loco_t::shape_t*)this = loco_t::shape_t(
      fan_init_struct(
        typename loco_t::shapes_t::letter_t::properties_t,
        .camera = p.camera,
        .viewport = p.viewport,
        .position = p.position,
        .font_size = p.font_size,
        .letter_id = p.letter_id,
        .color = p.color
      ));
  }
};

struct text_properties_t {
  loco_t::camera_t* camera = &gloco->default_camera->camera;
  loco_t::viewport_t* viewport = &gloco->default_camera->viewport;
  std::string text = "";
  fan::color color = fan::colors::white;
  fan::vec3 position = fan::vec3(fan::math::inf, -0.9, 0);
};

struct text_t : loco_t::shape_t {
  text_t(text_properties_t p = text_properties_t()) {
    *(loco_t::shape_t*)this = loco_t::shape_t(
      fan_init_struct(
        typename loco_t::shapes_t::responsive_text_t::properties_t,
        .camera = p.camera,
        .viewport = p.viewport,
        .position = p.position.x == fan::math::inf ? fan::vec3(-1 + 0.025 * p.text.size(), -0.9, 0) : p.position,
        .text = p.text,
        .line_limit = 1,
        .letter_size_y_multipler = 1,
        .size = fan::vec2(0.025 * p.text.size(), 0.1),
        .color = p.color
      ));
  }
};
#endif

#if defined(loco_button)
struct button_properties_t {
  loco_t::theme_t* theme = &gloco->default_theme;
  loco_t::camera_t* camera = &gloco->default_camera->camera;
  loco_t::viewport_t* viewport = &gloco->default_camera->viewport;
  fan::vec3 position = fan::vec3(0, 0, 0);
  fan::vec2 size = fan::vec2(0.1, 0.1);
  std::string text = "button";
  loco_t::mouse_move_cb_t mouse_move_cb = [](const loco_t::mouse_move_data_t&) -> int { return 0; };
  loco_t::mouse_button_cb_t mouse_button_cb = [](const loco_t::mouse_button_data_t&) -> int { return 0; };
};

struct button_t : loco_t::shape_t {
  button_t(button_properties_t p = button_properties_t()) : loco_t::shape_t(
    fan_init_struct(
      typename loco_t::shapes_t::button_t::properties_t,
      .theme = p.theme,
      .camera = p.camera,
      .viewport = p.viewport,
      .position = p.position,
      .size = p.size,
      .text = p.text,
      .mouse_button_cb = p.mouse_button_cb
    )) {}
};
#endif

#if defined(loco_imgui)
struct imgui_element_nr_t : loco_t::imgui_draw_cb_nr_t {
  using base_t = loco_t::imgui_draw_cb_nr_t;

  imgui_element_nr_t() { /**(nr_t*)this = cid_list_gnric(); should be default*/ }

  imgui_element_nr_t(const imgui_element_nr_t& nr) : imgui_element_nr_t() {
    if (nr.is_invalid()) {
      return;
    }
    init();
  }

  imgui_element_nr_t(imgui_element_nr_t&& nr) {
    NRI = nr.NRI;
    nr.invalidate_soft();
  }
  ~imgui_element_nr_t() {
    invalidate();
  }


  imgui_element_nr_t& operator=(const imgui_element_nr_t& id) {
    if (!is_invalid()) {
      invalidate();
    }
    if (id.is_invalid()) {
      return *this;
    }

    if (this != &id) {
      init();
    }
    return *this;
  }

  imgui_element_nr_t& operator=(imgui_element_nr_t&& id) {
    if (!is_invalid()) {
      invalidate();
    }
    if (id.is_invalid()) {
      return *this;
    }

    if (this != &id) {
      if (!is_invalid()) {
        invalidate();
      }
      NRI = id.NRI;

      id.invalidate_soft();
    }
    return *this;
  }

  void init() {
    *(base_t*)this = gloco->m_imgui_draw_cb.NewNodeLast();
  }

  bool is_invalid() const {
    return loco_t::imgui_draw_cb_inric(*this);
  }

  void invalidate_soft() {
    *(base_t*)this = gloco->m_imgui_draw_cb.gnric();
  }

  void invalidate() {
    if (is_invalid()) {
      return;
    }
    gloco->m_imgui_draw_cb.unlrec(*this);
    *(base_t*)this = gloco->m_imgui_draw_cb.gnric();
  }

  void set(const auto& lambda) {
    gloco->m_imgui_draw_cb[*this] = lambda;
  }
};

struct imgui_element_t : imgui_element_nr_t {
  imgui_element_t() = default;
  imgui_element_t(const auto& lambda) {
    imgui_element_nr_t::init();
    imgui_element_nr_t::set(lambda);
  }
};

struct imgui_shape_element_t : imgui_element_t, loco_t::shape_t {
  imgui_shape_element_t(const auto& properties, const auto& lambda)
    : imgui_element_t(lambda), loco_t::shape_t(properties) {
  }
};
#endif

#if defined(loco_vfi)

// REQUIRES to be allocated by new since lambda captures this
// also container that it's stored in, must not change pointers
template <typename T>
struct vfi_root_custom_t {
  void set_root(const loco_t::shapes_t::vfi_t::properties_t& p) {
    loco_t::shapes_t::vfi_t::properties_t in = p;
    in.shape_type = loco_t::shapes_t::vfi_t::shape_t::rectangle;
    in.shape.rectangle->viewport = &gloco->default_camera->viewport;
    in.shape.rectangle->camera = &gloco->default_camera->camera;
    in.keyboard_cb = [this, user_cb = p.keyboard_cb](const auto& d) -> int {
      if (d.key == fan::key_c &&
        (d.keyboard_state == fan::keyboard_state::press ||
          d.keyboard_state == fan::keyboard_state::repeat)) {
        this->resize = true;
        return user_cb(d);
      }
      this->resize = false;
      return 0;
      };
    in.mouse_button_cb = [this, user_cb = p.mouse_button_cb](const auto& d) -> int {
      if (d.button != fan::mouse_left) {
        return 0;
      }
      if (d.button_state != fan::mouse_state::press) {
        this->move = false;
        d.flag->ignore_move_focus_check = false;
        return 0;
      }
      if (d.mouse_stage != loco_t::shapes_t::vfi_t::mouse_stage_e::inside) {
        return 0;
      }
      
      if (move_and_resize_auto) {
        d.flag->ignore_move_focus_check = true;
        this->move = true;
        this->click_offset = get_position() - d.position;
        gloco->shapes.vfi.set_focus_keyboard(d.vfi->focus.mouse);
      }
      return user_cb(d);
    };
    in.mouse_move_cb = [this, user_cb = p.mouse_move_cb](const auto& d) -> int {
      if (move_and_resize_auto) {
        if (this->resize && this->move) {
          fan::vec2 new_size = (d.position - get_position());
          static constexpr fan::vec2 min_size(10, 10);
          new_size.clamp(min_size);
          this->set_size(new_size.x);
          return user_cb(d);
        }
        else if (this->move) {
          fan::vec3 p = get_position();
          this->set_position(fan::vec3(d.position + click_offset, p.z));
          return user_cb(d);
        }
      }
      else {
        return user_cb(d);
      }
      return 0;
    };
    vfi_root = in;
  }
  void push_child(const loco_t::shape_t& shape) {
    children.push_back({shape});
  }
  fan::vec3 get_position() {
    return vfi_root.get_position();
  }
  void set_position(const fan::vec3& position) {
    fan::vec2 root_pos = vfi_root.get_position();
    fan::vec2 offset = position - root_pos;
    vfi_root.set_position(fan::vec3(root_pos + offset, position.z));
    for (auto& child : children) {
      child.set_position(fan::vec3(fan::vec2(child.get_position()) + offset, position.z));
    }
  }
  fan::vec2 get_size() {
    return vfi_root.get_size();
  }
  void set_size(const fan::vec2& size) {
    fan::vec2 root_pos = vfi_root.get_size();
    fan::vec2 offset = size - root_pos;
    vfi_root.set_size(root_pos + offset);
    for (auto& child : children) {
      child.set_size(fan::vec2(child.get_size()) + offset);
    }
  }
  fan::vec2 click_offset = 0;
  bool move = false;
  bool resize = false;
  
  bool move_and_resize_auto = true;

  loco_t::shape_t vfi_root;
  struct child_data_t : loco_t::shape_t, T {
    
  };
  std::vector<child_data_t> children;
};

using vfi_root_t = vfi_root_custom_t<__empty_struct>;

#endif