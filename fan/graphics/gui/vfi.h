struct vfi_t {

  typedef uint16_t shape_type_t;

  inline static constexpr shape_type_t shape_type = loco_t::shape_type_t::vfi;

  struct shape_t { // using these on msvc gives intenal compiler error
    enum shape {
      always = 0,
      rectangle = 1
    };
  };
  struct shape_properties_always_t {
    f32_t z;
  };
  struct shape_data_always_t {

  };
  struct shape_properties_rectangle_t {
    fan::vec3 position;
    fan::vec2 size;
    fan::vec3 angle;
    fan::vec2 rotation_point;
    loco_t::viewport_t viewport;
    loco_t::camera_t camera;
  };
  struct shape_data_rectangle_t {
    fan::vec2 position;
    fan::vec2 size;
    fan::vec3 angle;
    fan::vec2 rotation_point;
    loco_t::viewport_t viewport;
    loco_t::camera_t camera;
    shape_data_rectangle_t& operator=(const shape_properties_rectangle_t& p) {
      position = *(fan::vec2*)&p.position;
      size = p.size;
      viewport = p.viewport;
      camera = p.camera;
      angle = p.angle;
      rotation_point = p.rotation_point;
      return *this;
    }
  };

  struct iflags_t {
    iflags_t() : ignore_button(0) {

    }
    uint32_t ignore_button : 1;
    //static constexpr shape_type_t always_check_top_focus = 1 << 0;
  };

  typedef uint16_t shape_id_integer_t;

  struct focus_method_mouse_flag {
    focus_method_mouse_flag() : ignore_move_focus_check(0) {}
    uint32_t ignore_move_focus_check : 1;
  };

  enum class mouse_stage_e {
    outside,
    viewport_inside
  };

  struct mouse_move_data_t {
    vfi_t* vfi;
    fan::vec2 position;
    mouse_stage_e mouse_stage;
    focus_method_mouse_flag* flag;
  };

  struct mouse_button_data_t {
    vfi_t* vfi;
    fan::vec2 position;
    uint16_t button;
    fan::mouse_state button_state;
    mouse_stage_e mouse_stage;
    focus_method_mouse_flag* flag;
  };

  struct keyboard_data_t {
    vfi_t* vfi;
    int key;
    fan::keyboard_state_t keyboard_state;
  };

  struct text_data_t {
    vfi_t* vfi;
    uint32_t key;
  };

  using mouse_move_cb_t = std::function<int(const mouse_move_data_t&)>;
  using mouse_button_cb_t = std::function<int(const mouse_button_data_t&)>;
  using keyboard_cb_t = std::function<int(const keyboard_data_t&)>;
  using text_cb_t = std::function<int(const text_data_t&)>;

  struct common_shape_data_t {

    mouse_button_cb_t mouse_button_cb = [](const mouse_button_data_t&) -> int { return 0; };
    mouse_move_cb_t mouse_move_cb = [](const mouse_move_data_t&) -> int { return 0; };
    keyboard_cb_t keyboard_cb = [](const keyboard_data_t&) -> int { return 0; };
    text_cb_t text_cb = [](const text_data_t&) -> int { return 0; };
    f32_t depth;
    union{
      struct always_t {
        always_t() = default;
        using base_t = shape_data_always_t;
        always_t(shape_data_always_t& b) {
          memcpy(data, &b, sizeof(base_t));
        }
        inline operator shape_data_always_t&() {
          return *(shape_data_always_t*)data;
        }
        inline operator shape_data_always_t&() const {
          return *(shape_data_always_t*)data;
        }
        inline shape_data_always_t* operator->() {
          return reinterpret_cast<shape_data_always_t*>(data);
        }
        inline const shape_data_always_t* operator->() const {
          return reinterpret_cast<const shape_data_always_t*>(data);
        }
        uint8_t data[sizeof(shape_data_always_t)];
      }always;
      struct rectangle_t {
        rectangle_t() = default;
        using base_t = shape_data_rectangle_t;
        rectangle_t(shape_data_rectangle_t& b) {
          memcpy(data, &b, sizeof(shape_data_rectangle_t));
        }
        operator shape_data_rectangle_t&() {
          return *(shape_data_rectangle_t*)data;
        }
        operator shape_data_rectangle_t&() const {
          return *(shape_data_rectangle_t*)data;
        }
        inline auto* operator->() {
          return reinterpret_cast<shape_data_rectangle_t*>(data);
        }
        inline const auto* operator->() const {
          return reinterpret_cast<const shape_data_rectangle_t*>(data);
        }
        uint8_t data[sizeof(shape_data_rectangle_t)];
      }rectangle;
    }shape;
  }; 

  struct common_shape_properties_t {
    using type_t = vfi_t;

    shape_type_t shape_type;
    mouse_button_cb_t mouse_button_cb = [](const mouse_button_data_t&) -> int { return 0; };
    mouse_move_cb_t mouse_move_cb = [](const mouse_move_data_t&) -> int { return 0; };
    keyboard_cb_t keyboard_cb = [](const keyboard_data_t&) -> int { return 0; };
    text_cb_t text_cb = [](const text_data_t&) -> int { return 0; };
    struct {
      struct always_t {
        always_t() = default;
        using base_t = shape_properties_always_t;
        always_t(shape_properties_always_t& b) {
          memcpy(data, &b, sizeof(base_t));
        }
        operator shape_properties_always_t&() {
          return *(shape_properties_always_t*)data;
        }
        operator shape_properties_always_t&() const {
          return *(shape_properties_always_t*)data;
        }
        shape_properties_always_t* operator->() {
          return reinterpret_cast<shape_properties_always_t*>(data);
        }
        const shape_properties_always_t* operator->() const {
          return reinterpret_cast<const shape_properties_always_t*>(data);
        }
        uint8_t data[sizeof(shape_properties_always_t)];
      }always;
      struct rectangle_t {
        rectangle_t() = default;
        using base_t = shape_properties_rectangle_t;
        rectangle_t(shape_properties_rectangle_t& b) {
          memcpy(data, &b, sizeof(shape_properties_rectangle_t));
        }
        operator shape_properties_rectangle_t& () {
          return *reinterpret_cast<shape_properties_rectangle_t*>(data);
        }
        operator const shape_properties_rectangle_t& () const {
          return *reinterpret_cast<const shape_properties_rectangle_t*>(data);
        }
        inline const shape_properties_rectangle_t* operator->() const {
          return reinterpret_cast<const shape_properties_rectangle_t*>(data);
        }
        inline shape_properties_rectangle_t* operator->() {
          return reinterpret_cast<shape_properties_rectangle_t*>(data);
        }
        uint8_t data[sizeof(shape_properties_rectangle_t)];
      }rectangle;
    }shape;
    iflags_t flags;
    bool ignore_init_move = false;
  }; 

  using set_shape_t = decltype(common_shape_data_t::shape);
  using set_rectangle_t = decltype(set_shape_t::rectangle)::base_t;

  //struct shape_id_wrap_t {
  //  shape_id_wrap_t() = default;
  //  shape_id_wrap_t(loco_t::shape_t* nt) : n(nt){

  //  }
  //  bool operator==(shape_id_wrap_t id) {
  //    return ((shape_list_nr_t)*this).NRI == (id.operator shape_list_nr_t&()).NRI;
  //  }
  //  operator shape_list_nr_t () const {
  //    return *(shape_list_nr_t*)n->gdp4();
  //  }
  //  operator shape_list_nr_t&() {
  //    return *(shape_list_nr_t*)n->gdp4();
  //  }
  //  operator loco_t::shape_t& () {
  //    return *(loco_t::shape_t*)n;
  //  }
  //  bool is_invalid() {
  //    return n == nullptr || shape_list_inric(operator shape_list_nr_t&());
  //  }
  //  shape_id_wrap_t& operator=(const shape_list_nr_t& nr) {
  //    operator shape_list_nr_t () = nr;
  //    return *this;
  //  }
  //  shape_id_wrap_t& operator=(const loco_t::shape_t& id) {
  //    operator shape_list_nr_t () = shape_id_wrap_t((loco_t::shape_t*)&id).operator shape_list_nr_t();
  //    return *this;
  //  }
  //  void invalidate() {
  //    if (is_invalid()) {
  //      return;
  //    }
  //    gloco->vfi.shape_list.unlrec(operator shape_list_nr_t&());
  //    operator shape_list_nr_t&() = gloco->vfi.shape_list.gnric();
  //  }
  //  loco_t::shape_t* n = nullptr;
  //};

  struct {
    loco_t::shape_t mouse;
    loco_t::shape_t keyboard;
    loco_t::shape_t text;

    struct {
      struct {
        fan::vec2 position;
        focus_method_mouse_flag flags;
      }mouse;
      struct {

      }keyboard;
      struct {

      }text;
    }method;
  }focus;

  using properties_t = common_shape_properties_t;

  loco_t::shape_t push_back(const properties_t& p) {

    loco_t::kps_t::vfi_t keypack;
    ri_t instance;
    instance.shape_data = new common_shape_data_t;
    instance.shape_type = p.shape_type;
    instance.flags = p.flags;
    instance.shape_data->mouse_move_cb = p.mouse_move_cb;
    instance.shape_data->mouse_button_cb = p.mouse_button_cb;
    instance.shape_data->keyboard_cb = p.keyboard_cb;
    instance.shape_data->text_cb = p.text_cb;
    switch(p.shape_type) {
      case shape_t::always:{
        instance.shape_data->depth = p.shape.always->z;
        break;
      }
      case shape_t::rectangle:{
        instance.shape_data->depth = p.shape.rectangle->position.z;
        if (p.shape.rectangle->camera.iic()) {
          instance.shape_data->shape.rectangle->camera = gloco->orthographic_render_view.camera;
          fan::print("warning using default camera");
        }
        else {
          instance.shape_data->shape.rectangle->camera = p.shape.rectangle->camera;
        } 
        if (p.shape.rectangle->viewport.iic()) {
          instance.shape_data->shape.rectangle->viewport = gloco->orthographic_render_view.viewport;
          fan::print("warning using default viewport");
        }
        else {
          instance.shape_data->shape.rectangle->viewport = p.shape.rectangle->viewport;
        }
        
        instance.shape_data->shape.rectangle->position = *(fan::vec2*)&p.shape.rectangle->position;
        instance.shape_data->shape.rectangle->angle = p.shape.rectangle->angle;
        instance.shape_data->shape.rectangle->rotation_point = p.shape.rectangle->rotation_point;
        instance.shape_data->shape.rectangle->size = p.shape.rectangle->size;
        break;
      }
    }

    auto ret = gloco->shape_add(loco_t::shape_type_t::vfi, 0, instance,
      loco_t::Key_e::depth, (uint16_t)p.shape.rectangle->position.z,
      loco_t::Key_e::viewport, p.shape.rectangle->viewport,
      loco_t::Key_e::camera, p.shape.rectangle->camera,
      loco_t::Key_e::ShapeType, (loco_t::loco_t::shaper_t::ShapeTypeIndex_t)loco_t::shape_type_t::vfi
    );

    fan::vec2 mouse_position = gloco->get_mouse_position();
    fan::vec2 tp = transform(mouse_position, p.shape_type, instance.shape_data);

    if (focus.mouse.iic()) {
      if (!p.ignore_init_move && viewport_inside(p.shape_type, instance.shape_data, tp) == mouse_stage_e::viewport_inside) {
        feed_mouse_move(mouse_position);
      }
    }

    return ret;
  }
  void erase(loco_t::shape_t& in) {
    bool fm = focus.mouse == in;
    if (fm) {
      focus.mouse.sic();
    }
    if (focus.keyboard == in) {
      focus.keyboard.sic();
    }
    if (focus.text == in) {
      focus.text.sic();
    }
    auto data = ((ri_t*)in.GetData(gloco->shaper))->shape_data;
    gloco->shaper.remove(in);
    delete data;
    if (fm) {
      feed_mouse_move(gloco->get_mouse_position());
    }
  }
  template <typename T>
  void set_always(loco_t::shape_t in, auto T::*member, auto value) {
    ((ri_t*)focus.mouse.GetData(gloco->shaper))->shape_data->shape.always->*member = value;
  }
  template <typename T>
  void set_rectangle(loco_t::shape_t in, auto T::*member, auto value) {
    ((T*)&((ri_t*)focus.mouse.GetData(gloco->shaper))->shape_data->shape.rectangle)->*member = value;
  }

  void set_common_data(loco_t::shape_t in, auto common_shape_data_t::*member, auto value) {
    ((ri_t*)focus.mouse.GetData(gloco->shaper))->shape_data->*member = value;
  }

  static fan::vec2 transform_position(const fan::vec2& p, loco_t::viewport_t viewport, loco_t::camera_t camera) {

#if fan_debug >= fan_debug_high
    if (viewport.iic()) {
      fan::throw_error("invalid viewport");
    }
    if (camera.iic()) {
      fan::throw_error("invalid camera");
    }
#endif

    auto v = gloco->viewport_get(viewport);
    auto c = gloco->camera_get(camera);

    fan::vec2 viewport_position = v.viewport_position;
    fan::vec2 viewport_size = v.viewport_size;

    f32_t l = c.coordinates.left;
    f32_t r = c.coordinates.right;
    f32_t t = c.coordinates.up;
    f32_t b = c.coordinates.down;

    fan::vec2 tp = p - viewport_position;
    fan::vec2 d = viewport_size;
    tp /= d;
    tp = fan::vec2(r * tp.x - l * tp.x + l, b * tp.y - t * tp.y + t);
    return tp;
  }

  vfi_t::mouse_stage_e viewport_inside(shape_type_t shape_type, common_shape_data_t* data, const fan::vec2& p) {
    switch (shape_type) {
    case shape_t::always: {
      return mouse_stage_e::viewport_inside;
    }
    case shape_t::rectangle: {
      fan::vec2 size = data->shape.rectangle->size;
      bool in = fan_2d::collision::rectangle::point_inside_no_rotation(
        p,
        data->shape.rectangle->position,
        size
      );
      /* if (!loco->get_context().viewport_list[data->shape.rectangle.viewport].viewport_id->inside_wir(p)) {
      return mouse_stage_e::outside;
      }*/
      return in ? mouse_stage_e::viewport_inside : mouse_stage_e::outside;
    }
    default: {
      fan::throw_error("invalid shape_type");
      return mouse_stage_e::outside;
    }
    }
  }

  fan::vec2 transform(const fan::vec2& v, shape_type_t shape_type, common_shape_data_t* shape_data) {
    auto& context = gloco->get_context();
    switch (shape_type) {
    case shape_t::always: {
      return v;
    }
    case shape_t::rectangle: {
      fan::vec2 camera_position = gloco->camera_get(shape_data->shape.rectangle->camera).position;
      auto p = transform_position(
        v,
        shape_data->shape.rectangle->viewport,
        shape_data->shape.rectangle->camera
      ) + camera_position;
      return p;
    }
    default: {
      fan::throw_error("invalid shape type");
      return {};
    }
    }
  }

  void init_focus_mouse_flag() {
    focus.method.mouse.flags.ignore_move_focus_check = false;
  }

  // might fail because shape_type isnt set

  loco_t::shape_t& get_focus_mouse() {
    return focus.mouse;
  }

  // otherwise copy with const&
  void set_focus_mouse(const loco_t::shaper_t::ShapeID_t& id) {
    focus.mouse.NRI = id.NRI;
    init_focus_mouse_flag();
  }

  loco_t::shape_t& get_focus_keyboard() {
    return focus.keyboard;
  }

  void set_focus_keyboard(const loco_t::shaper_t::ShapeID_t& id) {
    focus.keyboard.NRI = id.NRI;
  }

  loco_t::shape_t& get_focus_text() {
    return focus.text;
  }

  void set_focus_text(const loco_t::shaper_t::ShapeID_t& id) {
    focus.text.NRI = id.NRI;
  }

  void invalidate_focus_mouse() {
    focus.mouse.sic();
    focus.method.mouse.flags.ignore_move_focus_check = false;
  }

  void invalidate_focus_keyboard() {
    focus.keyboard.sic();
  }

  void invalidate_focus_text() {
    focus.text.sic();
  }

  void feed_mouse_move(const fan::vec2& position) {
    focus.method.mouse.position = position;
    mouse_move_data_t mouse_move_data;
    mouse_move_data.vfi = this;
    mouse_move_data.flag = &focus.method.mouse.flags;
    if (!focus.mouse.iic()) {
      auto& data = *(ri_t*)focus.mouse.GetData(gloco->shaper);
      fan::vec2 tp = transform(position, data.shape_type, data.shape_data);
      mouse_move_data.mouse_stage = viewport_inside(data.shape_type, data.shape_data, tp);
      mouse_move_data.position = tp;
      auto bcbfm = focus.mouse.NRI;
      data.shape_data->mouse_move_cb(mouse_move_data);
      if (bcbfm != focus.mouse.NRI) {
        data = *(ri_t*)focus.mouse.GetData(gloco->shaper);
        tp = transform(position, data.shape_type, data.shape_data);
        mouse_move_data.mouse_stage = viewport_inside(data.shape_type, data.shape_data, tp);
      }
      if (focus.method.mouse.flags.ignore_move_focus_check == true) {
        return;
      }
    }

    f32_t closest_z = -1;
    loco_t::shaper_t::ShapeID_t closest_z_nr;
    closest_z_nr.sic();

    {
      loco_t::shaper_t::KeyTraverse_t KeyTraverse;
      KeyTraverse.Init(gloco->shaper);

      while (KeyTraverse.Loop(gloco->shaper)) {

        loco_t::shaper_t::KeyTypeIndex_t kti = KeyTraverse.kti(gloco->shaper);
        if (kti == loco_t::Key_e::ShapeType) {
          auto sti = *(loco_t::shaper_t::ShapeTypeIndex_t*)KeyTraverse.kd();
          if (sti != loco_t::shape_type_t::vfi) {
            continue;
          }
        }
        switch (kti) {
            case loco_t::Key_e::draw_mode: 
            case loco_t::Key_e::vertex_count: 
            continue;
        }
        if (!KeyTraverse.isbm) {
          continue;
        }
        loco_t::shaper_t::BlockTraverse_t BlockTraverse;
        BlockTraverse.Init(gloco->shaper, KeyTraverse.bmid());

        do {
         for (int i = 0; i < BlockTraverse.GetAmount(gloco->shaper); ++i) {
           auto& data = ((ri_t*)BlockTraverse.GetData(gloco->shaper))[i];
           fan::vec2 tp = transform(position, data.shape_type, data.shape_data);
           mouse_move_data.mouse_stage = viewport_inside(data.shape_type, data.shape_data, tp);
           if (mouse_move_data.mouse_stage == mouse_stage_e::viewport_inside) {
             if (data.shape_data->depth > closest_z) {
               closest_z = data.shape_data->depth;
               closest_z_nr = *gloco->shaper.GetShapeID(loco_t::shape_type_t::vfi, BlockTraverse.GetBlockID(), i);
             }
           }
         }
        } while (BlockTraverse.Loop(gloco->shaper));

      }
    }
    if (closest_z != -1) {
      auto* data = (ri_t*)closest_z_nr.GetData(gloco->shaper);
      fan::vec2 tp = transform(position, data->shape_type, data->shape_data);
      mouse_move_data.position = tp;
      mouse_move_data.mouse_stage = viewport_inside(data->shape_type, data->shape_data, tp);
      set_focus_mouse(closest_z_nr); // can be wrong
      data->shape_data->mouse_move_cb(mouse_move_data);
      return;
    }
    focus.mouse.sic();
    return;
  }

  void feed_mouse_button(uint16_t button, fan::mouse_state state) {
    mouse_button_data_t mouse_button_data;
    mouse_button_data.vfi = this;
    if (focus.mouse.iic()) {
      return;
    }

    mouse_button_data.button = button;
    mouse_button_data.button_state = state;

    auto* data = (ri_t*)focus.mouse.GetData(gloco->shaper);

    mouse_button_data.position = transform(focus.method.mouse.position, data->shape_type, data->shape_data);
    mouse_button_data.mouse_stage = viewport_inside(data->shape_type, data->shape_data, mouse_button_data.position);
    mouse_button_data.flag = &focus.method.mouse.flags;
    auto bcbfm = focus.mouse.NRI;

    data->shape_data->mouse_button_cb(mouse_button_data);

    if (bcbfm != focus.mouse.NRI) {
      if (focus.mouse.iic()) {
        return;
      }
      data = (ri_t*)focus.mouse.GetData(gloco->shaper);
      mouse_button_data.position = transform(focus.method.mouse.position, data->shape_type, data->shape_data);
      mouse_button_data.mouse_stage = viewport_inside(data->shape_type, data->shape_data, mouse_button_data.position);
    }

    if (mouse_button_data.mouse_stage == mouse_stage_e::outside) {
      if (focus.method.mouse.flags.ignore_move_focus_check == false) {
        focus.mouse.sic();
        feed_mouse_move(focus.method.mouse.position);
      }
    }
  }

  void feed_keyboard(int key, fan::keyboard_state_t keyboard_state) {
    keyboard_data_t keyboard_data;
    keyboard_data.vfi = this;
    if (focus.keyboard.iic()) {
      return;
    }

    keyboard_data.key = key;
    keyboard_data.keyboard_state = keyboard_state;

    ((ri_t*)focus.keyboard.GetData(gloco->shaper))->shape_data->keyboard_cb(keyboard_data);
  }

  void feed_text(uint32_t key) {
    text_data_t text_data;
    text_data.vfi = this;
    if (focus.text.iic()) {
      return;
    }

    text_data.key = key;

    ((ri_t*)focus.text.GetData(gloco->shaper))->shape_data->text_cb(text_data);
  }

  fan::vec3 get_position(loco_t::shape_t& in) {
    auto& shape = *(ri_t*)in.GetData(gloco->shaper);
    switch (shape.shape_type) {
    case vfi_t::shape_t::rectangle: {
      return fan::vec3(
        shape.shape_data->shape.rectangle->position,
        shape.shape_data->depth
      );
    }
    }
    fan::throw_error("invalid get_position for id");
    return 0;
  }

  void set_position(loco_t::shape_t& in, const fan::vec3& position) {
    auto& shape = *(ri_t*)in.GetData(gloco->shaper);
    switch (shape.shape_type) {
    case vfi_t::shape_t::rectangle: {
      shape.shape_data->shape.rectangle->position = *(fan::vec2*)&position;
      shape.shape_data->depth = position.z;
      return;
    }
    }
    fan::throw_error("invalid set_position for id");
  }

  fan::vec2 get_size(loco_t::shape_t& in) {
    auto& shape = *(ri_t*)in.GetData(gloco->shaper);
    switch (shape.shape_type) {
    case vfi_t::shape_t::rectangle: {
      return shape.shape_data->shape.rectangle->size;
    }
    }
    fan::throw_error("invalid get_position for id");
    return 0;
  }

  void set_size(loco_t::shape_t& in, const fan::vec2& size) {
    auto& shape = *(ri_t*)in.GetData(gloco->shaper);
    switch (shape.shape_type) {
    case vfi_t::shape_t::rectangle: {
      shape.shape_data->shape.rectangle->size = size;
      return;
    }
    }
    fan::throw_error("invalid set_position for id");
  }

  void set_angle(loco_t::shape_t& in, const fan::vec3& angle) {
    auto& shape = *(ri_t*)in.GetData(gloco->shaper);
    switch (shape.shape_type) {
    case vfi_t::shape_t::rectangle: {
      shape.shape_data->shape.rectangle->angle = angle;
      return;
    }
    }
    fan::throw_error("invalid set_position for id");
  }

  void set_rotation_point(loco_t::shape_t& in, const fan::vec2& rotation_point) {
    auto& shape = *(ri_t*)in.GetData(gloco->shaper);
    switch (shape.shape_type) {
    case vfi_t::shape_t::rectangle: {
      shape.shape_data->shape.rectangle->rotation_point = rotation_point;
      return;
    }
    }
    fan::throw_error("invalid set_position for id");
  }

#pragma pack(push, 1)
  struct ri_t {
    shape_type_t shape_type;
    common_shape_data_t* shape_data = 0;
    iflags_t flags;
  };
#pragma pack(pop)


  void open() {
    focus.mouse.sic();
    focus.keyboard.sic();
    focus.text.sic();

    loco_t::shaper_t::BlockProperties_t bp;
    bp.MaxElementPerBlock = (loco_t::shaper_t::MaxElementPerBlock_t)MaxElementPerBlock;
    bp.RenderDataSize = 0;
    bp.DataSize = sizeof(ri_t);

    gloco->shaper.SetShapeType(loco_t::shape_type_t::vfi, bp);

    loco_t::get_position_functions[loco_t::shape_type_t::vfi] = [](loco_t::shape_t* shape) {
      return gloco->vfi.get_position(*shape);
    };
    loco_t::set_position2_functions[loco_t::shape_type_t::vfi] = [](loco_t::shape_t* shape, const fan::vec2& position) {
      gloco->vfi.set_position(*shape, fan::vec3(position, gloco->vfi.get_position(*shape).z));
    };
    loco_t::set_position3_functions[loco_t::shape_type_t::vfi] = [](loco_t::shape_t* shape, const fan::vec3& position) {
      gloco->vfi.set_position(*shape, position);
    };
    loco_t::get_size_functions[loco_t::shape_type_t::vfi] = [](loco_t::shape_t* shape) {
      return gloco->vfi.get_size(*shape);
    };
    loco_t::set_size_functions[loco_t::shape_type_t::vfi] = [](loco_t::shape_t* shape, const fan::vec2& size) {
      gloco->vfi.set_size(*shape, size);
    };
  }
};