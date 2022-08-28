// very flexible input

struct vfi_t {

  loco_t* get_loco() {
    return OFFSETLESS(this, loco_t, vfi_var_name);
  }

  void open() {
    focus.mouse.invalidate();
    focus.keyboard.invalidate();
    shape_list_open(&shape_list);
  }
  void close() {
    shape_list_close(&shape_list);
  }

  typedef uint16_t shape_type_t;

  struct shape_t {
    static constexpr shape_type_t always = 0;
    static constexpr shape_type_t rectangle = 1;
  };
  struct shape_properties_always_t {
    f32_t z;
  };
  struct shape_data_always_t {

  };
  struct shape_properties_rectangle_t {
    fan::vec3 position;
    fan::vec2 size;
    fan::opengl::viewport_list_NodeReference_t viewport;
    fan::opengl::matrices_list_NodeReference_t matrices;
  };
  struct shape_data_rectangle_t {
    fan::vec2 position;
    fan::vec2 size;
    fan::opengl::viewport_list_NodeReference_t viewport;
    fan::opengl::matrices_list_NodeReference_t matrices;
    shape_data_rectangle_t& operator=(const shape_properties_rectangle_t& p) {
      position = p.position;
      size = p.size;
      viewport = p.viewport;
      matrices = p.matrices;
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
    inside
  };

  struct mouse_move_data_t {
    vfi_t* vfi;
    fan::vec2 position;
    mouse_stage_e mouse_stage;
    focus_method_mouse_flag* flag;
    uint64_t udata;
  };

  struct mouse_button_data_t {
    vfi_t* vfi;
    fan::vec2 position;
    uint16_t button;
    fan::key_state button_state;
    mouse_stage_e mouse_stage;
    focus_method_mouse_flag* flag;
    uint64_t udata;
  };

  struct keyboard_data_t {
    vfi_t* vfi;
    uint16_t key;
    fan::key_state key_state;
    uint64_t udata;
  };

  typedef void(*mouse_move_cb_t)(const mouse_move_data_t&);
  typedef void(*mouse_button_cb_t)(const mouse_button_data_t&);

  typedef void(*keyboard_cb_t)(const keyboard_data_t&);

  struct common_shape_data_t { 
    mouse_move_cb_t mouse_move_cb;
    mouse_button_cb_t mouse_button_cb;
    keyboard_cb_t keyboard_cb;
    f32_t depth;
    union{
      shape_data_always_t always; 
      shape_data_rectangle_t rectangle; 
    }shape;
  }; 

  struct common_shape_properties_t {
    shape_type_t shape_type;
    mouse_move_cb_t mouse_move_cb = [](const mouse_move_data_t&) -> void {};
    mouse_button_cb_t mouse_button_cb = [](const mouse_button_data_t&) -> void {};
    keyboard_cb_t keyboard_cb = [](const keyboard_data_t&) -> void {};
    union {
      shape_properties_always_t always; 
      shape_properties_rectangle_t rectangle; 
    }shape;
    iflags_t flags;
    uint64_t udata;
  }; 

  using set_shape_t = decltype(common_shape_data_t::shape);
  using set_rectangle_t = decltype(set_shape_t::rectangle);

  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix shape_list
  #define BLL_set_type_node shape_id_integer_t
  #define BLL_set_node_data \
    shape_type_t shape_type; \
    common_shape_data_t shape_data; \
    iflags_t flags; \
    uint64_t udata;
  #define BLL_set_Link 1
  #define BLL_set_StructFormat 1
  #define BLL_set_NodeReference_Overload_Declare \
    bool is_invalid() const { \
      return NRI == (shape_id_integer_t)fan::uninitialized; \
    } \
    void invalidate() { \
      NRI = fan::uninitialized; \
    }
  #include _FAN_PATH(BLL/BLL.h)

  typedef shape_list_NodeReference_t shape_id_t;

  struct {
    shape_id_t mouse;
    shape_id_t keyboard;

    struct {
      struct {
        fan::vec2 position;
        focus_method_mouse_flag flags;
      }mouse;
      struct {

      }keyboard;
    }method;
  }focus;

  shape_list_t shape_list;

  using properties_t = common_shape_properties_t;

  shape_id_t push_shape(const properties_t& p) {
    auto nr = shape_list_NewNodeLast(&shape_list);
    auto n = shape_list_GetNodeByReference(&shape_list, nr);
    n->data.shape_type = p.shape_type;
    n->data.flags = p.flags;
    n->data.udata = p.udata;
    n->data.shape_data.mouse_move_cb = p.mouse_move_cb;
    n->data.shape_data.mouse_button_cb = p.mouse_button_cb;
    n->data.shape_data.keyboard_cb = p.keyboard_cb;
    switch(p.shape_type) {
      case shape_t::always:{
        n->data.shape_data.depth = p.shape.always.z;
        break;
      }
      case shape_t::rectangle:{
        n->data.shape_data.depth = p.shape.rectangle.position.z;
        n->data.shape_data.shape.rectangle = p.shape.rectangle;
        break;
      }
    }
    return nr;
  }
  void erase(shape_id_t id) {
    if (focus.mouse == id) {
      focus.mouse.invalidate();
    }
    if (focus.keyboard == id) {
      focus.keyboard.invalidate();
    }
    shape_list_Unlink(&shape_list, id);
    shape_list_Recycle(&shape_list, id);
  }
    template <typename T>
  void set_always(shape_id_t id, auto T::*member, auto value) {
    auto n = shape_list_GetNodeByReference(&shape_list, id);
    n->data.shape_data.shape.always.*member = value;
  }
  template <typename T>
  void set_rectangle(shape_id_t id, auto T::*member, auto value) {
    auto n = shape_list_GetNodeByReference(&shape_list, id);
    n->data.shape_data.shape.rectangle.*member = value;
  }

  static fan::vec2 transform_position(const fan::vec2& p, fan::opengl::viewport_t* viewport, fan::opengl::matrices_t* matrices) {
      
    fan::vec2 viewport_position = viewport->get_position(); 
    fan::vec2 viewport_size = viewport->get_size();

    f32_t l = matrices->coordinates.left;
    f32_t r = matrices->coordinates.right;
    f32_t t = matrices->coordinates.top;
    f32_t b = matrices->coordinates.bottom;

    fan::vec2 tp = p - viewport_position;
    fan::vec2 d = viewport_size;
    tp /= d;
    tp = fan::vec2(r * tp.x - l * tp.x + l, b * tp.y - t * tp.y + t);
    return tp;
  }

  mouse_stage_e inside(loco_t* loco, shape_type_t shape_type, common_shape_data_t* data, const fan::vec2& p) {
    switch(shape_type) {
      case shape_t::always: {
        return mouse_stage_e::inside;
      }
      case shape_t::rectangle: {
        fan::vec2 size = data->shape.rectangle.size;
        bool in = fan_2d::collision::rectangle::point_inside_no_rotation(
          p,
          data->shape.rectangle.position - size,
          data->shape.rectangle.position + size
        );
        return in ? mouse_stage_e::inside : mouse_stage_e::outside;
      }
    }
  };

  fan::vec2 transform(const fan::vec2& v, shape_type_t shape_type, common_shape_data_t* shape_data) {
    loco_t* loco = get_loco();
    switch (shape_type) {
      case shape_t::always: {
        return v;
      }
      case shape_t::rectangle: {
        return transform_position(
          v,
          fan::opengl::viewport_list_GetNodeByReference(
            &loco->get_context()->viewport_list,
            shape_data->shape.rectangle.viewport
          )->data.viewport_id,
          fan::opengl::matrices_list_GetNodeByReference(
            &loco->get_context()->matrices_list,
            shape_data->shape.rectangle.matrices
          )->data.matrices_id
        );
        break;
      }
    }
  }

  void init_focus_mouse_flag() {
    focus.method.mouse.flags.ignore_move_focus_check = false;
  }

  shape_id_t get_focus_mouse() {
    return focus.mouse;
  }
  void set_focus_mouse(shape_id_t id) {
    focus.mouse = id;
    init_focus_mouse_flag();
  }
  shape_id_t get_focus_keyboard() {
    return focus.mouse;
  }
  void set_focus_keyboard(shape_id_t id) {
    focus.keyboard = id;
  }

  void invalidate_focus_mouse() {
    focus.mouse.invalidate();
    focus.method.mouse.flags.ignore_move_focus_check = false;
  }
  void invalidate_focus_keyboard() {
    focus.keyboard.invalidate();
  }

  uint64_t get_mouse_udata() {
    #if fan_debug >= fan_debug_low
      if (focus.mouse.is_invalid()) {
          fan::throw_error("trying to get id even though none is selected");
      }
    #endif
    return shape_list_GetNodeByReference(&shape_list, focus.mouse)->data.udata;
  }

  uint64_t get_keyboard_udata() {
    #if fan_debug >= fan_debug_low
      if (focus.keyboard.is_invalid()) {
          fan::throw_error("trying to get id even though none is selected");
      }
    #endif
    return shape_list_GetNodeByReference(&shape_list, focus.keyboard)->data.udata;
  }

  void feed_mouse_move(const fan::vec2& position) {
    loco_t* loco = get_loco();
    focus.method.mouse.position = position;
    mouse_move_data_t mouse_move_data;
    mouse_move_data.vfi = this;
    if (!focus.mouse.is_invalid()) {
      mouse_move_data.flag = &focus.method.mouse.flags;
      auto* data = &shape_list_GetNodeByReference(&shape_list, focus.mouse)->data;
      fan::vec2 tp = transform(position, data->shape_type, &data->shape_data);
      mouse_move_data.mouse_stage = inside(loco, data->shape_type, &data->shape_data, tp);
      mouse_move_data.udata = data->udata;
      mouse_move_data.position = tp;
      shape_id_t bcbfm = focus.mouse;
      data->shape_data.mouse_move_cb(mouse_move_data);
      if (bcbfm != focus.mouse) {
        data = &shape_list_GetNodeByReference(&shape_list, focus.mouse)->data;
        tp = transform(position, data->shape_type, &data->shape_data);
        mouse_move_data.mouse_stage = inside(loco, data->shape_type, &data->shape_data, tp);
      }
      if (focus.method.mouse.flags.ignore_move_focus_check == true) {
        return;
      }
      if (data->shape_type != shape_t::always && mouse_move_data.mouse_stage == mouse_stage_e::inside) {
        return;
      }
    }

    f32_t closest_z = -1;
    shape_list_NodeReference_t closest_z_nr;

    auto it = shape_list_GetNodeFirst(&shape_list);
    while(it != shape_list.dst) {
      auto* n = shape_list_GetNodeByReference(&shape_list, it);
      fan::vec2 tp = transform(position, n->data.shape_type, &n->data.shape_data);
      mouse_move_data.mouse_stage = inside(loco, n->data.shape_type, &n->data.shape_data, tp);
      if (mouse_move_data.mouse_stage == mouse_stage_e::inside) {
        if (n->data.shape_data.depth > closest_z) {
          closest_z = n->data.shape_data.depth;
          closest_z_nr = it;
        }
      }
      it = n->NextNodeReference;
    }
    if (closest_z != -1) {
      auto* n = shape_list_GetNodeByReference(&shape_list, closest_z_nr);
      fan::vec2 tp = transform(position, n->data.shape_type, &n->data.shape_data);
      mouse_move_data.position = tp;
      mouse_move_data.mouse_stage = inside(loco, n->data.shape_type, &n->data.shape_data, tp);
      set_focus_mouse(closest_z_nr);
      mouse_move_data.udata = n->data.udata;
      n->data.shape_data.mouse_move_cb(mouse_move_data);
      return;
    }
    focus.mouse.invalidate();
    return;
  }

  void feed_mouse_button(uint16_t button, fan::key_state state) {
    loco_t* loco = get_loco();
    mouse_button_data_t mouse_button_data;
    mouse_button_data.vfi = this;
    if (focus.mouse.is_invalid()) {
      return;
    }

    mouse_button_data.button = button;
    mouse_button_data.button_state = state;

    auto* data = &shape_list_GetNodeByReference(&shape_list, focus.mouse)->data;

    mouse_button_data.position = transform(focus.method.mouse.position, data->shape_type, &data->shape_data);
    mouse_button_data.mouse_stage = inside(loco, data->shape_type, &data->shape_data, mouse_button_data.position);
    mouse_button_data.flag = &focus.method.mouse.flags;
    mouse_button_data.udata = data->udata;
    shape_id_t bcbfm = focus.mouse;

    data->shape_data.mouse_button_cb(mouse_button_data);

    if (bcbfm != focus.mouse) {
      if (focus.mouse.is_invalid()) {
        return;
      }
      data = &shape_list_GetNodeByReference(&shape_list, focus.mouse)->data;
      mouse_button_data.position = transform(focus.method.mouse.position, data->shape_type, &data->shape_data);
      mouse_button_data.mouse_stage = inside(loco, data->shape_type, &data->shape_data, mouse_button_data.position);
    }

    if (mouse_button_data.mouse_stage == mouse_stage_e::outside) {
      if (focus.method.mouse.flags.ignore_move_focus_check == false) {
        focus.mouse.invalidate();
        feed_mouse_move(focus.method.mouse.position);
      }
    }
  }
   void feed_keyboard(uint16_t key, fan::key_state key_state) {
    loco_t* loco = get_loco();
    keyboard_data_t keyboard_data;
    keyboard_data.vfi = this;
    if (focus.keyboard.is_invalid()) {
      return;
    }

    keyboard_data.key = key;
    keyboard_data.key_state = key_state;

    auto* data = &shape_list_GetNodeByReference(&shape_list, focus.keyboard)->data;

    keyboard_data.udata = data->udata;
    shape_id_t bcbfk = focus.keyboard;

    data->shape_data.keyboard_cb(keyboard_data);
  }
};