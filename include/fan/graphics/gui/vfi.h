// very flexible input

struct vfi_t {

  void open() {
    focus.mouse.invalidate();
    depth_list.open();
  }
  void close() {
    assert(0); // close shape list
    depth_list.close();
  }

  typedef uint16_t shape_type_t;

  struct shape_t {
    static constexpr shape_type_t always = 0;
    static constexpr shape_type_t rectangle = 1;
  };
  struct shape_properties_always_t {

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
    fan::vec3 position;
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
    //static constexpr shape_type_t always_check_top_focus = 1 << 0;
  };

  typedef uint16_t shape_id_integer_t;

  enum class mouse_stage_e {
    outside,
    inside
  };

  struct mouse_move_data_t {
    vfi_t* vfi;
    fan::vec2 position;
    mouse_stage_e mouse_stage;
    uint64_t udata;
  };

  struct mouse_button_data_t {
    vfi_t* vfi;
    uint16_t key;
    fan::key_state key_state;
    mouse_stage_e mouse_stage;
    uint64_t udata;
  };

  typedef void(*mouse_move_cb_t)(const mouse_move_data_t&);
  typedef void(*mouse_button_cb_t)(const mouse_button_data_t&);

  struct common_shape_data_t { 
    mouse_move_cb_t mouse_move_cb;
    mouse_button_cb_t mouse_button_cb;
    union{
      shape_data_always_t always; 
      shape_data_rectangle_t rectangle; 
    }shape;
  }; 

  struct common_shape_properties_t { 
    uint32_t depth;
    shape_type_t shape_type;
    mouse_move_cb_t mouse_move_cb = [](const mouse_move_data_t&) -> void {};
    mouse_button_cb_t mouse_button_cb = [](const mouse_button_data_t&) -> void {};
    union {
      shape_properties_always_t always; 
      shape_properties_rectangle_t rectangle; 
    }shape;
    iflags_t flags;
    uint64_t udata;
  }; 

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

    union {
      struct {
        bool ignore_move_focus_check;
      }mouse;
      struct {

      }keyboard;
    }method;
  }focus;

  fan::hector_t<shape_list_t> depth_list;

  uint32_t push_depth() {
    auto l_shape_list = &depth_list[depth_list.resize(depth_list.size() + 1)];
    shape_list_open(l_shape_list);
    return depth_list.size() - 1;
  }

  using properties_t = common_shape_properties_t;

  void push_shape(const properties_t& p) {
    auto nr = shape_list_NewNodeLast(&depth_list[p.depth]);
    auto n = shape_list_GetNodeByReference(&depth_list[p.depth], nr);
    n->data.shape_type = p.shape_type;
    n->data.flags = p.flags;
    n->data.udata = p.udata;
    n->data.shape_data.mouse_move_cb = p.mouse_move_cb;
    n->data.shape_data.mouse_button_cb = p.mouse_button_cb;
    switch(p.shape_type) {
      case shape_t::always:{
        break;
      }
      case shape_t::rectangle:{
        n->data.shape_data.shape.rectangle = p.shape.rectangle;
        break;
      }
    }
  }

  static fan::vec2 transform_position(const fan::vec2& p, fan::opengl::viewport_t* viewport, fan::opengl::matrices_t* matrices) {
    fan::vec2 viewport_position = viewport->get_viewport_position(); 
    fan::vec2 viewport_size = viewport->get_viewport_size();
    fan::vec2 x;
    x.x = (p.x - viewport_position.x - viewport_size.x / 2) / (viewport_size.x / 2);
    x.y = (p.y - viewport_position.y - viewport_size.y / 2) / (viewport_size.y / 2) + (viewport_position.y / viewport_size.y) * 2;
    return x;
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

  struct magic_vec2 : fan::vec2 {
    using fan::vec2::_vec2;

    fan::vec2 transform(loco_t* loco, shape_type_t shape_type, common_shape_data_t* shape_data) const {
      switch (shape_type) {
        case shape_t::always: {
          break;
        }
        case shape_t::rectangle: {
          return transform_position(
            *this,
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
  };

  void feed_mouse_move(loco_t* loco, magic_vec2 position) {
    uint32_t depth = depth_list.size() - 1;
    if (depth == -1) {
      return;
    }

    mouse_move_data_t mouse_move_data;
    mouse_move_data.vfi = this;
    if (!focus.mouse.is_invalid()) {
      shape_list_t* shape_list = &depth_list[depth];
      auto* data = &shape_list_GetNodeByReference(shape_list, focus.mouse)->data;
      fan::vec2 tp = position.transform(loco, data->shape_type, &data->shape_data);
      mouse_move_data.mouse_stage = inside(loco, data->shape_type, &data->shape_data, tp);
      mouse_move_data.udata = data->udata;
      mouse_move_data.position = tp;
      if (focus.method.mouse.ignore_move_focus_check == true) {
        data->shape_data.mouse_move_cb(mouse_move_data);
        return;
      }
      if (mouse_move_data.mouse_stage == mouse_stage_e::inside) {
        data->shape_data.mouse_move_cb(mouse_move_data);
        return;
      }
      data->shape_data.mouse_move_cb(mouse_move_data);
    }

    for (uint32_t i = depth + 1; i--; ) {
      shape_list_t* shape_list = &depth_list[i];
      auto it = shape_list_GetNodeFirst(shape_list);
      while(it != shape_list->dst) {
        auto* n = shape_list_GetNodeByReference(shape_list, it);
        fan::vec2 tp = position.transform(loco, n->data.shape_type, &n->data.shape_data);
        auto stage = inside(loco, n->data.shape_type, &n->data.shape_data, tp);
        if (stage == mouse_stage_e::inside) {
          focus.mouse = it;
          mouse_move_data.mouse_stage = stage;
          mouse_move_data.udata = n->data.udata;
          n->data.shape_data.mouse_move_cb(mouse_move_data);
          return;
        }
        it = n->NextNodeReference;
      }
    }
    focus.mouse.invalidate();
    return;
  }

  void feed_mouse_button(loco_t* loco, uint16_t button, fan::key_state state, uint32_t depth) {
    uint32_t depth = depth_list.size() - 1;
    if (depth == -1) {
      return;
    }

    mouse_button_data_t mouse_button_data;
    mouse_button_data.vfi = this;
    if (focus.mouse.is_invalid()) {
      return;
    }
    if (!focus.mouse.is_invalid()) {
      shape_list_t* shape_list = &depth_list[depth];
      auto* data = &shape_list_GetNodeByReference(shape_list, focus.mouse)->data;
      fan::vec2 tp = position.transform(loco, data->shape_type, &data->shape_data);
      mouse_button_data.mouse_stage = inside(loco, data->shape_type, &data->shape_data, tp);
      mouse_button_data.udata = data->udata;
      mouse_button_data.position = tp;
      if (focus.method.mouse.ignore_move_focus_check == true) {
        data->shape_data.mouse_move_cb(mouse_button_data);
        return;
      }
      if (mouse_button_data.mouse_stage == mouse_stage_e::inside) {
        data->shape_data.mouse_move_cb(mouse_button_data);
        return;
      }
      data->shape_data.mouse_move_cb(mouse_button_data);
    }

    for (uint32_t i = depth + 1; i--; ) {
      shape_list_t* shape_list = &depth_list[i];
      auto it = shape_list_GetNodeFirst(shape_list);
      while(it != shape_list->dst) {
        auto* n = shape_list_GetNodeByReference(shape_list, it);
        fan::vec2 tp = position.transform(loco, n->data.shape_type, &n->data.shape_data);
        auto stage = inside(loco, n->data.shape_type, &n->data.shape_data, tp);
        if (stage == mouse_stage_e::inside) {
          focus.mouse = it;
          mouse_button_data.mouse_stage = stage;
          mouse_button_data.udata = n->data.udata;
          n->data.shape_data.mouse_move_cb(mouse_button_data);
          return;
        }
        it = n->NextNodeReference;
      }
    }
    focus.mouse.invalidate();
    return;
  }
};