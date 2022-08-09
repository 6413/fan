struct rectangle_box_t {

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

  static constexpr uint32_t max_instance_size = std::min(256ull, 4096 / (sizeof(instance_t) / 4));

  struct instance_properties_t {
    struct key_t : fan::masterpiece_t<
      fan::opengl::matrices_list_NodeReference_t,
      fan::opengl::viewport_list_NodeReference_t
    > {}key;

    fan::opengl::theme_list_NodeReference_t theme;
  };

  struct properties_t : instance_t {
   // properties_t() : theme(fan_2d::graphics::gui::themes::deep_red()) {}
    union {
      struct {
        fan::opengl::matrices_list_NodeReference_t matrices;
        fan::opengl::viewport_list_NodeReference_t viewport;
        fan::opengl::theme_list_NodeReference_t theme;
      };
      instance_properties_t instance_properties;
    };
  };

  void push_back(loco_t* loco, fan::opengl::cid_t* cid, properties_t& p) {
    auto theme = get_theme(loco, p.theme);
    p.color = theme->button.color;
    p.outline_color = theme->button.outline_color;
    p.outline_size = theme->button.outline_size;

    sb_push_back(loco, cid, p);
  }
  void erase(loco_t* loco, fan::opengl::cid_t* cid) {
    sb_erase(loco, cid);
  }

  void draw(loco_t* loco) {
    sb_draw(loco);
  }

  #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle_box.vs)
  #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle_box.fs)
  #include _FAN_PATH(graphics/opengl/2D/objects/shape_builder.h)

  void open(loco_t* loco) {
    sb_open(loco);
  }
  void close(loco_t* loco) {
    sb_close(loco);
  }

  fan_2d::graphics::gui::theme_t* get_theme(loco_t* loco, fan::opengl::theme_list_NodeReference_t nr) {
    return fan::opengl::theme_list_GetNodeByReference(&loco->get_context()->theme_list, nr)->data.theme_id;
  }
  fan_2d::graphics::gui::theme_t* get_theme(loco_t* loco, fan::opengl::cid_t* cid) {
    auto block_node = bll_block_GetNodeByReference(&blocks, *(bll_block_NodeReference_t*)&cid->block_id);
    return get_theme(loco, block_node->data.block.p[cid->instance_id].theme);
  }
  void set_theme(loco_t* loco, fan::opengl::cid_t* cid, fan_2d::graphics::gui::theme_t* theme) {
    set(loco, cid, &instance_t::color, theme->button.color);
    set(loco, cid, &instance_t::outline_color, theme->button.outline_color);
    set(loco, cid, &instance_t::outline_size, theme->button.outline_size);
    auto block_node = bll_block_GetNodeByReference(&blocks, *(bll_block_NodeReference_t*)&cid->block_id);
    block_node->data.block.p[cid->instance_id].theme = theme;
  }
};