struct sb_sprite_name {

  struct instance_t {
    fan::vec3 position = 0;
  private:
    f32_t pad;
  public:
    fan::vec2 size = 0;
    fan::vec2 rotation_point = 0;
    fan::color color = fan::colors::white;
    fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
    f32_t angle = 0;
    fan::vec2 tc_position = 0;
    fan::vec2 tc_size = 1;
  };

  struct instance_properties_t {
    struct key_t : fan::masterpiece_t<
      fan::opengl::textureid_t<0>,
      fan::opengl::matrices_list_NodeReference_t,
      fan::opengl::viewport_list_NodeReference_t
    > {}key;
  };

  struct properties_t : instance_t {
    union {
      struct {
        fan::opengl::textureid_t<0> image;
        fan::opengl::matrices_list_NodeReference_t matrices;
        fan::opengl::viewport_list_NodeReference_t viewport;
      };
      instance_properties_t instance_properties;
    };
  };

  void push_back(fan::opengl::cid_t* cid, properties_t& p) {
    sb_push_back(cid, p);
  }
  void erase(fan::opengl::cid_t* cid) {
    sb_erase(cid);
  }

  void draw() {
    sb_draw();
  }

  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(instance_t) / 4));
  #ifndef sb_shader_vertex_path
    #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/sprite.vs)
  #endif
  #ifndef sb_shader_fragment_path
    #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/sprite.fs)
  #endif
  #include _FAN_PATH(graphics/opengl/2D/objects/shape_builder.h)

  void open() {
    sb_open();
  }
  void close() {
    sb_close();
  }

  /*void set_matrices(loco_t* loco, fan::opengl::cid_t* cid, fan::opengl::matrices_list_NodeReference_t n) {
    auto block = sb_get_block(loco, cid);
    *block->p[cid->instance_id].key.get_value<0>() = n;
  }

  void set_viewport(loco_t* loco, fan::opengl::cid_t* cid, fan::opengl::viewport_list_NodeReference_t n) {
    auto block = sb_get_block(loco, cid);
    *block->p[cid->instance_id].key.get_value<1>() = n;
  }*/

  void set_image(fan::opengl::cid_t* cid, uint32_t id) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    auto node = fan::opengl::_image_list_GetNodeByReference(&loco->get_context()->image_list, *block->p[cid->instance_id].key.get_value<0>());
    node->data.texture_id = id;
    //sb_set_key(cid, &properties_t::image, n);
  }
  void set_viewport_value(fan::opengl::cid_t* cid, fan::vec2 p, fan::vec2 s) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    auto node = fan::opengl::viewport_list_GetNodeByReference(&loco->get_context()->viewport_list, *block->p[cid->instance_id].key.get_value<2>());
    node->data.viewport_id->set_viewport(loco->get_context(), p, s, loco->get_window()->get_size());
    //sb_set_key(cid, &properties_t::image, n);
  }
};

#undef sb_shader_vertex_path
#undef sb_shader_fragment_path
#undef sb_sprite_name