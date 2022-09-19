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

  #define hardcode0_t fan::opengl::textureid_t<0>
  #define hardcode0_n image
  #define hardcode1_t fan::opengl::matrices_list_NodeReference_t
  #define hardcode1_n matrices
  #define hardcode2_t fan::opengl::viewport_list_NodeReference_t
  #define hardcode2_n viewport
  #include _FAN_PATH(graphics/opengl/2D/objects/hardcode_open.h)

  struct instance_properties_t {
    struct key_t : parsed_masterpiece_t {}key;
    expand_get_functions
  };

  struct properties_t : instance_t, instance_properties_t {
    properties_t() = default;
    properties_t(const instance_t& i) : instance_t(i) {}
    properties_t(const instance_properties_t& p) : instance_properties_t(p) {}
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

  fan::opengl::matrices_t* get_matrices(fan::opengl::cid_t* cid) {
    auto block = sb_get_block(cid);
    loco_t* loco = get_loco();
    return loco->get_context()->matrices_list[*block->p[cid->instance_id].key.get_value<
      instance_properties_t::key_t::get_index_with_type<fan::opengl::matrices_list_NodeReference_t>()
    >()].matrices_id;
  }
  void set_matrices(fan::opengl::cid_t* cid, fan::opengl::matrices_list_NodeReference_t n) {
    sb_set_key<instance_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
  }

  fan::opengl::viewport_t* get_viewport(fan::opengl::cid_t* cid) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    return loco->get_context()->viewport_list[*block->p[cid->instance_id].key.get_value<
      instance_properties_t::key_t::get_index_with_type<fan::opengl::viewport_list_NodeReference_t>()
    >()].viewport_id;
  }
  void set_viewport(fan::opengl::cid_t* cid, fan::opengl::viewport_list_NodeReference_t n) {
    sb_set_key<instance_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
  }

  /*

  void set(loco_t* loco, fan::opengl::cid_t* cid, fan::opengl::viewport_list_NodeReference_t n) {
    auto block = sb_get_block(loco, cid);
    *block->p[cid->instance_id].key.get_value<1>() = n;
  }*/

  void set_image(fan::opengl::cid_t* cid, fan::opengl::image_t* n) {
    sb_set_key<instance_properties_t::key_t::get_index_with_type<fan::opengl::textureid_t<0>>()>(cid, n);
  }
  void set_viewport_value(fan::opengl::cid_t* cid, fan::vec2 p, fan::vec2 s) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    loco->get_context()->viewport_list[*block->p[cid->instance_id].key.get_value<2>()].viewport_id->set(loco->get_context(), p, s, loco->get_window()->get_size());
    //sb_set_key(cid, &properties_t::image, n);
  }
};

#undef sb_shader_vertex_path
#undef sb_shader_fragment_path
#undef sb_sprite_name

#include _FAN_PATH(graphics/opengl/2D/objects/hardcode_close.h)