struct letter_t {

  struct instance_t {
    fan::vec3 position;
    f32_t outline_size;
    fan::vec2 size;
    fan::vec2 tc_position;
    fan::color color = fan::colors::white;
    fan::color outline_color;
    fan::vec2 tc_size;
  private:
    f32_t pad[2];
  };

  static constexpr uint32_t max_instance_size = fan::min(256ull, 4096 / (sizeof(instance_t) / 4));

  #define hardcode0_t fan::opengl::matrices_list_NodeReference_t
  #define hardcode0_n matrices
  #define hardcode1_t fan::opengl::viewport_list_NodeReference_t
  #define hardcode1_n viewport
  #include _FAN_PATH(graphics/opengl/2D/objects/hardcode_open.h)

  struct instance_properties_t {
    struct key_t : parsed_masterpiece_t {}key;
    expand_get_functions

    f32_t font_size;
    uint16_t letter_id;
  };

  struct properties_t : instance_t, instance_properties_t {
    properties_t() = default;
    properties_t(const instance_t& i) : instance_t(i) {}
    properties_t(const instance_properties_t& p) : instance_properties_t(p) {}
  };

  void push_back(fan::opengl::cid_t* cid, properties_t& p) {
    loco_t* loco = get_loco();
    fan::font::character_info_t si = loco->font.info.get_letter_info(p.letter_id, p.font_size);

    p.tc_position = si.glyph.position / loco->font.image.size;
    p.tc_size.x = si.glyph.size.x / loco->font.image.size.x;
    p.tc_size.y = si.glyph.size.y / loco->font.image.size.y;

    p.size = si.metrics.size / 2;

    sb_push_back(cid, p);
  }
  void erase(fan::opengl::cid_t* cid) {
    sb_erase(cid);
  }

  void draw() {
    loco_t* loco = get_loco();
    m_shader.use(loco->get_context());
    m_shader.set_int(loco->get_context(), "_t00", 0);
    loco->get_context()->opengl.call(loco->get_context()->opengl.glActiveTexture, fan::opengl::GL_TEXTURE0);
    loco->font.image.bind_texture(loco->get_context());
    
    sb_draw();
  }

  #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/letter.vs)
  #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/letter.fs)
  #include _FAN_PATH(graphics/opengl/2D/objects/shape_builder.h)

  void open() {
    sb_open();
  }
  void close() {
    sb_close();
  }

  void set_matrices(fan::opengl::cid_t* cid, fan::opengl::matrices_list_NodeReference_t n) {
    sb_set_key<instance_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
  }

  void set_viewport(fan::opengl::cid_t* cid, fan::opengl::viewport_list_NodeReference_t n) {
    sb_set_key<instance_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
  }
};

#include _FAN_PATH(graphics/opengl/2D/objects/hardcode_close.h)