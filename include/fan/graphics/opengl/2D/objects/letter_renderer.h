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

  struct instance_properties_t {
    struct key_t : fan::masterpiece_t<
      fan::opengl::matrices_list_NodeReference_t,
      fan::opengl::viewport_list_NodeReference_t
    > {}key;
    f32_t font_size;
    uint16_t letter_id;
  };

  struct properties_t : instance_t {
    properties_t() : font_size(16) {}
    properties_t(const instance_t& it) {
      position = it.position;
      outline_size = it.outline_size;
      size = it.size;
      tc_position = it.tc_position;
      color = it.color;
      outline_color = it.outline_color;
      tc_size = it.tc_size;
    }
    union {
      struct {
        fan::opengl::matrices_list_NodeReference_t matrices;
        fan::opengl::viewport_list_NodeReference_t viewport;
        f32_t font_size;
        uint16_t letter_id;
      };
      instance_properties_t instance_properties;
    };
  };

  void push_back(loco_t* loco, fan::opengl::cid_t* cid, properties_t& p) {
    fan::font::single_info_t si = loco->font.info.get_letter_info(p.letter_id, p.font_size);

    p.tc_position = si.glyph.position / loco->font.image.size;
    p.tc_size.x = si.glyph.size.x / loco->font.image.size.x;
    p.tc_size.y = si.glyph.size.y / loco->font.image.size.y;

    p.size = si.metrics.size / 2;

    sb_push_back(loco, cid, p);
  }
  void erase(loco_t* loco, fan::opengl::cid_t* cid) {
    sb_erase(loco, cid);
  }

  void draw(loco_t* loco) {
    m_shader.use(loco->get_context());
    m_shader.set_int(loco->get_context(), "_t00", 0);
    loco->get_context()->opengl.call(loco->get_context()->opengl.glActiveTexture, fan::opengl::GL_TEXTURE0);
    loco->font.image.bind_texture(loco->get_context());
    
    sb_draw(loco);
  }

  properties_t get_properties(loco_t* loco, fan::opengl::cid_t* cid) {
    properties_t p;
    p = *sb_get_block(loco, cid)->uniform_buffer.get_instance(loco->get_context(), cid->instance_id);
    p.instance_properties = sb_get_block(loco, cid)->p[cid->instance_id];
    return p;
  }

  #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/letter.vs)
  #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/letter.fs)
  #include _FAN_PATH(graphics/opengl/2D/objects/shape_builder.h)

  void open(loco_t* loco) {
    sb_open(loco);
  }
  void close(loco_t* loco) {
    sb_close(loco);
  }

  void set_matrices(loco_t* loco, fan::opengl::cid_t* cid, fan::opengl::matrices_list_NodeReference_t n) {
    sb_set_key(loco, cid, &properties_t::matrices, n);
  }

  void set_viewport(loco_t* loco, fan::opengl::cid_t* cid, fan::opengl::viewport_list_NodeReference_t n) {
    sb_set_key(loco, cid, &properties_t::viewport, n);
  }
};