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

  static constexpr uint32_t max_instance_size = std::min(256ull, 4096 / (sizeof(instance_t) / 4));

  typedef fan::masterpiece_t<
    fan::opengl::matrices_list_NodeReference_t,
    fan::opengl::viewport_list_NodeReference_t
  >
    block_properties_t;

  struct properties_t : instance_t {
    f32_t font_size = 16;
    uint16_t letter_id;
    union {
      struct {
        fan::opengl::matrices_list_NodeReference_t matrices;
        fan::opengl::viewport_list_NodeReference_t viewport;
      };
      block_properties_t block_properties;
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

  void draw(loco_t* loco) {
    m_shader.use(loco->get_context());
    m_shader.set_int(loco->get_context(), "_t00", 0);
    loco->get_context()->opengl.call(loco->get_context()->opengl.glActiveTexture, fan::opengl::GL_TEXTURE0);
    loco->font.image.bind_texture(loco->get_context());
    
    sb_draw(loco);
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
};