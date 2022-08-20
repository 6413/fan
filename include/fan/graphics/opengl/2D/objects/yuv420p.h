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
      fan::opengl::textureid_t<1>,
      fan::opengl::textureid_t<2>,
      fan::opengl::matrices_list_NodeReference_t,
      fan::opengl::viewport_list_NodeReference_t
    > {}key;
  };

  struct properties_t : instance_t {

    void load_yuv(fan::opengl::context_t* context, uint8_t* data, const fan::vec2& image_size) {

      fan::opengl::image_t image[3];

      uint64_t offset = 0;

      fan::opengl::image_t::load_properties_t lp;
      lp.format = fan::opengl::GL_RED;
      lp.internal_format = fan::opengl::GL_RED;

      offset += image_size.multiply();

      fan::webp::image_info_t ii;

      ii.data = data;
      ii.size = image_size;

      image[0].load(context, ii, lp);

      ii.data = data + offset;
      ii.size = image_size / 2;
      image[1].load(context, ii, lp);

      offset += image_size.multiply() / 4;

      ii.data = data + offset;
      image[2].load(context, ii, lp);

      y = &image[0];
      u = &image[1];
      v = &image[2];
    }

    union {
      struct {
        fan::opengl::textureid_t<0> y;
        fan::opengl::textureid_t<1> u;
        fan::opengl::textureid_t<2> v;
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
  #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/yuv420p.fs)
  #endif
  #include _FAN_PATH(graphics/opengl/2D/objects/shape_builder.h)
  #undef sb_shader_vertex_path
  #undef sb_shader_fragment_path

  void open() {
    sb_open();
  }
  void close() {
    sb_close();
  }

  void set_image_data(fan::opengl::cid_t* cid, 
    fan::opengl::textureid_t<0> y,
    fan::opengl::textureid_t<1> u,
    fan::opengl::textureid_t<2> v
  ) {
    sb_set_key(cid, &properties_t::y, y);
    sb_set_key(cid, &properties_t::u, u);
    sb_set_key(cid, &properties_t::v, v);
  }

  /*void set_matrices(loco_t* loco, fan::opengl::cid_t* cid, fan::opengl::matrices_list_NodeReference_t n) {
  auto block = sb_get_block(loco, cid);
  *block->p[cid->instance_id].key.get_value<0>() = n;
  }

  void set_viewport(loco_t* loco, fan::opengl::cid_t* cid, fan::opengl::viewport_list_NodeReference_t n) {
  auto block = sb_get_block(loco, cid);
  *block->p[cid->instance_id].key.get_value<1>() = n;
  }*/
};

#undef sb_sprite_name