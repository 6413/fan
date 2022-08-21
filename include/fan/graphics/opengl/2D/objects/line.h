struct line_t {

  struct instance_t {
    fan::color color;
    fan::vec3 src;
  private:
    f32_t pad;
  public:
    fan::vec2 dst;
  private:
    f32_t pad2[2];
  };

  struct instance_properties_t {
    struct key_t : fan::masterpiece_t<
      fan::opengl::matrices_list_NodeReference_t,
      fan::opengl::viewport_list_NodeReference_t
    > {}key;
  };

  struct properties_t : instance_t {
    union {
      struct {
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
    sb_draw(fan::opengl::GL_LINES);
  }

  static constexpr uint32_t max_instance_size = fan::min(256ull, 4096 / (sizeof(instance_t) / 4));
  #define sb_vertex_count 2
  #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/line.vs)
  #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/line.fs)
  #include _FAN_PATH(graphics/opengl/2D/objects/shape_builder.h)

  void open() {
    sb_open();
  }
  void close() {
    sb_close();
  }

  void set_line(fan::opengl::cid_t* cid, const fan::vec3& src, const fan::vec3& dst) {
    set(cid, &instance_t::src, src);
    set(cid, &instance_t::dst, dst);
  }
};