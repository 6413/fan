struct rectangle_t {

  struct instance_t {
    fan::vec3 position = 0;
  private:
    f32_t pad[1];
  public:
    fan::vec2 size = 0;
    fan::vec2 rotation_point = 0;
    fan::color color = fan::colors::white;
    fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
    f32_t angle = 0;
  };

  static constexpr uint32_t max_instance_size = std::min(256ull, 4096 / (sizeof(instance_t) / 4));

  typedef fan::masterpiece_t<
    fan::opengl::matrices_list_NodeReference_t,
    fan::opengl::viewport_list_NodeReference_t
  >
  block_properties_t;
      
  struct properties_t : instance_t {
    union {
      struct {
        fan::opengl::matrices_list_NodeReference_t matrices;
        fan::opengl::viewport_list_NodeReference_t viewport;
      };
      block_properties_t block_properties;
    };
  };

  void push_back(loco_t* loco, fan::opengl::cid_t* cid, properties_t& p) {
    sb_push_back(loco, cid, p);
    block_properties_t::get_type<0>::type x;
  }

  void draw(loco_t* loco) {
    sb_draw(loco);
  }

  #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle.vs)
  #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle.fs)
  #include _FAN_PATH(graphics/opengl/2D/objects/shape_builder.h)

  void open(loco_t* loco) {
    sb_open(loco);
  }
  void close(loco_t* loco) {
    sb_close(loco);
  }
};