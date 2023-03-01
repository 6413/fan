struct sprite1_t {

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

  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(instance_t) / 4));

  typedef fan::masterpiece_t<
    fan::opengl::textureid_t<0>,
    fan::opengl::textureid_t<1>,
    fan::opengl::camera_list_NodeReference_t,
    fan::opengl::viewport_list_NodeReference_t
  >instance_properties_t;

  struct properties_t : instance_t {
    union {
      struct {
        fan::opengl::textureid_t<0> image;
        fan::opengl::textureid_t<1> light_map;
        fan::opengl::camera_list_NodeReference_t camera;
        fan::opengl::viewport_list_NodeReference_t viewport;
      };
      instance_properties_t instance_properties;
    };
  };

  void push_back(loco_t* loco, fan::opengl::cid_t* cid, properties_t& p) {
    sb_push_back(loco, cid, p);
  }
  void erase(loco_t* loco, fan::opengl::cid_t* cid) {
    sb_erase(loco, cid);
  }

  void draw(loco_t* loco) {
    sb_draw(loco);
  }

  #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/sprite.vs)
  #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/sprite1.fs)
  #include _FAN_PATH(graphics/opengl/2D/objects/shape_builder.h)

  void open(loco_t* loco) {
    sb_open(loco);
  }
  void close(loco_t* loco) {
    sb_close(loco);
  }
};