struct rectangle_t {

  struct vi_t {
    loco_rectangle_vi_t
  };

  struct bm_properties_t {
    loco_rectangle_bm_properties_t
  };

  struct cid_t;

	struct ri_t {
    loco_rectangle_ri_t
	};

  #pragma pack(push, 1)
  struct properties_t {
    loco_rectangle_properties_t

    make_address_wrapper(vi);
    loco_rectangle_vi_t;

    make_address_wrapper(ri);
    loco_rectangle_ri_t;

    make_address_wrapper(bm_properties);
    loco_rectangle_bm_properties_t
  };
  #pragma pack(pop)

  void push_back(fan::graphics::cid_t* cid, properties_t p) {
    get_key_value(uint16_t) = p.position.z;
    get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;

    sb_push_back(cid, p);
  }
  void erase(fan::graphics::cid_t* cid) {
    sb_erase(cid);
  }

  void draw() {
    sb_draw();
  }

  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(vi_t) / 4));
  #if defined(loco_opengl)
    #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle.vs)
    #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle.fs)
  #endif

  #include _FAN_PATH(graphics/shape_builder.h)

  rectangle_t() {
    sb_open();
  }
  ~rectangle_t() {
    sb_close();
  }

  void set_camera(fan::graphics::cid_t* cid, loco_t::camera_list_NodeReference_t n) {
    sb_set_key<bm_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
  }

  void set_viewport(fan::graphics::cid_t* cid, fan::graphics::viewport_list_NodeReference_t n) {
    sb_set_key<bm_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
  }
};