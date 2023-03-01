struct line_t {

  struct vi_t {
    fan::color color;
    fan::vec3 src;
  private:
    f32_t pad;
  public:
    fan::vec3 dst;
  private:
    f32_t pad2[1];
  };

  struct bm_properties_t {
     using parsed_masterpiece_t = fan::masterpiece_t<
      uint16_t,
      loco_t::camera_list_NodeReference_t,
      fan::graphics::viewport_list_NodeReference_t
    >;
    struct key_t : parsed_masterpiece_t {
    }key;
  };

  struct cid_t;

	struct ri_t : bm_properties_t {
		cid_t* cid;
	};

  struct properties_t : vi_t, ri_t {
    loco_t::camera_t* camera = 0;
    fan::graphics::viewport_t* viewport = 0;
  };

  #undef make_key_value
  
  void push_back(fan::graphics::cid_t* cid, properties_t& p) {
    
    get_key_value(uint16_t) = p.src.z;
    get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;

    #if defined(loco_vulkan)
      auto loco = get_loco();
      auto& camera = loco->camera_list[p.camera];
      if (camera.camera_index.line == (decltype(camera.camera_index.line))-1) {
        camera.camera_index.line = m_camera_index++;
        m_shader.set_camera(loco, camera.camera_id, camera.camera_index.line);
      }
    #endif

    sb_push_back(cid, p);
  }
  void erase(fan::graphics::cid_t* cid) {
    sb_erase(cid);
  }

  void draw() {
    sb_draw(
      #if defined (loco_opengl)
        fan::opengl::GL_LINES
      #endif
    );
  }

  static constexpr uint32_t max_instance_size = fan::min(256ull, 4096 / (sizeof(vi_t) / 4));

  #define sb_vertex_count 2

 #if defined(loco_opengl)
    #ifndef sb_shader_vertex_path
      #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/line.vs)
    #endif
    #ifndef sb_shader_fragment_path
      #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/line.fs)
    #endif
  #elif defined(loco_vulkan)
    #define vulkan_buffer_count 4
    #define sb_shader_vertex_path graphics/glsl/vulkan/2D/objects/line.vert
    #define sb_shader_fragment_path graphics/glsl/vulkan/2D/objects/line.frag
  #endif

  #define vk_sb_ssbo
  #define vk_sb_vp
  #include _FAN_PATH(graphics/shape_builder.h)

  line_t() {
    sb_open();
  }
  ~line_t() {
    sb_close();
  }

  void set_line(fan::graphics::cid_t* cid, const fan::vec3& src, const fan::vec3& dst) {
    auto v = sb_get_vi(cid);
    if (v.src.z != src.z) {
      sb_set_depth(cid, src.z);
    }
    set(cid, &vi_t::src, src);
    set(cid, &vi_t::dst, dst);
  }

  #if defined(loco_vulkan)
    uint32_t m_camera_index = 0;
  #endif
};