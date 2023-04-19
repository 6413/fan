struct line_t {

  static constexpr typename loco_t::shape_type_t::_t shape_type = loco_t::shape_type_t::line;

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
    bool blending = false;
	};

  struct properties_t : vi_t, ri_t {
    using type_t = line_t;
    loco_t::camera_t* camera = 0;
    fan::graphics::viewport_t* viewport = 0;
  };

  #undef make_key_value
  
  void push_back(loco_t::cid_nt_t& id, properties_t& p) {
    
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

    sb_push_back(id, p);
  }
  void erase(loco_t::cid_nt_t& id) {
    sb_erase(id);
  }

  void draw(const redraw_key_t &redraw_key, loco_bdbt_NodeReference_t key_root) {
    if (redraw_key.blending) {
      m_current_shader = &m_blending_shader;
    }
    else {
      m_current_shader = &m_shader;
    }
    sb_draw(key_root, fan::opengl::GL_LINES);
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

  void set_line(loco_t::cid_nt_t& id, const fan::vec3& src, const fan::vec3& dst) {
    auto v = sb_get_vi(id);
    if (v.src.z != src.z) {
      sb_set_depth(id, src.z);
    }
    set(id, &vi_t::src, src);
    set(id, &vi_t::dst, dst);
  }

  properties_t get_properties(loco_t::cid_nt_t& id) {
    properties_t p = sb_get_properties(id);
    p.camera = get_loco()->camera_list[*p.key.get_value<loco_t::camera_list_NodeReference_t>()].camera_id;
    p.viewport = get_loco()->get_context()->viewport_list[*p.key.get_value<fan::graphics::viewport_list_NodeReference_t>()].viewport_id;
    return p;
  }

  #if defined(loco_vulkan)
    uint32_t m_camera_index = 0;
  #endif
};