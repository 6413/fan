struct circle_t {

  static constexpr typename loco_t::shape_type_t shape_type = loco_t::shape_type_t::circle;

  struct vi_t {
    fan::vec3 position = 0;
    f32_t radius = 0;
    fan::vec2 rotation_point = 0;
  private:
    f32_t pad[2];
  public:
    fan::color color = fan::colors::white;
    fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
    fan::vec3 angle = 0;
  };

  struct context_key_t {
    using parsed_masterpiece_t = fan::masterpiece_t<
      uint16_t,
      loco_t::camera_list_NodeReference_t,
      fan::graphics::viewport_list_NodeReference_t
    >;
    struct key_t : parsed_masterpiece_t {}key;
  };

  struct cid_t;

  struct ri_t {
    cid_t* cid;
    bool blending = false;
  };

  struct properties_t : vi_t, ri_t {
    /*todo cloned from context_key_t - make define*/
    using parsed_masterpiece_t = fan::masterpiece_t<
      uint16_t,
      loco_t::camera_list_NodeReference_t,
      fan::graphics::viewport_list_NodeReference_t
    >;
    struct key_t : parsed_masterpiece_t {}key;

    using type_t = circle_t;

    loco_t::camera_t* camera = 0;
    fan::graphics::viewport_t* viewport = 0;
  };

  void push_back(loco_t::cid_nt_t& id, properties_t& p) {
    #if defined(loco_vulkan)
    auto loco = get_loco();
    auto& camera = loco->camera_list[p.camera];
    if (camera.camera_index.rectangle == (decltype(camera.camera_index.rectangle))-1) {
      camera.camera_index.rectangle = m_camera_index++;
      m_shader.set_camera(loco, camera.camera_id, camera.camera_index.rectangle);
    }
    #endif

    get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;

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
    sb_draw(key_root);
  }

  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(vi_t) / 4));
  #if defined(loco_opengl)
    #define sb_shader_vertex_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/circle.vs)
    #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/circle.fs)
  #endif

  circle_t() {
    sb_open(sb_shader_vertex_path, sb_shader_fragment_path);
  }
  ~circle_t() {
    sb_close();
  }

  #include _FAN_PATH(graphics/shape_builder.h)

  void set_camera(loco_t::cid_nt_t& id, loco_t::camera_list_NodeReference_t n) {
    sb_set_context_key<decltype(n)>(id, n);
  }

  void set_viewport(loco_t::cid_nt_t& id, fan::graphics::viewport_list_NodeReference_t n) {
    sb_set_context_key<decltype(n)>(id, n);
  }

  fan::vec2 get_size(loco_t::cid_nt_t& id) {
    return sb_get_vi(id).radius;
  }
  
  properties_t get_properties(loco_t::cid_nt_t& id) {
    properties_t p = sb_get_properties(id);
    return p;
  }

  #if defined(loco_vulkan)
  #if defined (vk_shape_wboit)
  fan::vulkan::shader_t render_fullscreen_shader;
  #endif

  uint32_t m_camera_index = 0;
  #endif

};

#undef vulkan_buffer_count