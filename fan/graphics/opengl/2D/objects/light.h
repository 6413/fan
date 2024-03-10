struct sb_shape_name {

  static constexpr typename loco_t::shape_type_t shape_type = loco_t::shape_type_t::light;

  struct vi_t {
    loco_t::position3_t position = 0;
    f32_t parallax_factor = 0;
    fan::vec2 size = 0;
    fan::vec2 rotation_point = 0;
    fan::color color = fan::colors::white;
    fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
    f32_t pad;
    fan::vec3 angle = 0;
    f32_t pad2;
  };

  struct context_key_t {
    using parsed_masterpiece_t = fan::masterpiece_t<
      loco_t::camera_list_NodeReference_t,
      fan::graphics::viewport_list_NodeReference_t
    >;
    struct key_t : parsed_masterpiece_t {}key;
  };

  struct cid_t;

  struct ri_t {
    bool blending = true;
  };

  struct properties_t : vi_t, ri_t, context_key_t {
    using type_t = sb_shape_name;
    loco_t::camera_t* camera = &gloco->default_camera->camera;
    loco_t::viewport_t* viewport = &gloco->default_camera->viewport;
  };

  void push_back(loco_t::cid_nt_t& id, properties_t& p) {
    get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;

   sb_push_back(id, p);
  }
  void erase(loco_t::cid_nt_t& id) {
    sb_erase(id);
  }

  void draw() {
    gloco->get_context().set_depth_test(false);
    gloco->get_context().opengl.call(gloco->get_context().opengl.glEnable, fan::opengl::GL_BLEND);
    gloco->get_context().opengl.call(gloco->get_context().opengl.glBlendFunc, fan::opengl::GL_ONE, fan::opengl::GL_ONE);

    #if defined(loco_framebuffer)
    unsigned int attachments[sizeof(gloco->color_buffers) / sizeof(gloco->color_buffers[0])];

    for (uint8_t i = 0; i < std::size(gloco->color_buffers); ++i) {
      attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
    }

    gloco->get_context().opengl.call(gloco->get_context().opengl.glDrawBuffers, std::size(attachments), attachments);
    #endif
    //
    sb_draw(key_root);
    //
    gloco->get_context().set_depth_test(true);

    #if defined(loco_framebuffer)
    for (uint8_t i = 0; i < std::size(gloco->color_buffers); ++i) {
      attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
    }
    gloco->get_context().opengl.call(gloco->get_context().opengl.glDrawBuffers, 1, attachments);
    #endif
  }

  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(vi_t) / 4));
  #if defined(loco_opengl)
    #define sb_shader_vertex_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/light.vs)
    #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/light.fs)
  #endif

  sb_shape_name() {

    sb_open(sb_shader_vertex_path, sb_shader_fragment_path);

    m_current_shader = &m_blending_shader;
    gloco->m_draw_queue_light.push_back([&] {
      draw();
    });
  }
  ~sb_shape_name() {
    sb_close();
  }


  #define sb_has_own_key_root 1
  #define sb_ignore_3_key 1
  #include _FAN_PATH(graphics/shape_builder.h)

  //void set_camera(loco_t::cid_nt_t& id, loco_t::camera_list_NodeReference_t n) {
  //  sb_set_context_key<loco_t::camera_list_NodeReference_t>(id, n);
  //}

  //void set_viewport(loco_t::cid_nt_t& id, fan::graphics::viewport_list_NodeReference_t n) {
  //  sb_set_context_key<fan::graphics::viewport_list_NodeReference_t>(id, n);
  //}

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