struct sb_shape_name {

  static constexpr typename loco_t::shape_type_t shape_type = loco_t::shape_type_t::light;

  struct vi_t {
    loco_t::position3_t position = 0;
    f32_t parallax_factor = 0;
    fan::vec2 size = 0;
    fan::vec2 rotation_point = 0;
    fan::color color = fan::colors::white;
    fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
    f32_t angle = 0;
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

  //void draw(bool blending = false) {
  //   /* #if defined(loco_framebuffer)
  //    #if defined(sb_is_light)

  //    loco->get_context()->set_depth_test(false);
  //    if constexpr (std::is_same<std::remove_pointer_t<decltype(this)>, sb_shape_name>::value) {
  //      loco->get_context()->opengl.call(loco->get_context()->opengl.glBlendFunc, fan::opengl::GL_ONE, fan::opengl::GL_ONE);

  //      unsigned int attachments[sizeof(loco->color_buffers) / sizeof(loco->color_buffers[0])];

  //      for (uint8_t i = 0; i < std::size(loco->color_buffers); ++i) {
  //        attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
  //      }

  //      loco->get_context()->opengl.call(loco->get_context()->opengl.glDrawBuffers, std::size(attachments), attachments);

  //    }
  //    #endif
  //    #endif*/
  //  sb_draw(gloco->root);
  //  /*
  //   #if defined(loco_framebuffer)
  //    #if defined(sb_is_light)
  //    loco->get_context()->opengl.call(loco->get_context()->opengl.glDisable, fan::opengl::GL_BLEND);
  //    loco->get_context()->set_depth_test(true);
  //    if constexpr (std::is_same<std::remove_pointer_t<decltype(this)>, sb_shape_name>::value) {
  //      loco->get_context()->opengl.call(loco->get_context()->opengl.glBlendFunc, fan::opengl::GL_SRC_ALPHA, fan::opengl::GL_ONE_MINUS_SRC_ALPHA);
  //      unsigned int attachments[sizeof(loco->color_buffers) / sizeof(loco->color_buffers[0])];

  //      for (uint8_t i = 0; i < std::size(loco->color_buffers); ++i) {
  //        attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
  //      }

  //      loco->get_context()->opengl.call(loco->get_context()->opengl.glDrawBuffers, 1, attachments);
  //    }
  //    #endif
  //    #endif
  //  */
  //}

  void draw() {
    gloco->get_context()->set_depth_test(false);
    gloco->get_context()->opengl.call(gloco->get_context()->opengl.glEnable, fan::opengl::GL_BLEND);
    gloco->get_context()->opengl.call(gloco->get_context()->opengl.glBlendFunc, fan::opengl::GL_ONE, fan::opengl::GL_ONE);

    unsigned int attachments[sizeof(gloco->color_buffers) / sizeof(gloco->color_buffers[0])];

    for (uint8_t i = 0; i < std::size(gloco->color_buffers); ++i) {
      attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
    }

    gloco->get_context()->opengl.call(gloco->get_context()->opengl.glDrawBuffers, std::size(attachments), attachments);

    //
    sb_draw(key_root);
    //
    gloco->get_context()->set_depth_test(true);

    for (uint8_t i = 0; i < std::size(gloco->color_buffers); ++i) {
      attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
    }
    gloco->get_context()->opengl.call(gloco->get_context()->opengl.glDrawBuffers, 1, attachments);
  }

  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(vi_t) / 4));
  #if defined(loco_opengl)
    #define sb_shader_vertex_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/light.vs)
    #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/light.fs)
  #endif

  #define sb_has_own_key_root 1
  #define sb_ignore_3_key 1
  #include _FAN_PATH(graphics/shape_builder.h)

  sb_shape_name() {

    sb_open();

    #if defined(loco_wboit) && defined(vk_shape_wboit) && defined(loco_vulkan)
    fan::vulkan::pipeline_t::properties_t p;

    auto loco = get_loco();
    auto context = loco->get_context();

    render_fullscreen_shader.open(context, &loco->m_write_queue);
    render_fullscreen_shader.set_vertex(context, _FAN_PATH_QUOTE(graphics / glsl / vulkan / 2D / objects / fullscreen.vert.spv));
    render_fullscreen_shader.set_fragment(context, _FAN_PATH_QUOTE(graphics / glsl / vulkan / 2D / objects / fullscreen.frag.spv));
    p.descriptor_layout_count = 1;
    p.descriptor_layout = &m_ssbo.m_descriptor.m_layout;
    p.shader = &render_fullscreen_shader;
    p.push_constants_size = p.push_constants_size = sizeof(loco_t::push_constants_t);
    p.subpass = 1;
    VkPipelineColorBlendAttachmentState color_blend_attachment{};
    color_blend_attachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT |
      VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT |
      VK_COLOR_COMPONENT_A_BIT
      ;
    color_blend_attachment.blendEnable = VK_TRUE;
    color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
    p.color_blend_attachment_count = 1;
    p.color_blend_attachment = &color_blend_attachment;
    p.enable_depth_test = false;
    context->render_fullscreen_pl.open(context, p);

    #endif

    m_current_shader = &m_blending_shader;
    gloco->m_draw_queue_light.push_back([&] {
      draw();
    });
  }
  ~sb_shape_name() {
    sb_close();
  }

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