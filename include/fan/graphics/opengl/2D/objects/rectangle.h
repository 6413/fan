struct rectangle_t {

  static constexpr typename loco_t::shape_type_t::_t shape_type = 6;

  struct vi_t {
    loco_rectangle_vi_t
  };

  struct bm_properties_t {
    loco_rectangle_bm_properties_t
  };

  struct cid_t;

  struct ri_t : bm_properties_t {
    loco_rectangle_ri_t
  };

  struct properties_t : vi_t, ri_t {
    using type_t = rectangle_t;
    loco_rectangle_properties_t
  };

  void push_back(fan::graphics::cid_t* cid, properties_t p) {
    get_key_value(uint16_t) = p.position.z;
    get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;

    #if defined(loco_vulkan)
    auto loco = get_loco();
    auto& camera = loco->camera_list[p.camera];
    if (camera.camera_index.rectangle == (decltype(camera.camera_index.rectangle))-1) {
      camera.camera_index.rectangle = m_camera_index++;
      m_shader.set_camera(loco, camera.camera_id, camera.camera_index.rectangle);
    }
    #endif

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
  #elif defined(loco_vulkan)
  #if defined(loco_wboit)
  #define vulkan_buffer_count 4
  #else
  #define vulkan_buffer_count 4
  #endif

  #define sb_shader_vertex_path graphics/glsl/vulkan/2D/objects/rectangle.vert
  #define sb_shader_fragment_path graphics/glsl/vulkan/2D/objects/rectangle.frag
  #endif

  #define vk_sb_ssbo
  #define vk_sb_vp
  #include _FAN_PATH(graphics/shape_builder.h)

  rectangle_t() {
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

  #if defined(loco_vulkan)
  #if defined (vk_shape_wboit)
  fan::vulkan::shader_t render_fullscreen_shader;
  #endif

  uint32_t m_camera_index = 0;
  #endif

};

#undef vulkan_buffer_count