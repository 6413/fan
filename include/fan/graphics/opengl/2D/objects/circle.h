struct circle_t {

  static constexpr typename loco_t::shape_type_t::_t shape_type = loco_t::shape_type_t::circle;

  struct vi_t {
    fan::vec3 position = 0;
    f32_t radius = 0;
    fan::vec2 rotation_point = 0;
  private:
    f32_t pad[2];
  public:
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
    cid_t* cid;
    bool blending = false;
  };

  #define make_key_value(type, name) \
    type& name = *key.get_value<decltype(key)::get_index_with_type<type>()>();

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

  #undef make_key_value

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
  #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/circle.vs)
  #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/circle.fs)
  #elif defined(loco_vulkan)
  #if defined(loco_wboit)
  #define vulkan_buffer_count 4
  #else
  #define vulkan_buffer_count 4
  #endif

  #define sb_shader_vertex_path graphics/glsl/vulkan/2D/objects/circle.vert
  #define sb_shader_fragment_path graphics/glsl/vulkan/2D/objects/circle.frag
  #endif

  #define vk_sb_ssbo
  #define vk_sb_vp
  #include _FAN_PATH(graphics/shape_builder.h)

  circle_t() {
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
  ~circle_t() {
    sb_close();
  }

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