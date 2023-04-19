struct sb_shape_name {

  loco_bdbt_NodeReference_t root;

  static constexpr typename loco_t::shape_type_t::_t shape_type = loco_t::shape_type_t::light;

  struct vi_t {
    loco_light_vi_t
  };

  struct bm_properties_t {
    using parsed_masterpiece_t = fan::masterpiece_t<
      uint16_t,
      loco_t::camera_list_NodeReference_t,
      fan::graphics::viewport_list_NodeReference_t
    >;
    struct key_t : parsed_masterpiece_t {}key;
  };

  struct cid_t;

  struct ri_t : bm_properties_t {
    loco_light_ri_t
  };

  #define make_key_value(type, name) \
    type& name = *key.get_value<decltype(key)::get_index_with_type<type>()>();

  struct properties_t : vi_t, ri_t {
    using type_t = sb_shape_name;
    loco_light_properties_t
  };

  #undef make_key_value

  void push_back(loco_t::cid_nt_t& id, properties_t& p) {

    get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;

    shape_bm_NodeReference_t bm_id;

    {
      loco_bdbt_NodeReference_t key_root = root;

      loco_bdbt_Key_t<sizeof(bm_properties_t::key_t) * 8> k;
      typename decltype(k)::KeySize_t ki;
      k.q(&gloco->bdbt, &p.key, &ki, &key_root);
      if(ki != sizeof(bm_properties_t::key_t) * 8){
        bm_id = push_new_bm(p);
        k.a(&gloco->bdbt, &p.key, ki, key_root, bm_id.NRI);
      }
      else {
        bm_id.NRI = key_root;
      }
    };

    vi_t it = p;
    shape_bm_Node_t* bmn = bm_list.GetNodeByReference(bm_id);
    block_t* last_block = &blocks[bmn->data.last_block].block;

    if (last_block->uniform_buffer.size() == max_instance_size) {
      auto nnr = blocks.NewNode();
      blocks.linkNext(bmn->data.last_block, nnr);
      bmn->data.last_block = nnr;
      last_block = &blocks[bmn->data.last_block].block;
      last_block->open(gloco, this);
    }
    block_t* block = last_block;
    block->uniform_buffer.push_ram_instance(gloco->get_context(), it);

    const uint32_t instance_id = block->uniform_buffer.size() - 1;

    block->id[instance_id] = id;

    block->uniform_buffer.common.edit(
      gloco->get_context(),
      &gloco->m_write_queue,
      instance_id * sizeof(vi_t),
      instance_id * sizeof(vi_t) + sizeof(vi_t)
    );

    id->bm_id = bm_id.NRI;
    id->block_id = bmn->data.last_block.NRI;
    id->instance_id = instance_id;
    id->shape_type = -1;

    gloco->types.iterate([&]<typename T>(auto shape_index, T shape) {
      using shape_t = std::remove_pointer_t<std::remove_pointer_t<T>>;
      if constexpr (std::is_same_v<shape_t, std::remove_reference_t<decltype(*this)>>) {
        id->shape_type = shape_t::shape_type;
      }
    });

    block->p[instance_id] = *(ri_t*)&p;
  }
  void erase(loco_t::cid_nt_t& id) {
    auto bm_id = *(shape_bm_NodeReference_t*)&id->bm_id;
    auto bm_node = bm_list.GetNodeByReference(bm_id);

    auto block_id = *(bll_block_NodeReference_t*)&id->block_id;
    auto block_node = blocks.GetNodeByReference(*(bll_block_NodeReference_t*)&id->block_id);
    auto block = &block_node->data.block;

    auto& last_block_id = bm_node->data.last_block;
    auto* last_block_node = blocks.GetNodeByReference(last_block_id);
    block_t* last_block = &last_block_node->data.block;
    uint32_t last_instance_id = last_block->uniform_buffer.size() - 1;

    if (block_id == last_block_id && id->instance_id == block->uniform_buffer.size() - 1) {
      block->uniform_buffer.m_size -= sizeof(vi_t);
      if (block->uniform_buffer.size() == 0) {
        auto lpnr = block_node->PrevNodeReference;
        if (last_block_id == bm_node->data.first_block) {
          loco_bdbt_Key_t<sizeof(bm_properties_t::key_t) * 8> k;
          typename decltype(k)::KeySize_t ki;
          k.r(&gloco->bdbt, &bm_node->data.instance_properties.key, root);
          bm_list.Recycle(bm_id);
        }
        else {
          //fan::print("here");
          last_block_id = lpnr;
        }
        block->close(gloco);
        blocks.Unlink(block_id);
        blocks.Recycle(block_id);
      }
      id->bm_id = 0;
      id->block_id = 0;
      id->instance_id = 0;
      id->instance_id = -1;
      return;
    }

    vi_t* last_instance_data = last_block->uniform_buffer.get_instance(gloco->get_context(), last_instance_id);

    block->uniform_buffer.copy_instance(
      gloco->get_context(),
      &gloco->m_write_queue,
      id->instance_id,
      last_instance_data
    );

    last_block->uniform_buffer.m_size -= sizeof(vi_t);

    block->p[id->instance_id] = last_block->p[last_instance_id];

    block->id[id->instance_id] = last_block->id[last_instance_id];
    block->id[id->instance_id]->block_id = block_id.NRI;
    block->id[id->instance_id]->instance_id = id->instance_id;

    if (last_block->uniform_buffer.size() == 0) {
      auto lpnr = last_block_node->PrevNodeReference;

      last_block->close(gloco);
      blocks.Unlink(last_block_id);
      blocks.Recycle(last_block_id);

      bm_node->data.last_block = lpnr;
    }
    id->bm_id = 0;
    id->block_id = 0;
    id->instance_id = 0;
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
    gloco->get_context()->opengl.call(gloco->get_context()->opengl.glBlendFunc, fan::opengl::GL_ONE, fan::opengl::GL_ONE);

    unsigned int attachments[sizeof(gloco->color_buffers) / sizeof(gloco->color_buffers[0])];

    for (uint8_t i = 0; i < std::size(gloco->color_buffers); ++i) {
      attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
    }

    gloco->get_context()->opengl.call(gloco->get_context()->opengl.glDrawBuffers, std::size(attachments), attachments);

    //
    sb_draw(root);
    //
    gloco->get_context()->set_depth_test(true);

    for (uint8_t i = 0; i < std::size(gloco->color_buffers); ++i) {
      attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
    }
    gloco->get_context()->opengl.call(gloco->get_context()->opengl.glDrawBuffers, 1, attachments);
  }

  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(vi_t) / 4));
  #if defined(loco_opengl)
  #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/light.vs)
  #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/sb_fragment_shader)
  #elif defined(loco_vulkan)
  #endif

  #define vk_sb_ssbo
  #define vk_sb_vp
  //#define sb_inline_draw
  #include _FAN_PATH(graphics/shape_builder.h)

  sb_shape_name() {
    root = loco_bdbt_NewNode(&gloco->bdbt);

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
  }
  ~sb_shape_name() {
    sb_close();
  }

  void set_camera(loco_t::cid_nt_t& id, loco_t::camera_list_NodeReference_t n) {
    sb_set_key<bm_properties_t::key_t::get_index_with_type<decltype(n)>()>(id, n);
  }

  void set_viewport(loco_t::cid_nt_t& id, fan::graphics::viewport_list_NodeReference_t n) {
    sb_set_key<bm_properties_t::key_t::get_index_with_type<decltype(n)>()>(id, n);
  }

  properties_t get_properties(loco_t::cid_nt_t& id) {
    properties_t p = sb_get_properties(id);
    p.camera = gloco->camera_list[*p.key.get_value<loco_t::camera_list_NodeReference_t>()].camera_id;
    p.viewport = gloco->get_context()->viewport_list[*p.key.get_value<fan::graphics::viewport_list_NodeReference_t>()].viewport_id;
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