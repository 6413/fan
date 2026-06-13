VkSampler post_process_sampler;

loco_t& get_loco() {
  return (*OFFSETLESS(this, loco_t, vk));
}
#define loco get_loco()

#if defined(FAN_2D)
void shapes_open() {
  struct shape_descriptor_t {
    uint16_t shape_type;
    std::size_t sizeof_vi;
    std::size_t sizeof_ri;
    fan::graphics::shape_gl_init_list_t locations;
    fan::graphics::shader_t shader;
    fan::graphics::shaper_t::ShapeRenderDataSize_t instance_count;
    bool instanced;
  };

#define SHAPE_DESC(shape, shader) \
  shape_descriptor_t{ \
    fan::graphics::shapes::shape##_t::shape_type, \
    sizeof(fan::graphics::shapes::shape##_t::vi_t), \
    sizeof(fan::graphics::shapes::shape##_t::ri_t), \
    fan::graphics::shape_gl_init_list_t{ \
      .ptr = fan::graphics::g_shapes->shape.get_locations().data(), \
      .count = static_cast<int>(fan::graphics::g_shapes->shape.get_locations().size()) \
    }, \
    shader, \
    1, \
    true \
  }

  auto& sh = loco.shaders;

  const shape_descriptor_t descriptors[] = {
    SHAPE_DESC(sprite, sh.sprite),
    SHAPE_DESC(rectangle, sh.rectangle),
  };

  for (auto& d : descriptors) {
    loco.shape_open(
      d.shape_type,
      d.sizeof_vi,
      d.sizeof_ri,
      d.locations,
      d.shader,
      d.instance_count,
      d.instanced
    );
  }

#undef SHAPE_DESC
}
#endif

#if defined(FAN_2D)
void shaders_compile() {
  auto& sh = loco.shaders;

  auto compile = [&](fan::graphics::shader_t& out, const char* vs, const char* fs) {
    out = loco.shader_create();
    loco.shader_set_vertex(out, vs, fan::graphics::read_shader(vs));
    loco.shader_set_fragment(out, fs, fan::graphics::read_shader(fs));
    loco.shader_compile(out);
  };

#define C(n) compile(sh.n, "shaders/vulkan/2D/objects/" #n ".vert", "shaders/vulkan/2D/objects/" #n ".frag")

  C(sprite);
  C(rectangle);

#undef C
}
#endif

void begin_render_pass() {
  fan::vulkan::context_t& context = loco.context.vk;
  VkRenderPassBeginInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = context.render_pass;
  renderPassInfo.framebuffer = context.swap_chain_framebuffers[context.image_index];
  renderPassInfo.renderArea.offset = { 0, 0 };
  renderPassInfo.renderArea.extent.width = context.swap_chain_size.x;
  renderPassInfo.renderArea.extent.height = context.swap_chain_size.y;

  // TODO
  VkClearValue clear_values[
    3
  ]{};
  fan::color clear_color = loco.get_clear_color();
  clear_values[0].color = { { clear_color.r, clear_color.g, clear_color.b, clear_color.a } };
  clear_values[1].color = { { clear_color.r, clear_color.g, clear_color.b, clear_color.a } };
  clear_values[2].depthStencil = { 1.0f, 0 };


  renderPassInfo.clearValueCount = std::size(clear_values);
  renderPassInfo.pClearValues = clear_values;

  if (loco.get_render_shapes_top()) {
  //  renderPassInfo.clearValueCount = 0;
  }


  vkCmdBeginRenderPass(context.command_buffers[context.current_frame], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
}

void begin_draw() {
  vkQueueWaitIdle(loco.context.vk.graphics_queue);
  fan::vulkan::context_t& context = loco.context.vk;
  vkWaitForFences(context.device, 1, &context.in_flight_fences[context.current_frame], VK_TRUE, UINT64_MAX);
    
  loco.vk.image_error = vkAcquireNextImageKHR(
    context.device,
    context.swap_chain,
    UINT64_MAX,
    context.image_available_semaphores[context.current_frame],
    VK_NULL_HANDLE,
    &context.image_index
  );

  if (loco.vk.image_error == VK_ERROR_OUT_OF_DATE_KHR || loco.vk.image_error == VK_SUBOPTIMAL_KHR) {
    context.recreate_swap_chain(&loco.window, loco.vk.image_error);
    loco.vk.image_error = vkAcquireNextImageKHR(
      context.device,
      context.swap_chain,
      UINT64_MAX,
      context.image_available_semaphores[context.current_frame],
      VK_NULL_HANDLE,
      &context.image_index
    );
  }

  if (loco.vk.image_error != VK_SUCCESS) { 
    context.command_buffer_in_use = false; 
    return; 
  }

  {
    VkDescriptorImageInfo mainInfo{};
    mainInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    mainInfo.imageView = context.mainColorImageViews[context.image_index].image_view;
    mainInfo.sampler = VK_NULL_HANDLE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = loco.vk.d_attachments.m_descriptor_set[context.current_frame];
    write.dstBinding = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    write.descriptorCount = 1;
    write.pImageInfo = &mainInfo;

    vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
  }

  vkResetFences(context.device, 1, &context.in_flight_fences[context.current_frame]);
  vkResetCommandBuffer(context.command_buffers[context.current_frame], /*VkCommandBufferResetFlagBits*/ 0);
  {
    context.image_pool.clear();

    fan::graphics::image_list_t::nrtra_t nrtra;
    fan::graphics::image_nr_t nr;

    std::uint32_t max_image_id = 0;
    VkDescriptorImageInfo fallback {};
    bool has_fallback = false;

    nrtra.Open(&loco.image_list, &nr);
    while (nrtra.Loop(&loco.image_list, &nr)) {
      auto img = loco.image_get(nr).vk;

      if (img.image_view == VK_NULL_HANDLE || img.sampler == VK_NULL_HANDLE) {
        continue;
      }

      max_image_id = std::max(max_image_id, (std::uint32_t)nr.NRI);

      if (!has_fallback) {
        fallback.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        fallback.imageView = img.image_view;
        fallback.sampler = img.sampler;
        has_fallback = true;
      }
    }
    nrtra.Close(&loco.image_list);

    if (has_fallback) {
      context.image_pool.assign(max_image_id + 1, fallback);

      nrtra.Open(&loco.image_list, &nr);
      while (nrtra.Loop(&loco.image_list, &nr)) {
        auto img = loco.image_get(nr).vk;

        if (img.image_view == VK_NULL_HANDLE || img.sampler == VK_NULL_HANDLE) {
          continue;
        }

        VkDescriptorImageInfo image_info {};
        image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image_info.imageView = img.image_view;
        image_info.sampler = img.sampler;
        context.image_pool[nr.NRI] = image_info;
      }
      nrtra.Close(&loco.image_list);
    }
  }

  #if defined(FAN_2D)
  {
    for (auto& st : fan::graphics::g_shapes->shaper.ShapeTypes) {
      if (st.sti == (decltype(st.sti))-1) {
        continue;
      }
      // TODO add more shapes here to enable textures
      if (st.sti != fan::graphics::shapes::shape_type_t::sprite) {
        continue;
      }
      auto& vk_data = st.renderer.vk;
      // todo slow, use only pointer
      vk_data.shape_data.m_descriptor.m_properties[2].image_infos = loco.context.vk.image_pool;
      // doesnt like multiple frames in flight
      vk_data.shape_data.m_descriptor.update(loco.context.vk, 1, 2, vk_data.shape_data.m_descriptor.m_properties[2].image_infos.size());
    }
  }
#endif

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  if (vkBeginCommandBuffer(context.command_buffers[context.current_frame], &beginInfo) != VK_SUCCESS) {
    fan::throw_error("failed to begin recording command buffer!");
  }
  context.command_buffer_in_use = true;

  for (auto& i : context.begin_cmd_cb) {
    i(context.command_buffers[context.current_frame]);
  }

  
  if (loco.get_render_shapes_top() == false) {
    begin_render_pass();
  }
}

void shapes_draw() {

  loco.context.vk.memory_queue.process(loco.context.vk);

#if defined(FAN_2D)
  fan::graphics::shaper_t::KeyTraverse_t KeyTraverse;
  KeyTraverse.Init(fan::graphics::g_shapes->shaper);

  uint32_t texture_count = 0;
  fan::graphics::viewport_t viewport;
  viewport.sic();
  loco_t::camera_t camera;
  camera.sic();
  fan::graphics::image_t texture;
  texture.sic();

  bool did_draw = false;

  bool light_buffer_enabled = false;

  auto& cmd_buffer = loco.context.vk.command_buffers[loco.context.vk.current_frame];
    
  while (KeyTraverse.Loop(fan::graphics::g_shapes->shaper)) {
    did_draw = true;
    
    fan::graphics::shaper_t::KeyTypeIndex_t kti = KeyTraverse.kti(fan::graphics::g_shapes->shaper);


    switch (kti) {
    case fan::graphics::Key_e::ShapeType: {
      // if i remove this why it breaks/corrupts?
      if (*(fan::graphics::shaper_t::ShapeTypeIndex_t*)KeyTraverse.kd() == fan::graphics::shapes::shape_type_t::light_end) {
        continue;
      }
      break;
      }
      case fan::graphics::Key_e::blending: {
        
        break;
      }
      case fan::graphics::Key_e::image: {
        texture = *(fan::graphics::image_t*)KeyTraverse.kd();
        if (texture.iic() == false) {
          // TODO FIX + 0
          
          //++texture_count;
        }
        break;
      }
      case fan::graphics::Key_e::viewport: {
        viewport = *(fan::graphics::viewport_t*)KeyTraverse.kd();
        break;
      }
      case fan::graphics::Key_e::camera: {
        camera = *(loco_t::camera_t*)KeyTraverse.kd();
        break;
      }
    }

    if (KeyTraverse.isbm) {
      fan::graphics::shaper_t::BlockTraverse_t BlockTraverse;
      fan::graphics::shaper_t::ShapeTypeIndex_t shape_type = BlockTraverse.Init(fan::graphics::g_shapes->shaper, KeyTraverse.bmid());

      if (shape_type == fan::graphics::shapes::shape_type_t::light_end) {
        break;
      }

      do {
        auto shader_nr = fan::graphics::g_shapes->shaper.GetShader(shape_type);
        auto camera_data = loco.camera_get(camera);

        auto& shader = *(fan::vulkan::context_t::shader_t*)loco.context_functions.shader_get(&loco.context.vk, shader_nr);
        shader.projection_view_block->edit_instance(
          loco.context.vk, 
          0, 
          &fan::vulkan::view_projection_t::view,
          camera_data.view
        );
        shader.projection_view_block->edit_instance(
          loco.context.vk,
          0, 
          &fan::vulkan::view_projection_t::projection,
          camera_data.projection
        );

        auto& st = fan::graphics::g_shapes->shaper.GetShapeTypes(shape_type);
        auto& vk_data = st.renderer.vk;
        auto current_frame = loco.context.vk.current_frame;
        vk_data.shape_data.m_descriptor.m_properties[0].buffer =
          vk_data.shape_data.common.memory[current_frame].buffer;
        vk_data.shape_data.m_descriptor.m_properties[1].buffer =
          shader.projection_view_block->common.memory[current_frame].buffer;
        vk_data.shape_data.m_descriptor.m_properties[1].range =
          shader.projection_view_block->m_size;
        vk_data.shape_data.m_descriptor.update(loco.context.vk, 2, 0);
        auto* descriptor_set =
          &vk_data.shape_data.m_descriptor.m_descriptor_set[current_frame];
        {
           vkCmdBindDescriptorSets(
            cmd_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            vk_data.pipeline.m_layout,
            0,
            1,
            descriptor_set,
            0,
            nullptr
          );
        }
        {
          auto viewport_data = loco.viewport_get(viewport);
          VkViewport vk_viewport = {};
          vk_viewport.x = viewport_data.position.x;
          vk_viewport.y = viewport_data.position.y;
          vk_viewport.width = viewport_data.size.x;
          vk_viewport.height = viewport_data.size.y;
          vk_viewport.minDepth = 0.0f;
          vk_viewport.maxDepth = 1.0f;

          vkCmdSetViewport(cmd_buffer, 0, 1, &vk_viewport);

          VkRect2D scissor = {};
          scissor.offset.x = viewport_data.position.x;
          scissor.offset.y = viewport_data.position.y;
          scissor.extent = viewport_data.size;

          vkCmdSetScissor(cmd_buffer, 0, 1, &scissor);
        }

        vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_data.pipeline.m_pipeline);
        vkCmdBindDescriptorSets(
          cmd_buffer,
          VK_PIPELINE_BIND_POINT_GRAPHICS,
          vk_data.pipeline.m_layout,
          0,
          1,
          descriptor_set,
          0,
          nullptr
        );
        fan::vulkan::context_t::push_constants_t vp;
        vp.camera_id = 0; // TODO ??
        vp.texture_id = texture.NRI; // hope that bll doesnt hold sentinels
        // todo use shaper
        vkCmdPushConstants(
          cmd_buffer, 
          vk_data.pipeline.m_layout,
          VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,
          0,
          sizeof(fan::vulkan::context_t::push_constants_t),
          &vp
        );
        auto& shape_data = fan::graphics::g_shapes->shaper.GetShapeTypes(shape_type).renderer.vk;
        auto off = BlockTraverse.GetRenderDataOffset(fan::graphics::g_shapes->shaper) / fan::graphics::g_shapes->shaper.GetRenderDataSize(shape_type);
        //vk_data.shape_data.m_descriptor.update(loco.context.vk, 3, 0, 1, 0);

        vkCmdDraw(
          cmd_buffer, 
          shape_data.vertex_count, 
          BlockTraverse.GetAmount(fan::graphics::g_shapes->shaper), 
          0,
          off
        );
      } while (BlockTraverse.Loop(fan::graphics::g_shapes->shaper));
    }
  }
  if (!did_draw) {
    VkRect2D scissor = {};
    vkCmdSetScissor(cmd_buffer, 0, 1, &scissor);
  }
#endif
}

void init() {
  auto nr = loco.shader_create();
  loco.shader_set_vertex(nr, "shaders/vulkan/loco_fbo.vert", fan::graphics::read_shader("shaders/vulkan/loco_fbo.vert"));
  loco.shader_set_fragment(nr, "shaders/vulkan/loco_fbo.frag", fan::graphics::read_shader("shaders/vulkan/loco_fbo.frag"));
  loco.shader_compile(nr);

  VkPipelineColorBlendAttachmentState color_blend_attachment = fan::vulkan::get_default_color_blend();
  fan::vulkan::context_t::pipeline_t::properties_t p;

  std::vector<fan::vulkan::write_descriptor_set_t> ds_properties(2);

  VkDescriptorImageInfo imageInfo{};
  //gloco()->context.vk.create_texture_sampler(post_process_sampler, fan::vulkan::context_t::image_load_properties_t());

  imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  imageInfo.imageView = gloco()->get_context().vk.mainColorImageViews[0].image_view;
  //imageInfo.sampler = post_process_sampler;

  ds_properties[0].use_image = 1;
  ds_properties[0].binding = 1;
  ds_properties[0].dst_binding = 1;
  ds_properties[0].type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
  ds_properties[0].flags = VK_SHADER_STAGE_FRAGMENT_BIT;
  for (uint32_t i = 0; i < fan::vulkan::max_textures; ++i) {
    ds_properties[0].image_infos[i] = imageInfo;
  }

  imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  imageInfo.imageView = gloco()->get_context().vk.postProcessedColorImageViews[0].image_view;
  //imageInfo.sampler = post_process_sampler;

  imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  ds_properties[1].use_image = 1;
  ds_properties[1].binding = 2;
  ds_properties[1].dst_binding = 2;
  ds_properties[1].type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
  ds_properties[1].flags = VK_SHADER_STAGE_FRAGMENT_BIT;
  for (uint32_t i = 0; i < fan::vulkan::max_textures; ++i) {
    ds_properties[1].image_infos[i] = imageInfo;
  }

  loco.vk.d_attachments.open(loco.context.vk, ds_properties);
  loco.vk.d_attachments.update(loco.context.vk, 2);

  p.shader = nr;
  p.subpass = 1;
  p.descriptor_layout = &loco.vk.d_attachments.m_layout;
  p.descriptor_layout_count = 1;
  p.color_blend_attachment = &color_blend_attachment;
  p.color_blend_attachment_count = 1;
  p.enable_depth_test = false;
  loco.vk.post_process.open(loco.context.vk, p);

  window_resize_handle = loco.window.add_resize_callback([&](const auto& d) {
    loco.camera_set_ortho(
      loco.orthographic_render_view.camera,
      fan::vec2(0, d.size.x),
      fan::vec2(0, d.size.y)
    );

    loco.viewport_set(loco.orthographic_render_view.viewport, fan::vec2(0, 0), d.size);
    loco.viewport_set(loco.perspective_render_view.viewport, fan::vec2(0, 0), d.size);
  });
}

#undef loco
