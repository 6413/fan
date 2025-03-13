VkSampler post_process_sampler;

loco_t& get_loco() {
  return (*OFFSETLESS(this, loco_t, vk));
}
#define loco get_loco()

void shapes_open() {
  loco.shape_functions.resize(loco.shape_functions.size() + 1); // button
  loco.shape_open<loco_t::sprite_t>(
    &loco.sprite,
    "shaders/vulkan/2D/objects/sprite.vert",
    "shaders/vulkan/2D/objects/sprite.frag"
  );
  loco.shape_functions.resize(loco.shape_functions.size() + 1); // text
  loco.shape_functions.resize(loco.shape_functions.size() + 1); // hitbox
  loco.shape_functions.resize(loco.shape_functions.size() + 1); // line
  loco.shape_functions.resize(loco.shape_functions.size() + 1); // mark

  loco.shape_open<loco_t::rectangle_t>(
    &loco.rectangle,
    "shaders/vulkan/2D/objects/rectangle.vert",
    "shaders/vulkan/2D/objects/rectangle.frag"
  );

  {
    auto nr = loco.shader_create();
    loco.shader_set_vertex(nr, loco.read_shader("shaders/vulkan/loco_fbo.vert"));
    loco.shader_set_fragment(nr, loco.read_shader("shaders/vulkan/loco_fbo.frag"));
    loco.shader_compile(nr);

    VkPipelineColorBlendAttachmentState color_blend_attachment = fan::vulkan::get_default_color_blend();
    fan::vulkan::context_t::pipeline_t::properties_t p;

    std::vector<fan::vulkan::write_descriptor_set_t> ds_properties(2);

    VkDescriptorImageInfo imageInfo{};
    gloco->context.vk.create_texture_sampler(post_process_sampler, fan::vulkan::context_t::image_load_properties_t());

    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = gloco->get_context().vk.mainColorImageViews[0].image_view;
    imageInfo.sampler = post_process_sampler;

    ds_properties[0].use_image = 1;
    ds_properties[0].binding = 1;
    ds_properties[0].dst_binding = 1;
    ds_properties[0].type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    ds_properties[0].flags = VK_SHADER_STAGE_FRAGMENT_BIT;
    for (uint32_t i = 0; i < fan::vulkan::max_textures; ++i) {
      ds_properties[0].image_infos[i] = imageInfo;
    }

    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = gloco->get_context().vk.postProcessedColorImageViews[0].image_view;
    imageInfo.sampler = post_process_sampler;

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
    loco.vk.post_process.open(loco.context.vk, p);
  }
}

uint32_t draw_shapes() {
  vkWaitForFences(loco.context.vk.device, 1, &loco.context.vk.in_flight_fences[loco.context.vk.current_frame], VK_TRUE, UINT64_MAX);
    
  VkResult err = vkAcquireNextImageKHR(
    loco.context.vk.device,
    loco.context.vk.swap_chain,
    UINT64_MAX,
    loco.context.vk.image_available_semaphores[loco.context.vk.current_frame],
    VK_NULL_HANDLE,
    &loco.context.vk.image_index
  );

  loco.context.vk.recreate_swap_chain(&loco.window, err);

  loco.context.vk.memory_queue.process(loco.context.vk);

  loco_t::shaper_t::KeyTraverse_t KeyTraverse;
  KeyTraverse.Init(loco.shaper);

  uint32_t texture_count = 0;
  loco_t::viewport_t viewport;
  viewport.sic();
  loco_t::camera_t camera;
  camera.sic();
  loco_t::image_t texture;
  texture.sic();

  bool did_draw = false;

  bool light_buffer_enabled = false;

  auto update_descriptors_after_cmd_buffer_reset = [&] {
    {
      for (auto& st : loco.shaper.ShapeTypes) {
        if (st.sti == (decltype(st.sti))-1) {
          continue;
        }
        // TODO add more shapes here to enable textures
        if (st.sti != loco_t::shape_type_t::sprite) {
          continue;
        }
        auto& vk_data = std::get<loco_t::shaper_t::ShapeType_t::vk_t>(st.renderer);
        // todo slow, use only pointer
        vk_data.shape_data.m_descriptor.m_properties[2].image_infos = loco.context.vk.image_pool;
        // doesnt like multiple frames in flight
        vk_data.shape_data.m_descriptor.update(loco.context.vk, 1, 2, vk_data.shape_data.m_descriptor.m_properties[2].image_infos.size());
      }
    }
  };

  loco.context.vk.begin_render(loco.clear_color, update_descriptors_after_cmd_buffer_reset);

  auto& cmd_buffer = loco.context.vk.command_buffers[loco.context.vk.current_frame];
    
  while (KeyTraverse.Loop(loco.shaper)) {
    did_draw = true;
    
    loco_t::shaper_t::KeyTypeIndex_t kti = KeyTraverse.kti(loco.shaper);


    switch (kti) {
    case Key_e::ShapeType: {
      // if i remove this why it breaks/corrupts?
      if (*(loco_t::shaper_t::ShapeTypeIndex_t*)KeyTraverse.kd() == loco_t::shape_type_t::light_end) {
        continue;
      }
      break;
      }
      case Key_e::blending: {
        
        break;
      }
      case Key_e::image: {
        texture = *(loco_t::image_t*)KeyTraverse.kd();
        if (texture.iic() == false) {
          // TODO FIX + 0
          
          //++texture_count;
        }
        break;
      }
      case Key_e::viewport: {
        viewport = *(loco_t::viewport_t*)KeyTraverse.kd();
        break;
      }
      case Key_e::camera: {
        camera = *(loco_t::camera_t*)KeyTraverse.kd();
        break;
      }
    }

    if (KeyTraverse.isbm) {
      loco_t::shaper_t::BlockTraverse_t BlockTraverse;
      loco_t::shaper_t::ShapeTypeIndex_t shape_type = BlockTraverse.Init(loco.shaper, KeyTraverse.bmid());

      if (shape_type == loco_t::shape_type_t::light_end) {
        break;
      }

      do {
        auto shader_nr = loco.shaper.GetShader(shape_type);
        auto camera_data = loco.camera_get(camera);

        auto& shader = *(fan::vulkan::context_t::shader_t*)loco.context_functions.shader_get(&loco.context.vk, shader_nr);
        shader.projection_view_block.edit_instance(
          loco.context.vk, 
          0, 
          &fan::vulkan::context_t::view_projection_t::view, 
          camera_data.m_view
        );
        shader.projection_view_block.edit_instance(
          loco.context.vk,
          0, 
          &fan::vulkan::context_t::view_projection_t::projection,
          camera_data.m_projection
        );

        auto& st = loco.shaper.GetShapeTypes(shape_type);
        auto& vk_data = std::get<loco_t::shaper_t::ShapeType_t::vk_t>(st.renderer);
        {
           vkCmdBindDescriptorSets(
            cmd_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            vk_data.pipeline.m_layout,
            0,
            1,
            vk_data.shape_data.m_descriptor.m_descriptor_set,
            0,
            nullptr
          );
        }
        {
          auto viewport_data = loco.viewport_get(viewport);
          VkViewport vk_viewport = {};
          vk_viewport.x = viewport_data.viewport_position.x;
          vk_viewport.y = viewport_data.viewport_position.y;
          vk_viewport.width = viewport_data.viewport_size.x;
          vk_viewport.height = viewport_data.viewport_size.y;
          vk_viewport.minDepth = 0.0f;
          vk_viewport.maxDepth = 1.0f;

          vkCmdSetViewport(cmd_buffer, 0, 1, &vk_viewport);

          VkRect2D scissor = {};
          scissor.offset = { 0, 0 };
          scissor.extent = viewport_data.viewport_size;

          vkCmdSetScissor(cmd_buffer, 0, 1, &scissor);
        }

        vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_data.pipeline.m_pipeline);
        vkCmdBindDescriptorSets(
          cmd_buffer,
          VK_PIPELINE_BIND_POINT_GRAPHICS,
          vk_data.pipeline.m_layout,
          0,
          1,
          vk_data.shape_data.m_descriptor.m_descriptor_set,
          0,
          nullptr
        );
        fan::vulkan::context_t::push_constants_t vp;
        vp.camera_id = 0;
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
        auto& shape_data = std::get<loco_t::shaper_t::ShapeType_t::vk_t>(loco.shaper.GetShapeTypes(shape_type).renderer);
        auto off = BlockTraverse.GetRenderDataOffset(loco.shaper) / loco.shaper.GetRenderDataSize(shape_type);
        //vk_data.shape_data.m_descriptor.update(loco.context.vk, 3, 0, 1, 0);

        vkCmdDraw(
          cmd_buffer, 
          shape_data.vertex_count, 
          BlockTraverse.GetAmount(loco.shaper), 
          0,
          off
        );
      } while (BlockTraverse.Loop(loco.shaper));
    }
  }
  if (did_draw == false) {
    return -0xfff;
  }
  return err;
}

#undef loco