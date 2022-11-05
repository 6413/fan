#if defined(loco_vulkan)
  std::array<fan::vulkan::write_descriptor_set_t, vulkan_buffer_count> ds_properties{0};

  #if defined(vk_sb_ssbo)
    ds_properties[0].binding = 0;
    ds_properties[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ds_properties[0].flags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    ds_properties[0].range = VK_WHOLE_SIZE;
    ds_properties[0].buffer = m_ssbo.common.memory[loco->get_context()->currentFrame].buffer;
    ds_properties[0].dst_binding = 0;
  #endif

  #if defined(vk_sb_vp)
    ds_properties[1].binding = 1;
    ds_properties[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ds_properties[1].flags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    ds_properties[1].buffer = m_shader.projection_view_block.common.memory[loco->get_context()->currentFrame].buffer;
    ds_properties[1].range = sizeof(fan::mat4) * 2;
    ds_properties[1].dst_binding = 1;
  #endif

  #if defined(vk_sb_image)
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = loco->unloaded_image.get(loco).image_view;
    imageInfo.sampler = loco->unloaded_image.get(loco).sampler;

    ds_properties[2].use_image = 1;
    ds_properties[2].binding = 2;
    ds_properties[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ds_properties[2].flags = VK_SHADER_STAGE_FRAGMENT_BIT;
    for (uint32_t i = 0; i < fan::vulkan::max_textures; ++i) {
      ds_properties[2].image_infos[i] = imageInfo;
    }
    ds_properties[2].dst_binding = 2;

  #endif

  #if defined(loco_wboit) && defined(vk_shape_wboit)
    VkSampler sampler;
    loco_t::image_t::createTextureSampler(loco, sampler);

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = loco->get_context()->vai_wboit_color.image_view;
    imageInfo.sampler = sampler;

    //assert(0);
    // these things for only rectangle, check that ds_properties index is right, and other settings below in pipeline
    ds_properties[2].use_image = 1;
    ds_properties[2].binding = 4;
    ds_properties[2].dst_binding = 4;
    ds_properties[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ds_properties[2].flags = VK_SHADER_STAGE_FRAGMENT_BIT;
    for (uint32_t i = 0; i < fan::vulkan::max_textures; ++i) {
      ds_properties[2].image_infos[i] = imageInfo;
    }

    imageInfo.imageView = loco->get_context()->vai_wboit_reveal.image_view;

    ds_properties[3] = ds_properties[2];
    ds_properties[3].binding = 5;
    ds_properties[3].dst_binding = 5;
    for (uint32_t i = 0; i < fan::vulkan::max_textures; ++i) {
      ds_properties[3].image_infos[i] = imageInfo;
    }
  #endif

  #if defined(loco_wboit) && defined(vk_shape_wboit)
    fan::vulkan::pipeline_t::properties_t p;

    auto context = get_loco()->get_context();

    render_fullscreen_shader.open(context);
    render_fullscreen_shader.set_vertex(context, _FAN_PATH_QUOTE(graphics/glsl/vulkan/2D/objects/fullscreen.vert.spv));
    render_fullscreen_shader.set_fragment(context, _FAN_PATH_QUOTE(graphics/glsl/vulkan/2D/objects/fullscreen.frag.spv));
    p.descriptor_layout_count = 1;
    p.descriptor_layout = &m_descriptor.m_layout;
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
#endif

#undef vk_sb_ssbo
#undef vk_sb_vp
#undef vk_sb_image