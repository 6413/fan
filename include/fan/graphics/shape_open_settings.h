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
    ds_properties[1].range = m_shader.projection_view_block.m_size;
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

    VkSampler sampler;
    loco_t::image_t::createTextureSampler(loco, sampler);

    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = loco->get_context()->vai_bitmap[0].image_view;
    imageInfo.sampler = sampler;

    //assert(0);
    // these things for only rectangle, check that ds_properties index is right, and other settings below in pipeline
    ds_properties[3].use_image = 1;
    ds_properties[3].binding = 3;
    ds_properties[3].dst_binding = 3;
    ds_properties[3].type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    ds_properties[3].flags = VK_SHADER_STAGE_FRAGMENT_BIT ;
    for (uint32_t i = 0; i < fan::vulkan::max_textures; ++i) {
      ds_properties[3].image_infos[i] = imageInfo;
    }

  //VkSampler sampler;
  //loco_t::image_t::createTextureSampler(loco, sampler);

  //imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  //imageInfo.imageView = loco->get_context()->vai_bitmap.image_view;
  //imageInfo.sampler = sampler;

  //
  //ds_properties[3].use_image = 1;
  //ds_properties[3].binding = 4;
  //ds_properties[3].dst_binding = 4;
  //ds_properties[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  //ds_properties[3].flags = VK_SHADER_STAGE_FRAGMENT_BIT;
  //for (uint32_t i = 0; i < fan::vulkan::max_textures; ++i) {
  //  ds_properties[3].image_infos[i] = imageInfo;
  //}

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
#endif

#undef vk_sb_ssbo
#undef vk_sb_vp
#undef vk_sb_image