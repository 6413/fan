#if defined(loco_opengl)
  sb_open();
#elif defined(loco_vulkan)
  std::array<fan::vulkan::write_descriptor_set_t, vulkan_buffer_count> ds_properties;

  #if defined(vk_sb_ssbo)
    ds_properties[0].binding = 0;
    ds_properties[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ds_properties[0].flags = VK_SHADER_STAGE_VERTEX_BIT;
    ds_properties[0].range = VK_WHOLE_SIZE;
    ds_properties[0].common = &m_ssbo.common;
    ds_properties[0].dst_binding = 0;
  #endif

  #if defined(vk_sb_vp)
    ds_properties[1].binding = 1;
    ds_properties[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ds_properties[1].flags = VK_SHADER_STAGE_VERTEX_BIT;
    ds_properties[1].common = &m_shader.projection_view_block.common;
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

  sb_open(ds_properties);
#endif

#undef vk_sb_ssbo
#undef vk_sb_vp
#undef vk_sb_image