loco_t& get_loco() {
  return (*OFFSETLESS(this, loco_t, vk));
}
#define loco get_loco()

void shapes_open() {
  loco.shape_functions.resize(loco.shape_functions.size() + 1); // button
  loco.shape_functions.resize(loco.shape_functions.size() + 1); // sprite
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

    VkPipelineColorBlendAttachmentState color_blend_attachment[1]{};
    color_blend_attachment[0].colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT |
      VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT |
      VK_COLOR_COMPONENT_A_BIT
      ;
    color_blend_attachment[0].blendEnable = VK_TRUE;
    color_blend_attachment[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blend_attachment[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment[0].colorBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blend_attachment[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment[0].alphaBlendOp = VK_BLEND_OP_ADD;

    fan::vulkan::context_t::pipeline_t::properties_t p;

    std::vector<fan::vulkan::write_descriptor_set_t> ds_properties(2);

    VkDescriptorImageInfo imageInfo{};
    VkSampler sampler;
    gloco->context.vk.create_texture_sampler(sampler, fan::vulkan::context_t::image_load_properties_t());

    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = gloco->get_context().vk.mainColorImageViews[0].image_view;
    imageInfo.sampler = sampler;

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
    imageInfo.sampler = sampler;

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

    p.shader = nr.vk;
    p.subpass = 1;
    p.descriptor_layout = &loco.vk.d_attachments.m_layout;
    p.descriptor_layout_count = 1;
    p.color_blend_attachment = color_blend_attachment;
    p.color_blend_attachment_count = std::size(color_blend_attachment);
    loco.vk.post_process.open(loco.context.vk, p);
  }
}

#undef loco