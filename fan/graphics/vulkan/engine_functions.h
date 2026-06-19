VkSampler post_process_sampler = VK_NULL_HANDLE;
VkRenderPass post_process_render_pass = VK_NULL_HANDLE;
VkRenderPass bloom_render_pass = VK_NULL_HANDLE;
VkRenderPass bloom_blend_render_pass = VK_NULL_HANDLE;
std::vector<VkFramebuffer> post_process_framebuffers;
fan::graphics::shader_t post_process_shader;
fan::graphics::shader_t bloom_downsample_shader;
fan::graphics::shader_t bloom_upsample_shader;
fan::vulkan::context_t::pipeline_t bloom_downsample_pipeline;
fan::vulkan::context_t::pipeline_t bloom_upsample_pipeline;
fan::vulkan::context_t::pipeline_t bloom_upsample_add_pipeline;
std::vector<fan::vulkan::context_t::descriptor_t> bloom_downsample_descriptors;
std::vector<fan::vulkan::context_t::descriptor_t> bloom_upsample_descriptors;
bool post_process_resources_open = false;
std::uint32_t bloom_mip_count = 6;
f32_t bloom_filter_radius = 0.1f;
f32_t bloom_threshold = 0.0f;
f32_t bloom_knee = 0.1f;
f32_t bloom_strength = 0.04f;
f32_t bloom_intensity = 1.0f;
f32_t bloom_dirt_intensity = 0.0f;
f32_t bloom_strength_scale = 0.1f;
f32_t gamma = 1.0f;
f32_t exposure = 1.0f;
f32_t contrast = 1.0f;
fan::vec3 bloom_tint = fan::vec3(1.0f, 1.0f, 1.0f);

struct shape_shader_pipeline_t {
  std::uint16_t shape_type;
  fan::graphics::shader_t shader;
  std::uint8_t draw_mode;
  fan::vulkan::context_t::pipeline_t pipeline;
};

std::vector<shape_shader_pipeline_t> shape_shader_pipelines;

struct bloom_mip_t {
  fan::vec2ui size;
  fan::vulkan::vai_t image;
  VkFramebuffer framebuffer = VK_NULL_HANDLE;
};

struct bloom_chain_t {
  std::vector<bloom_mip_t> mips;
};

std::vector<bloom_chain_t> bloom_chains;

struct post_process_push_constants_t {
  fan::vec4 bloom_tint_strength;
  fan::vec4 blur_focus;
  fan::vec4 window_frame;
  fan::vec4 params0;
  fan::vec4 tonemap;
};

struct bloom_downsample_push_constants_t {
  fan::vec4 resolution_threshold_knee_mip;
  fan::vec4 mode;
};

struct bloom_upsample_push_constants_t {
  fan::vec4 filter_radius;
};

loco_t& get_loco() {
  return (*OFFSETLESS(this, loco_t, vk));
}
#define loco get_loco()

void close_shape_shader_pipelines() {
  for (auto& i : shape_shader_pipelines) {
    i.pipeline.close(loco.context.vk);
  }
  shape_shader_pipelines.clear();
}

fan::vulkan::context_t::pipeline_t& get_shape_shader_pipeline(
  std::uint16_t shape_type,
  fan::graphics::shader_t shader,
  std::uint8_t draw_mode
) {
  for (auto& i : shape_shader_pipelines) {
    if (i.shape_type == shape_type && i.shader.gint() == shader.gint() && i.draw_mode == draw_mode) {
      return i.pipeline;
    }
  }

  auto& item = shape_shader_pipelines.emplace_back();
  item.shape_type = shape_type;
  item.shader = shader;
  item.draw_mode = draw_mode;

  auto& vk_data = fan::graphics::g_shapes->shaper.GetShapeTypes(shape_type).renderer.vk;
  VkPipelineColorBlendAttachmentState attachment = fan::vulkan::get_default_color_blend();
  fan::vulkan::context_t::pipeline_t::properties_t pipe_p{};
  pipe_p.color_blend_attachment_count = 1;
  pipe_p.color_blend_attachment = &attachment;
  pipe_p.shader = shader;
  pipe_p.descriptor_layout = &vk_data.shape_data.m_descriptor.m_layout;
  pipe_p.descriptor_layout_count = 1;
  pipe_p.push_constants_size = sizeof(fan::vulkan::context_t::push_constants_t);
  pipe_p.enable_depth_test = false;
  pipe_p.shape_type = (VkPrimitiveTopology)fan::graphics::get_draw_mode(draw_mode);
  item.pipeline.open(loco.context.vk, pipe_p);
  return item.pipeline;
}

VkFormat get_bloom_format() {
  return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
}

VkRenderPass make_color_render_pass(VkFormat format, VkAttachmentLoadOp load_op, VkImageLayout initial_layout, VkImageLayout final_layout) {
  fan::vulkan::context_t& context = loco.context.vk;

  VkAttachmentDescription attachment{};
  attachment.format = format;
  attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  attachment.loadOp = load_op;
  attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachment.initialLayout = initial_layout;
  attachment.finalLayout = final_layout;

  VkAttachmentReference color_ref{};
  color_ref.attachment = 0;
  color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_ref;

  VkSubpassDependency deps[2]{};
  deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
  deps[0].dstSubpass = 0;
  deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

  deps[1].srcSubpass = 0;
  deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
  deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

  VkRenderPassCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  info.attachmentCount = 1;
  info.pAttachments = &attachment;
  info.subpassCount = 1;
  info.pSubpasses = &subpass;
  info.dependencyCount = std::size(deps);
  info.pDependencies = deps;

  VkRenderPass render_pass = VK_NULL_HANDLE;
  if (vkCreateRenderPass(context.device, &info, nullptr, &render_pass) != VK_SUCCESS) {
    fan::throw_error("failed to create post process render pass");
  }
  return render_pass;
}

VkFramebuffer make_framebuffer(VkRenderPass render_pass, VkImageView image_view, fan::vec2ui size) {
  fan::vulkan::context_t& context = loco.context.vk;

  VkFramebufferCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  info.renderPass = render_pass;
  info.attachmentCount = 1;
  info.pAttachments = &image_view;
  info.width = size.x;
  info.height = size.y;
  info.layers = 1;

  VkFramebuffer framebuffer = VK_NULL_HANDLE;
  if (vkCreateFramebuffer(context.device, &info, nullptr, &framebuffer) != VK_SUCCESS) {
    fan::throw_error("failed to create post process framebuffer");
  }
  return framebuffer;
}

fan::vulkan::write_descriptor_set_t make_sampler_descriptor(std::uint32_t binding) {
  fan::vulkan::write_descriptor_set_t d{};
  d.use_image = true;
  d.binding = binding;
  d.dst_binding = binding;
  d.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  d.flags = VK_SHADER_STAGE_FRAGMENT_BIT;
  d.descriptor_count = 1;
  return d;
}

VkDescriptorImageInfo make_image_info(VkImageView image_view) {
  VkDescriptorImageInfo info{};
  info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  info.imageView = image_view;
  info.sampler = post_process_sampler;
  return info;
}

void update_single_sampler_descriptor(fan::vulkan::context_t::descriptor_t& descriptor, VkImageView image_view) {
  descriptor.m_properties[0].image_infos[0] = make_image_info(image_view);
  descriptor.update(loco.context.vk, 1, 0, 1);
}

void begin_color_pass(VkRenderPass render_pass, VkFramebuffer framebuffer, fan::vec2ui size, bool clear) {
  fan::vulkan::context_t& context = loco.context.vk;
  VkRenderPassBeginInfo info{};
  info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  info.renderPass = render_pass;
  info.framebuffer = framebuffer;
  info.renderArea.offset = {0, 0};
  info.renderArea.extent = {size.x, size.y};

  VkClearValue clear_value{};
  clear_value.color = {{0.f, 0.f, 0.f, 0.f}};
  if (clear) {
    info.clearValueCount = 1;
    info.pClearValues = &clear_value;
  }

  VkCommandBuffer cmd = context.command_buffers[context.current_frame];
  vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);

  VkViewport viewport{};
  viewport.x = 0.f;
  viewport.y = 0.f;
  viewport.width = (f32_t)size.x;
  viewport.height = (f32_t)size.y;
  viewport.minDepth = 0.f;
  viewport.maxDepth = 1.f;
  vkCmdSetViewport(cmd, 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = {size.x, size.y};
  vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void draw_fullscreen(
  fan::vulkan::context_t::pipeline_t& pipeline,
  fan::vulkan::context_t::descriptor_t& descriptor,
  const void* push_constants,
  std::uint32_t push_constants_size
) {
  fan::vulkan::context_t& context = loco.context.vk;
  VkCommandBuffer cmd = context.command_buffers[context.current_frame];

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.m_pipeline);
  vkCmdBindDescriptorSets(
    cmd,
    VK_PIPELINE_BIND_POINT_GRAPHICS,
    pipeline.m_layout,
    0,
    1,
    &descriptor.m_descriptor_set[context.current_frame],
    0,
    nullptr
  );
  if (push_constants_size != 0) {
    vkCmdPushConstants(
      cmd,
      pipeline.m_layout,
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      0,
      push_constants_size,
      push_constants
    );
  }
  vkCmdDraw(cmd, 6, 1, 0, 0);
}

void open_post_process_sampler() {
  if (post_process_sampler != VK_NULL_HANDLE) {
    return;
  }
  fan::vulkan::context_t::image_load_properties_t lp{};
  lp.min_filter = VK_FILTER_LINEAR;
  lp.mag_filter = VK_FILTER_LINEAR;
  lp.visual_output = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  loco.context.vk.create_texture_sampler(post_process_sampler, lp);
}

void open_bloom_chains() {
  fan::vulkan::context_t& context = loco.context.vk;
  bloom_chains.resize(context.swap_chain_image_views.size());

  for (std::size_t image_i = 0; image_i < bloom_chains.size(); ++image_i) {
    fan::vec2ui mip_size(
      std::max<std::uint32_t>(1, (std::uint32_t)context.swap_chain_size.x / 2),
      std::max<std::uint32_t>(1, (std::uint32_t)context.swap_chain_size.y / 2)
    );

    auto& chain = bloom_chains[image_i];
    chain.mips.resize(bloom_mip_count);

    for (std::uint32_t mip_i = 0; mip_i < bloom_mip_count; ++mip_i) {
      auto& mip = chain.mips[mip_i];
      mip.size = mip_size;

      fan::vulkan::vai_t::properties_t p{};
      p.swap_chain_size = mip_size;
      p.format = get_bloom_format();
      p.usage_flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      p.aspect_flags = VK_IMAGE_ASPECT_COLOR_BIT;
      mip.image.open(context, p);
      mip.image.transition_image_layout(context, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
      mip.framebuffer = make_framebuffer(bloom_render_pass, mip.image.image_view, mip_size);

      mip_size.x = std::max<std::uint32_t>(1, mip_size.x / 2);
      mip_size.y = std::max<std::uint32_t>(1, mip_size.y / 2);
    }
  }
}

void close_bloom_chains() {
  fan::vulkan::context_t& context = loco.context.vk;
  for (auto& chain : bloom_chains) {
    for (auto& mip : chain.mips) {
      if (mip.framebuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(context.device, mip.framebuffer, nullptr);
        mip.framebuffer = VK_NULL_HANDLE;
      }
      mip.image.close(context);
    }
    chain.mips.clear();
  }
  bloom_chains.clear();
}

void open_post_process_framebuffers() {
  fan::vulkan::context_t& context = loco.context.vk;
  post_process_framebuffers.resize(context.swap_chain_image_views.size());
  for (std::size_t i = 0; i < post_process_framebuffers.size(); ++i) {
    post_process_framebuffers[i] = make_framebuffer(
      post_process_render_pass,
      context.swap_chain_image_views[i],
      fan::vec2ui((std::uint32_t)context.swap_chain_size.x, (std::uint32_t)context.swap_chain_size.y)
    );
  }
}

void close_post_process_framebuffers() {
  fan::vulkan::context_t& context = loco.context.vk;
  for (VkFramebuffer& framebuffer : post_process_framebuffers) {
    if (framebuffer != VK_NULL_HANDLE) {
      vkDestroyFramebuffer(context.device, framebuffer, nullptr);
      framebuffer = VK_NULL_HANDLE;
    }
  }
  post_process_framebuffers.clear();
}

void open_post_process_descriptors() {
  fan::vulkan::context_t& context = loco.context.vk;

  std::vector<fan::vulkan::write_descriptor_set_t> final_descriptors(4);
  final_descriptors[0] = make_sampler_descriptor(0);
  final_descriptors[1] = make_sampler_descriptor(1);
  final_descriptors[2] = make_sampler_descriptor(2);
  final_descriptors[3] = make_sampler_descriptor(3);
  loco.vk.d_attachments.open(context, final_descriptors);

  std::vector<fan::vulkan::write_descriptor_set_t> sampler_descriptors(1);
  sampler_descriptors[0] = make_sampler_descriptor(0);

  bloom_downsample_descriptors.resize(bloom_mip_count);
  bloom_upsample_descriptors.resize(bloom_mip_count);
  for (std::uint32_t i = 0; i < bloom_mip_count; ++i) {
    bloom_downsample_descriptors[i].open(context, sampler_descriptors);
    bloom_upsample_descriptors[i].open(context, sampler_descriptors);
  }
}

void close_post_process_descriptors() {
  fan::vulkan::context_t& context = loco.context.vk;
  for (auto& descriptor : bloom_downsample_descriptors) {
    descriptor.close(context);
  }
  bloom_downsample_descriptors.clear();
  for (auto& descriptor : bloom_upsample_descriptors) {
    descriptor.close(context);
  }
  bloom_upsample_descriptors.clear();
  loco.vk.d_attachments.close(context);
}

void open_post_process_pipelines() {
  fan::vulkan::context_t& context = loco.context.vk;

  VkPipelineColorBlendAttachmentState replace_blend = fan::vulkan::get_default_color_blend();
  replace_blend.blendEnable = VK_FALSE;

  VkPipelineColorBlendAttachmentState add_blend = fan::vulkan::get_default_color_blend();
  add_blend.blendEnable = VK_TRUE;
  add_blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
  add_blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
  add_blend.colorBlendOp = VK_BLEND_OP_ADD;
  add_blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  add_blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  add_blend.alphaBlendOp = VK_BLEND_OP_ADD;

  fan::vulkan::context_t::pipeline_t::properties_t p{};
  p.descriptor_layout_count = 1;
  p.enable_depth_test = false;
  p.color_blend_attachment_count = 1;

  p.shader = post_process_shader;
  p.render_pass = post_process_render_pass;
  p.descriptor_layout = &loco.vk.d_attachments.m_layout;
  p.color_blend_attachment = &replace_blend;
  p.push_constants_size = sizeof(post_process_push_constants_t);
  loco.vk.post_process.open(context, p);

  p.shader = bloom_downsample_shader;
  p.render_pass = bloom_render_pass;
  p.descriptor_layout = &bloom_downsample_descriptors[0].m_layout;
  p.color_blend_attachment = &replace_blend;
  p.push_constants_size = sizeof(bloom_downsample_push_constants_t);
  bloom_downsample_pipeline.open(context, p);

  p.shader = bloom_upsample_shader;
  p.render_pass = bloom_render_pass;
  p.descriptor_layout = &bloom_upsample_descriptors[0].m_layout;
  p.color_blend_attachment = &replace_blend;
  p.push_constants_size = sizeof(bloom_upsample_push_constants_t);
  bloom_upsample_pipeline.open(context, p);

  p.render_pass = bloom_blend_render_pass;
  p.color_blend_attachment = &add_blend;
  bloom_upsample_add_pipeline.open(context, p);
}

void close_post_process_pipelines() {
  fan::vulkan::context_t& context = loco.context.vk;
  loco.vk.post_process.close(context);
  bloom_downsample_pipeline.close(context);
  bloom_upsample_pipeline.close(context);
  bloom_upsample_add_pipeline.close(context);
}

void open_swapchain_resources() {
  if (post_process_resources_open) {
    return;
  }

  fan::vulkan::context_t& context = loco.context.vk;
  open_post_process_sampler();

  post_process_render_pass = make_color_render_pass(
    context.swap_chain_image_format,
    VK_ATTACHMENT_LOAD_OP_CLEAR,
    VK_IMAGE_LAYOUT_UNDEFINED,
    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
  );
  bloom_render_pass = make_color_render_pass(
    get_bloom_format(),
    VK_ATTACHMENT_LOAD_OP_CLEAR,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
  );
  bloom_blend_render_pass = make_color_render_pass(
    get_bloom_format(),
    VK_ATTACHMENT_LOAD_OP_LOAD,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
  );

  open_post_process_framebuffers();
  open_bloom_chains();
  open_post_process_descriptors();
  open_post_process_pipelines();
  post_process_resources_open = true;
}

void close_swapchain_resources() {
  if (!post_process_resources_open) {
    return;
  }

  fan::vulkan::context_t& context = loco.context.vk;
  vkDeviceWaitIdle(context.device);
  close_shape_shader_pipelines();
  close_post_process_pipelines();
  close_post_process_descriptors();
  close_bloom_chains();
  close_post_process_framebuffers();

  if (post_process_render_pass != VK_NULL_HANDLE) {
    vkDestroyRenderPass(context.device, post_process_render_pass, nullptr);
    post_process_render_pass = VK_NULL_HANDLE;
  }
  if (bloom_render_pass != VK_NULL_HANDLE) {
    vkDestroyRenderPass(context.device, bloom_render_pass, nullptr);
    bloom_render_pass = VK_NULL_HANDLE;
  }
  if (bloom_blend_render_pass != VK_NULL_HANDLE) {
    vkDestroyRenderPass(context.device, bloom_blend_render_pass, nullptr);
    bloom_blend_render_pass = VK_NULL_HANDLE;
  }
  post_process_resources_open = false;
}

void close() {
  close_swapchain_resources();
  if (post_process_sampler != VK_NULL_HANDLE) {
    vkDestroySampler(loco.context.vk.device, post_process_sampler, nullptr);
    post_process_sampler = VK_NULL_HANDLE;
  }
}

void draw_bloom() {
  fan::vulkan::context_t& context = loco.context.vk;
  auto& chain = bloom_chains[context.image_index];
  if (chain.mips.empty()) {
    return;
  }

  for (std::uint32_t i = 0; i < chain.mips.size(); ++i) {
    VkImageView source = i == 0 ? context.mainColorImageViews[context.image_index].image_view : chain.mips[i - 1].image.image_view;
    bloom_downsample_push_constants_t pc{};
    fan::vec2 source_size = i == 0 ? context.swap_chain_size : fan::vec2(chain.mips[i - 1].size.x, chain.mips[i - 1].size.y);
    pc.resolution_threshold_knee_mip = fan::vec4(source_size.x, source_size.y, bloom_threshold, bloom_knee);
    pc.mode = fan::vec4(i == 0 ? 0.f : 1.f, 0.f, 0.f, 0.f);

    begin_color_pass(bloom_render_pass, chain.mips[i].framebuffer, chain.mips[i].size, true);
    draw_fullscreen(bloom_downsample_pipeline, bloom_downsample_descriptors[i], &pc, sizeof(pc));
    vkCmdEndRenderPass(context.command_buffers[context.current_frame]);
  }

  for (int i = (int)chain.mips.size() - 1; i > 0; --i) {
    auto& mip = chain.mips[i];
    auto& next_mip = chain.mips[i - 1];

    fan::vec2 texel_size = fan::vec2(1.0f / mip.size.x, 1.0f / mip.size.y) * bloom_filter_radius;
    bloom_upsample_push_constants_t pc{};
    pc.filter_radius = fan::vec4(texel_size.x * 3.f, texel_size.y * 3.f, 0.f, 0.f);

    begin_color_pass(bloom_blend_render_pass, next_mip.framebuffer, next_mip.size, false);
    draw_fullscreen(bloom_upsample_add_pipeline, bloom_upsample_descriptors[i], &pc, sizeof(pc));
    vkCmdEndRenderPass(context.command_buffers[context.current_frame]);
  }
}

void update_final_descriptor() {
  fan::vulkan::context_t& context = loco.context.vk;
  VkImageView scene = context.mainColorImageViews[context.image_index].image_view;
  VkImageView bloom = scene;
  if (context.image_index < bloom_chains.size() && !bloom_chains[context.image_index].mips.empty()) {
    bloom = bloom_chains[context.image_index].mips[0].image.image_view;
  }

  loco.vk.d_attachments.m_properties[0].image_infos[0] = make_image_info(scene);
  loco.vk.d_attachments.m_properties[1].image_infos[0] = make_image_info(bloom);
  loco.vk.d_attachments.m_properties[2].image_infos[0] = make_image_info(scene);
  loco.vk.d_attachments.m_properties[3].image_infos[0] = make_image_info(scene);
  loco.vk.d_attachments.update(context, 4, 0, 1);
}

void update_post_process_descriptors_before_cmd() {
  if (!post_process_resources_open) {
    return;
  }

  fan::vulkan::context_t& context = loco.context.vk;

  if (context.image_index < bloom_chains.size()) {
    auto& chain = bloom_chains[context.image_index];
    for (std::uint32_t i = 0; i < chain.mips.size(); ++i) {
      VkImageView source = i == 0 ? context.mainColorImageViews[context.image_index].image_view : chain.mips[i - 1].image.image_view;
      update_single_sampler_descriptor(bloom_downsample_descriptors[i], source);
    }
    for (int i = (int)chain.mips.size() - 1; i > 0; --i) {
      update_single_sampler_descriptor(bloom_upsample_descriptors[i], chain.mips[i].image.image_view);
    }
  }

  update_final_descriptor();
}

post_process_push_constants_t make_post_process_pc() {
  fan::vec2 window_size = loco.window.get_size();
  if (loco.open_props.blur_focus_follow_mouse) {
    fan::vec2 mouse_position = loco.get_raw_mouse_position();
    auto clamp01 = [](f32_t v) {
      return v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
    };
    f32_t x = window_size.x == 0.f ? 0.f : mouse_position.x / window_size.x;
    f32_t y = window_size.y == 0.f ? 0.f : mouse_position.y / window_size.y;
    loco.open_props.blur_focus_position = fan::vec2(clamp01(x), clamp01(y));
  }

  post_process_push_constants_t pc{};
  pc.bloom_tint_strength = fan::vec4(bloom_tint.x, bloom_tint.y, bloom_tint.z, bloom_strength * bloom_strength_scale);
  pc.blur_focus = fan::vec4(
    loco.open_props.blur_focus_position.x,
    loco.open_props.blur_focus_position.y,
    loco.open_props.blur_focus_radius,
    loco.open_props.blur_focus_falloff
  );
  pc.window_frame = fan::vec4(
    window_size.x,
    window_size.y,
    loco.renderer_state.clear_color.a,
    (f32_t)loco.open_props.post_process_mode
  );
  pc.params0 = fan::vec4(
    bloom_intensity,
    bloom_dirt_intensity,
    loco.open_props.blur_amount,
    loco.open_props.blur_focus_enabled ? 1.f : 0.f
  );
  pc.tonemap = fan::vec4(gamma, exposure, contrast, 0.f);
  return pc;
}

void draw_post_process() {
  fan::vulkan::context_t& context = loco.context.vk;
  VkCommandBuffer cmd = context.command_buffers[context.current_frame];

  vkCmdEndRenderPass(cmd);

  const bool bloom_enabled =
    loco.open_props.post_process_mode == loco_t::post_process_mode_e::bloom ||
    loco.open_props.post_process_mode == loco_t::post_process_mode_e::bloom_blur;

  if (bloom_enabled) {
    draw_bloom();
  }

  auto pc = make_post_process_pc();

  begin_color_pass(
    post_process_render_pass,
    post_process_framebuffers[context.image_index],
    fan::vec2ui((std::uint32_t)context.swap_chain_size.x, (std::uint32_t)context.swap_chain_size.y),
    true
  );
  draw_fullscreen(loco.vk.post_process, loco.vk.d_attachments, &pc, sizeof(pc));
  vkCmdEndRenderPass(cmd);
}

#if defined(FAN_2D)
void shapes_open() {
  struct shape_descriptor_t {
    uint16_t shape_type;
    std::size_t sizeof_vi;
    std::size_t sizeof_ri;
    fan::graphics::shape_gl_init_list_t locations;
    fan::graphics::shader_t shader;
    fan::graphics::shaper_t::ShapeRenderDataSize_t instance_count;
    std::uint32_t vertex_count;
    std::uint8_t draw_mode;
    bool instanced;
  };

#define SHAPE_DESC(shape, shader, vertex_count_, draw_mode_) \
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
    vertex_count_, \
    draw_mode_, \
    true \
  }

  auto& sh = loco.shaders;

  const shape_descriptor_t descriptors[] = {
    SHAPE_DESC(sprite, sh.sprite, 6, fan::graphics::primitive_topology_t::triangles),
    SHAPE_DESC(line, sh.line, 6, fan::graphics::primitive_topology_t::triangles),
    SHAPE_DESC(rectangle, sh.rectangle, 6, fan::graphics::primitive_topology_t::triangles),
    SHAPE_DESC(light, sh.light, 6, fan::graphics::primitive_topology_t::triangles),
    SHAPE_DESC(unlit_sprite, sh.unlit_sprite, 6, fan::graphics::primitive_topology_t::triangles),
    SHAPE_DESC(circle, sh.circle, 6, fan::graphics::primitive_topology_t::triangles),
    SHAPE_DESC(capsule, sh.capsule, 6, fan::graphics::primitive_topology_t::triangles),
    SHAPE_DESC(grid, sh.grid, 6, fan::graphics::primitive_topology_t::triangles),
    SHAPE_DESC(universal_image_renderer, sh.universal_image_renderer, 6, fan::graphics::primitive_topology_t::triangles),
    SHAPE_DESC(gradient, sh.gradient, 6, fan::graphics::primitive_topology_t::triangles),
    SHAPE_DESC(shader_shape, sh.shader_shape, 6, fan::graphics::primitive_topology_t::triangles),
    SHAPE_DESC(shadow, sh.shadow, 6, fan::graphics::primitive_topology_t::triangles),
#if defined(FAN_3D)
    SHAPE_DESC(rectangle3d, sh.rectangle3d, 36, fan::graphics::primitive_topology_t::triangles),
    SHAPE_DESC(line3d, sh.line3d, 2, fan::graphics::primitive_topology_t::lines),
#endif
  };

  for (auto& d : descriptors) {
    loco.shape_open(
      d.shape_type,
      d.sizeof_vi,
      d.sizeof_ri,
      d.locations,
      d.shader,
      d.instance_count,
      d.instanced,
      d.draw_mode
    );
    fan::graphics::g_shapes->shaper.GetShapeTypes(d.shape_type).renderer.vk.vertex_count = d.vertex_count;
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
  C(line);
  C(rectangle);
  C(light);
  C(unlit_sprite);
  C(circle);
  C(capsule);
  C(grid);
  C(universal_image_renderer);
  C(gradient);
  C(shader_shape);
  C(shadow);
#if defined(FAN_3D)
  C(rectangle3d);
  C(line3d);
#endif

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
    2
  ]{};
  fan::color clear_color = loco.get_clear_color();
  clear_values[0].color = { { clear_color.r, clear_color.g, clear_color.b, clear_color.a } };
  clear_values[1].depthStencil = { 1.0f, 0 };


  renderPassInfo.clearValueCount = std::size(clear_values);
  renderPassInfo.pClearValues = clear_values;

  if (loco.get_render_shapes_top()) {
  //  renderPassInfo.clearValueCount = 0;
  }


  vkCmdBeginRenderPass(context.command_buffers[context.current_frame], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
}


#if defined(FAN_2D)
void update_shape_descriptors_before_cmd() {
  if (fan::graphics::g_shapes == nullptr) {
    return;
  }
  if (loco.context.vk.image_pool.empty()) {
    return;
  }

  fan::graphics::shaper_t::KeyTraverse_t KeyTraverse;
  KeyTraverse.Init(fan::graphics::g_shapes->shaper);

  while (KeyTraverse.Loop(fan::graphics::g_shapes->shaper)) {
    if (!KeyTraverse.isbm) {
      continue;
    }

    fan::graphics::shaper_t::BlockTraverse_t BlockTraverse;
    fan::graphics::shaper_t::ShapeTypeIndex_t shape_type = BlockTraverse.Init(fan::graphics::g_shapes->shaper, KeyTraverse.bmid());

    if (shape_type == fan::graphics::shapes::shape_type_t::light_end) {
      continue;
    }

    auto shader_nr = fan::graphics::g_shapes->shaper.GetShader(shape_type);
    auto& shader = *(fan::vulkan::context_t::shader_t*)loco.context_functions.shader_get(&loco.context.vk, shader_nr);
    auto& st = fan::graphics::g_shapes->shaper.GetShapeTypes(shape_type);
    auto& vk_data = st.renderer.vk;
    auto current_frame = loco.context.vk.current_frame;

    vk_data.shape_data.m_descriptor.m_properties[0].buffer =
      vk_data.shape_data.common.memory[current_frame].buffer;
    vk_data.shape_data.m_descriptor.m_properties[1].buffer =
      shader.projection_view_block->common.memory[current_frame].buffer;
    vk_data.shape_data.m_descriptor.m_properties[1].range =
      shader.projection_view_block->m_size;
    vk_data.shape_data.m_descriptor.m_properties[2].image_infos = loco.context.vk.image_pool;
    vk_data.shape_data.m_descriptor.update(loco.context.vk, 3, 0, vk_data.shape_data.m_descriptor.m_properties[2].image_infos.size());
  }
}
#endif

void begin_draw() {
  fan::vulkan::context_t& context = loco.context.vk;
  vkWaitForFences(context.device, 1, &context.in_flight_fences[context.current_frame], VK_TRUE, UINT64_MAX);

  if (context.SwapChainRebuild) {
    close_swapchain_resources();
    context.recreate_swap_chain(&loco.window, VK_SUCCESS);
    open_swapchain_resources();
  }
    
  loco.vk.image_error = vkAcquireNextImageKHR(
    context.device,
    context.swap_chain,
    UINT64_MAX,
    context.image_available_semaphores[context.acquire_semaphore_index],
    VK_NULL_HANDLE,
    &context.image_index
  );

  if (loco.vk.image_error == VK_ERROR_OUT_OF_DATE_KHR || loco.vk.image_error == VK_SUBOPTIMAL_KHR) {
    close_swapchain_resources();
    context.recreate_swap_chain(&loco.window, loco.vk.image_error);
    open_swapchain_resources();
    loco.vk.image_error = vkAcquireNextImageKHR(
      context.device,
      context.swap_chain,
      UINT64_MAX,
      context.image_available_semaphores[context.acquire_semaphore_index],
      VK_NULL_HANDLE,
      &context.image_index
    );
  }

  if (loco.vk.image_error != VK_SUCCESS) { 
    context.command_buffer_in_use = false; 
    return; 
  }

  context.current_acquire_semaphore = context.image_available_semaphores[context.acquire_semaphore_index];
  context.acquire_semaphore_index = (context.acquire_semaphore_index + 1) % (std::uint32_t)context.image_available_semaphores.size();

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
  update_shape_descriptors_before_cmd();
#endif

  update_post_process_descriptors_before_cmd();

  for (auto& i : context.pre_begin_cmd_cb) {
    i();
  }

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
  fan::graphics::shader_t shader_nr;
  shader_nr.sic();

  bool visible = true;
  std::uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
  std::uint32_t vertex_count = (std::uint32_t)-1;
  bool did_draw = false;

  bool light_buffer_enabled = false;

  auto& cmd_buffer = loco.context.vk.command_buffers[loco.context.vk.current_frame];
    
  while (KeyTraverse.Loop(fan::graphics::g_shapes->shaper)) {
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
      case fan::graphics::Key_e::visible: {
        visible = *(std::uint8_t*)KeyTraverse.kd();
        break;
      }
      case fan::graphics::Key_e::shader: {
        shader_nr.gint() = *(fan::graphics::shader_raw_t*)KeyTraverse.kd();
        break;
      }
      case fan::graphics::Key_e::draw_mode: {
        draw_mode = *(std::uint8_t*)KeyTraverse.kd();
        break;
      }
      case fan::graphics::Key_e::vertex_count: {
        vertex_count = *(std::uint32_t*)KeyTraverse.kd();
        break;
      }
    }

    if (KeyTraverse.isbm) {
      fan::graphics::shaper_t::BlockTraverse_t BlockTraverse;
      fan::graphics::shaper_t::ShapeTypeIndex_t shape_type = BlockTraverse.Init(fan::graphics::g_shapes->shaper, KeyTraverse.bmid());

      if (shape_type == fan::graphics::shapes::shape_type_t::light_end) {
        break;
      }
      if (!visible) {
        continue;
      }

      do {
        auto default_shader_nr = fan::graphics::g_shapes->shaper.GetShader(shape_type);
        auto shape_shader_nr = shader_nr.iic() ? default_shader_nr : shader_nr;
        auto camera_data = loco.camera_get(camera);

        auto& shader = *(fan::vulkan::context_t::shader_t*)loco.context_functions.shader_get(&loco.context.vk, shape_shader_nr);
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
        auto& pipeline = shape_shader_nr.gint() == default_shader_nr.gint() ?
          vk_data.pipeline : get_shape_shader_pipeline(shape_type, shape_shader_nr, draw_mode);
        auto current_frame = loco.context.vk.current_frame;
        vk_data.shape_data.m_descriptor.m_properties[0].buffer =
          vk_data.shape_data.common.memory[current_frame].buffer;
        vk_data.shape_data.m_descriptor.m_properties[1].buffer =
          shader.projection_view_block->common.memory[current_frame].buffer;
        vk_data.shape_data.m_descriptor.m_properties[1].range =
          shader.projection_view_block->m_size;
        vk_data.shape_data.m_descriptor.update(
          loco.context.vk,
          3,
          0,
          vk_data.shape_data.m_descriptor.m_properties[2].image_infos.size()
        );
        auto* descriptor_set =
          &vk_data.shape_data.m_descriptor.m_descriptor_set[current_frame];
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

        vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.m_pipeline);
        vkCmdBindDescriptorSets(
          cmd_buffer,
          VK_PIPELINE_BIND_POINT_GRAPHICS,
          pipeline.m_layout,
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
          pipeline.m_layout,
          VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,
          0,
          sizeof(fan::vulkan::context_t::push_constants_t),
          &vp
        );
        auto& shape_data = fan::graphics::g_shapes->shaper.GetShapeTypes(shape_type).renderer.vk;
        auto off = BlockTraverse.GetRenderDataOffset(fan::graphics::g_shapes->shaper) / fan::graphics::g_shapes->shaper.GetRenderDataSize(shape_type);
        auto draw_vertex_count = vertex_count == (std::uint32_t)-1 ? shape_data.vertex_count : vertex_count;
        //vk_data.shape_data.m_descriptor.update(loco.context.vk, 3, 0, 1, 0);

        vkCmdDraw(
          cmd_buffer, 
          draw_vertex_count, 
          BlockTraverse.GetAmount(fan::graphics::g_shapes->shaper), 
          0,
          off
        );
        did_draw = true;
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
  post_process_shader = loco.shader_create();
  loco.shader_set_vertex(post_process_shader, "shaders/vulkan/loco_fbo.vert", fan::graphics::read_shader("shaders/vulkan/loco_fbo.vert"));
  loco.shader_set_fragment(post_process_shader, "shaders/vulkan/loco_fbo.frag", fan::graphics::read_shader("shaders/vulkan/loco_fbo.frag"));
  loco.shader_compile(post_process_shader);

  bloom_downsample_shader = loco.shader_create();
  loco.shader_set_vertex(bloom_downsample_shader, "shaders/vulkan/loco_fbo.vert", fan::graphics::read_shader("shaders/vulkan/loco_fbo.vert"));
  loco.shader_set_fragment(bloom_downsample_shader, "shaders/vulkan/downsample.frag", fan::graphics::read_shader("shaders/vulkan/downsample.frag"));
  loco.shader_compile(bloom_downsample_shader);

  bloom_upsample_shader = loco.shader_create();
  loco.shader_set_vertex(bloom_upsample_shader, "shaders/vulkan/loco_fbo.vert", fan::graphics::read_shader("shaders/vulkan/loco_fbo.vert"));
  loco.shader_set_fragment(bloom_upsample_shader, "shaders/vulkan/upsample.frag", fan::graphics::read_shader("shaders/vulkan/upsample.frag"));
  loco.shader_compile(bloom_upsample_shader);

  open_swapchain_resources();

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