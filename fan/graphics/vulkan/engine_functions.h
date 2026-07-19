VkSampler post_process_sampler = VK_NULL_HANDLE;
VkRenderPass post_process_render_pass = VK_NULL_HANDLE;
VkRenderPass bloom_render_pass = VK_NULL_HANDLE;
VkRenderPass bloom_blend_render_pass = VK_NULL_HANDLE;
std::vector<VkFramebuffer> post_process_framebuffers;
fan::graphics::shader_t post_process_shader;
fan::graphics::shader_t bloom_downsample_shader;
fan::graphics::shader_t bloom_upsample_shader;
fan::vulkan::context_t::compute_pipeline_t bloom_downsample_pipeline;
fan::vulkan::context_t::compute_pipeline_t bloom_upsample_pipeline;
std::vector<std::vector<fan::vulkan::context_t::descriptor_t>> bloom_downsample_descriptors;
std::vector<std::vector<fan::vulkan::context_t::descriptor_t>> bloom_upsample_descriptors;
bool post_process_resources_open = false;
std::uint32_t bloom_mip_count = 6;
f32_t bloom_filter_radius = 0.1f;
f32_t bloom_threshold = 0.0f;
f32_t bloom_knee = 0.1f;
f32_t bloom_strength = 0.0445f;
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
  std::uint32_t generation = 0;
  fan::vulkan::context_t::pipeline_t pipeline;
};

std::vector<shape_shader_pipeline_t> shape_shader_pipelines;

struct deferred_shape_pipeline_destroy_t {
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  std::uint32_t frames_left = 0;
};

std::vector<deferred_shape_pipeline_destroy_t> deferred_shape_pipeline_destroys;

struct bloom_mip_t {
  fan::vec2ui size;
  fan::vulkan::vai_t image;
  VkFramebuffer framebuffer = VK_NULL_HANDLE;
};

struct bloom_chain_t {
  std::vector<bloom_mip_t> mips;
};

std::vector<bloom_chain_t> bloom_chains;

struct particles_gpu_t {
  fan::vec4 position_shape;
  fan::vec4 loop_times;
  fan::vec4 count_life;
  fan::vec4 size;
  fan::vec4 color0;
  fan::vec4 color1;
  fan::vec4 velocity;
  fan::vec4 angle_velocity0;
  fan::vec4 angle_velocity1;
  fan::vec4 angle;
  fan::vec4 spawn_spread0;
  fan::vec4 spread1_jitter;
  fan::vec4 jitter_random_size;
  fan::vec4 color_random;
  fan::vec4 angle_random;
};

std::array<fan::vulkan::context_t::buffer_t, fan::vulkan::max_frames_in_flight> polygon_draw_buffers;
std::array<fan::vulkan::context_t::buffer_t, fan::vulkan::max_frames_in_flight> particle_draw_buffers;
std::array<VkDeviceSize, fan::vulkan::max_frames_in_flight> polygon_draw_capacity{};
std::array<VkDeviceSize, fan::vulkan::max_frames_in_flight> particle_draw_capacity{};

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
  return *loco_ptr;
}
#define loco get_loco()

void destroy_shape_pipeline_handles(VkPipeline pipeline, VkPipelineLayout layout) {
  auto& context = loco.context.vk;
  if (pipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(context.device, pipeline, nullptr);
  }
  if (layout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(context.device, layout, nullptr);
  }
}

void defer_close_shape_pipeline(fan::vulkan::context_t::pipeline_t& pipeline) {
  if (pipeline.m_pipeline == VK_NULL_HANDLE && pipeline.m_layout == VK_NULL_HANDLE) {
    return;
  }

  deferred_shape_pipeline_destroys.push_back({
    .pipeline = pipeline.m_pipeline,
    .layout = pipeline.m_layout,
    .frames_left = fan::vulkan::max_frames_in_flight
  });

  pipeline.m_pipeline = VK_NULL_HANDLE;
  pipeline.m_layout = VK_NULL_HANDLE;
}

void flush_deferred_shape_pipeline_destroys(bool force = false) {
  for (auto it = deferred_shape_pipeline_destroys.begin(); it != deferred_shape_pipeline_destroys.end();) {
    if (!force && it->frames_left != 0) {
      --it->frames_left;
      ++it;
      continue;
    }

    destroy_shape_pipeline_handles(it->pipeline, it->layout);
    it = deferred_shape_pipeline_destroys.erase(it);
  }
}

void close_shape_shader_pipelines() {
  flush_deferred_shape_pipeline_destroys(true);
  for (auto& i : shape_shader_pipelines) {
    destroy_shape_pipeline_handles(i.pipeline.m_pipeline, i.pipeline.m_layout);
    i.pipeline.m_pipeline = VK_NULL_HANDLE;
    i.pipeline.m_layout = VK_NULL_HANDLE;
  }
  shape_shader_pipelines.clear();
}

void close_shape_draw_buffers() {
  auto& context = loco.context.vk;
  for (std::uint32_t i = 0; i < fan::vulkan::max_frames_in_flight; ++i) {
    context.destroy_buffer(polygon_draw_buffers[i]);
    context.destroy_buffer(particle_draw_buffers[i]);
    polygon_draw_capacity[i] = 0;
    particle_draw_capacity[i] = 0;
  }
}

fan::vulkan::context_t::pipeline_t& get_shape_shader_pipeline(
  std::uint16_t shape_type,
  fan::graphics::shader_t shader,
  std::uint8_t draw_mode
) {
  auto generation = loco.context.vk.shaders.shader_get(shader).compile_generation;
  for (auto it = shape_shader_pipelines.begin(); it != shape_shader_pipelines.end();) {
    if (it->shape_type != shape_type || it->shader.gint() != shader.gint() || it->draw_mode != draw_mode) {
      ++it;
      continue;
    }
    if (it->generation == generation) {
      return it->pipeline;
    }
    defer_close_shape_pipeline(it->pipeline);
    it = shape_shader_pipelines.erase(it);
  }

  auto& item = shape_shader_pipelines.emplace_back();
  item.shape_type = shape_type;
  item.shader = shader;
  item.draw_mode = draw_mode;
  item.generation = generation;

  auto& vk_data = fan::graphics::g_shapes->shaper.GetShapeTypes(shape_type).renderer.vk;
  fan::vulkan::context_t::pipeline_t::properties_t pipe_p{};
  pipe_p.color_blend_attachments = {fan::vulkan::get_default_color_blend()};
  pipe_p.shader = shader;
  pipe_p.descriptor_layouts = {vk_data.shape_data.m_descriptor.m_layout};
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

fan::vulkan::write_descriptor_set_t make_compute_sampler_descriptor(std::uint32_t binding) {
  fan::vulkan::write_descriptor_set_t d{};
  d.use_image = true;
  d.binding = binding;
  d.dst_binding = binding;
  d.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  d.flags = VK_SHADER_STAGE_COMPUTE_BIT;
  d.descriptor_count = 1;
  return d;
}

fan::vulkan::write_descriptor_set_t make_storage_image_descriptor(std::uint32_t binding) {
  fan::vulkan::write_descriptor_set_t d{};
  d.use_image = true;
  d.binding = binding;
  d.dst_binding = binding;
  d.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  d.flags = VK_SHADER_STAGE_COMPUTE_BIT;
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
  bloom_downsample_descriptors.resize(context.swap_chain_image_views.size());
  bloom_upsample_descriptors.resize(context.swap_chain_image_views.size());

  for (std::size_t image_i = 0; image_i < bloom_chains.size(); ++image_i) {
    fan::vec2ui mip_size(
      std::max<std::uint32_t>(1, (std::uint32_t)context.swap_chain_size.x / 2),
      std::max<std::uint32_t>(1, (std::uint32_t)context.swap_chain_size.y / 2)
    );

    auto& chain = bloom_chains[image_i];
    chain.mips.resize(bloom_mip_count);

    bloom_downsample_descriptors[image_i].resize(bloom_mip_count);
    bloom_upsample_descriptors[image_i].resize(bloom_mip_count);

    for (std::uint32_t mip_i = 0; mip_i < bloom_mip_count; ++mip_i) {
      auto& mip = chain.mips[mip_i];
      mip.size = mip_size;

      fan::vulkan::vai_t::properties_t p{};
      p.swap_chain_size = mip_size;
      p.format = get_bloom_format();
      p.usage_flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
      p.aspect_flags = VK_IMAGE_ASPECT_COLOR_BIT;
      mip.image.open(context, p);
      // ALL mips must be GENERAL layout for compute storage writes.
      mip.image.transition_image_layout(context, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);

      // Downsample descriptors
      std::vector<fan::vulkan::write_descriptor_set_t> down_writes(2);
      down_writes[0] = make_compute_sampler_descriptor(0);
      down_writes[1] = make_storage_image_descriptor(1);

      VkImageView down_source = mip_i == 0 ? context.mainColorImageViews[image_i].image_view : chain.mips[mip_i - 1].image.image_view;
      down_writes[0].image_infos[0] = make_image_info(down_source);
      down_writes[0].image_infos[0].imageLayout = mip_i == 0 ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
      down_writes[1].image_infos[0] = make_image_info(mip.image.image_view);
      down_writes[1].image_infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

      bloom_downsample_descriptors[image_i][mip_i].open(context, down_writes);
      for (std::uint32_t f = 0; f < fan::vulkan::max_frames_in_flight; ++f) {
        context.current_frame = f;
        bloom_downsample_descriptors[image_i][mip_i].update(context, 2, 0, 1);
      }
      context.current_frame = 0;

      if (mip_i > 0) {
        std::vector<fan::vulkan::write_descriptor_set_t> up_writes(3);
        up_writes[0] = make_compute_sampler_descriptor(0);
        up_writes[1] = make_compute_sampler_descriptor(1);
        up_writes[2] = make_storage_image_descriptor(2);

        up_writes[0].image_infos[0] = make_image_info(mip.image.image_view);
        up_writes[0].image_infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        up_writes[1].image_infos[0] = make_image_info(chain.mips[mip_i - 1].image.image_view);
        up_writes[1].image_infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        up_writes[2].image_infos[0] = make_image_info(chain.mips[mip_i - 1].image.image_view);
        up_writes[2].image_infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        
        bloom_upsample_descriptors[image_i][mip_i].open(context, up_writes);
        for (std::uint32_t f = 0; f < fan::vulkan::max_frames_in_flight; ++f) {
          context.current_frame = f;
          bloom_upsample_descriptors[image_i][mip_i].update(context, 3, 0, 1);
        }
        context.current_frame = 0;
      }

      mip_size.x = std::max<std::uint32_t>(1, mip_size.x / 2);
      mip_size.y = std::max<std::uint32_t>(1, mip_size.y / 2);
    }
  }
}

void close_bloom_chains() {
  fan::vulkan::context_t& context = loco.context.vk;
  for (std::size_t image_i = 0; image_i < bloom_chains.size(); ++image_i) {
    auto& chain = bloom_chains[image_i];
    for (std::uint32_t mip_i = 0; mip_i < chain.mips.size(); ++mip_i) {
      bloom_downsample_descriptors[image_i][mip_i].close(context);
      bloom_upsample_descriptors[image_i][mip_i].close(context);
      chain.mips[mip_i].image.close(context);
    }
    chain.mips.clear();
  }
  bloom_chains.clear();
  bloom_downsample_descriptors.clear();
  bloom_upsample_descriptors.clear();
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
  loco.vk->d_attachments.open(context, final_descriptors);
}

void close_post_process_descriptors() {
  fan::vulkan::context_t& context = loco.context.vk;
  loco.vk->d_attachments.close(context);
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
  p.enable_depth_test = false;
  p.color_blend_attachments = {replace_blend};

  p.shader = post_process_shader;
  p.render_pass = post_process_render_pass;
  p.descriptor_layouts = {loco.vk->d_attachments.m_layout};
  p.push_constants_size = sizeof(post_process_push_constants_t);
  loco.vk->post_process.open(context, p);

  fan::vulkan::compute_pipeline_t::properties_t cp{};
  cp.shader = bloom_downsample_shader;
  cp.descriptor_layouts = {bloom_downsample_descriptors[0][0].m_layout};
  cp.push_constants_size = sizeof(bloom_downsample_push_constants_t);
  bloom_downsample_pipeline.open(context, cp);

  cp.shader = bloom_upsample_shader;
  cp.descriptor_layouts = {bloom_upsample_descriptors[0][1].m_layout};
  cp.push_constants_size = sizeof(bloom_upsample_push_constants_t);
  bloom_upsample_pipeline.open(context, cp);
}

void close_post_process_pipelines() {
  fan::vulkan::context_t& context = loco.context.vk;
  loco.vk->post_process.close(context);
  bloom_downsample_pipeline.close(context);
  bloom_upsample_pipeline.close(context);
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

  post_process_resources_open = false;
}

void close() {
  close_swapchain_resources();
  close_shape_draw_buffers();
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

  VkCommandBuffer cmd = context.command_buffers[context.current_frame];

  // Downsample pass
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bloom_downsample_pipeline.pipeline);

  {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = chain.mips[0].image.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 0, nullptr, 0, nullptr, 1, &barrier);
  }

  for (std::uint32_t i = 0; i < chain.mips.size(); ++i) {
    bloom_downsample_push_constants_t pc{};
    fan::vec2 source_size = i == 0 ? context.swap_chain_size : fan::vec2(chain.mips[i - 1].size.x, chain.mips[i - 1].size.y);
    pc.resolution_threshold_knee_mip = fan::vec4(source_size.x, source_size.y, bloom_threshold, bloom_knee);
    pc.mode = fan::vec4(i == 0 ? 0.f : 1.f, 0.f, 0.f, 0.f);

    vkCmdPushConstants(cmd, bloom_downsample_pipeline.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bloom_downsample_pipeline.pipeline_layout, 0, 1, &bloom_downsample_descriptors[context.image_index][i].m_descriptor_set[context.current_frame], 0, nullptr);

    vkCmdDispatch(cmd, (chain.mips[i].size.x + 7) / 8, (chain.mips[i].size.y + 7) / 8, 1);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = chain.mips[i].image.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 0, nullptr, 0, nullptr, 1, &barrier);
  }

  // Upsample pass
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bloom_upsample_pipeline.pipeline);

  for (int i = (int)chain.mips.size() - 1; i > 0; --i) {
    auto& mip = chain.mips[i];
    auto& next_mip = chain.mips[i - 1];

    fan::vec2 texel_size = fan::vec2(1.0f / mip.size.x, 1.0f / mip.size.y);
bloom_upsample_push_constants_t pc{};
pc.filter_radius = fan::vec4(texel_size.x * (1.0f + bloom_filter_radius), texel_size.y * (1.0f + bloom_filter_radius), 0.f, 0.f);


    vkCmdPushConstants(cmd, bloom_upsample_pipeline.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bloom_upsample_pipeline.pipeline_layout, 0, 1, &bloom_upsample_descriptors[context.image_index][i].m_descriptor_set[context.current_frame], 0, nullptr);

    vkCmdDispatch(cmd, (next_mip.size.x + 7) / 8, (next_mip.size.y + 7) / 8, 1);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = next_mip.image.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 0, nullptr, 0, nullptr, 1, &barrier);
  }

  // Final transition for mip[0] to SHADER_READ_ONLY_OPTIMAL for the tonemap pass
  if (!chain.mips.empty()) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = chain.mips[0].image.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
      0, 0, nullptr, 0, nullptr, 1, &barrier);
  }
}

void update_final_descriptor() {
  fan::vulkan::context_t& context = loco.context.vk;
  VkImageView scene = context.mainColorImageViews[context.image_index].image_view;
  VkImageView bloom = scene;
  if (context.image_index < bloom_chains.size() && !bloom_chains[context.image_index].mips.empty()) {
    bloom = bloom_chains[context.image_index].mips[0].image.image_view;
  }

  loco.vk->d_attachments.m_properties[0].image_infos[0] = make_image_info(scene);
  loco.vk->d_attachments.m_properties[0].image_infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  loco.vk->d_attachments.m_properties[1].image_infos[0] = make_image_info(bloom);
  loco.vk->d_attachments.m_properties[1].image_infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  loco.vk->d_attachments.m_properties[2].image_infos[0] = make_image_info(scene);
  loco.vk->d_attachments.m_properties[2].image_infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  loco.vk->d_attachments.m_properties[3].image_infos[0] = make_image_info(scene);
  loco.vk->d_attachments.m_properties[3].image_infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  
  // Only update the CURRENT frame's descriptor to avoid in-flight data races
  loco.vk->d_attachments.update(context, 4, 0, 1);
}

void update_post_process_descriptors_before_cmd() {
  if (!post_process_resources_open) {
    return;
  }

  fan::vulkan::context_t& context = loco.context.vk;

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
    1.f,
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
    loco.open_props.post_process_mode == fan::graphics::post_process_mode_e::bloom ||
    loco.open_props.post_process_mode == fan::graphics::post_process_mode_e::bloom_blur;

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
  draw_fullscreen(loco.vk->post_process, loco.vk->d_attachments, &pc, sizeof(pc));
  vkCmdEndRenderPass(cmd);
}

#if defined(FAN_2D)
void shapes_open() {
  struct shape_descriptor_t {
    uint16_t shape_type;
    std::size_t sizeof_vi;
    std::size_t sizeof_ri;
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
    SHAPE_DESC(polygon, sh.polygon, 0, fan::graphics::primitive_topology_t::triangles),
    SHAPE_DESC(grid, sh.grid, 6, fan::graphics::primitive_topology_t::triangles),
    SHAPE_DESC(particles, sh.particles, 0, fan::graphics::primitive_topology_t::triangles),
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
struct shader_preload_payload_t {
  const char* vs_path;
  const char* fs_path;
  std::string vs_code;
  std::string fs_code;
  std::vector<std::uint32_t> vs_spv;
  std::vector<std::uint32_t> fs_spv;
  fan::graphics::shader_t* out;
};
std::deque<shader_preload_payload_t> shader_preloads;

void shaders_compile_preload() {
  auto& sh = loco.shaders;

  auto preload = [&](fan::graphics::shader_t& out, const char* vs, const char* fs) {
    shader_preloads.push_back({
      vs, fs, "", "", {}, {}, &out
    });
    auto& payload = shader_preloads.back();

    loco.shader_preload_threads.emplace_back([&payload, p_loco = &loco]() {
      payload.vs_code = fan::graphics::read_shader(payload.vs_path);
      payload.fs_code = fan::graphics::read_shader(payload.fs_path);
      payload.vs_spv = p_loco->context.vk.shaders.load_or_compile(payload.vs_path, 0, payload.vs_code);
      payload.fs_spv = p_loco->context.vk.shaders.load_or_compile(payload.fs_path, 1, payload.fs_code);
    });
  };

#define C(n) preload(sh.n, "shaders/vulkan/2D/objects/" #n ".vert", "shaders/vulkan/2D/objects/" #n ".frag")

  C(sprite);
  C(line);
  C(rectangle);
  C(light);
  C(unlit_sprite);
  C(circle);
  C(capsule);
  C(polygon);
  C(grid);
  C(particles);
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

void shaders_compile() {
  for (auto& t : loco.shader_preload_threads) {
    if (t.joinable()) t.join();
  }
  loco.shader_preload_threads.clear();

  for (auto& payload : shader_preloads) {
    *payload.out = loco.shader_create();
    auto& list_item = fan::graphics::ctx().shader_list->operator[](*payload.out);
    list_item.path_vertex = std::string_view(payload.vs_path);
    list_item.path_fragment = std::string_view(payload.fs_path);
    list_item.svertex = std::move(payload.vs_code);
    list_item.sfragment = std::move(payload.fs_code);
    list_item.spv_vertex = std::move(payload.vs_spv);
    list_item.spv_fragment = std::move(payload.fs_spv);
    loco.shader_compile(*payload.out);
  }
  shader_preloads.clear();
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


void begin_draw() {
  fan::vulkan::context_t& context = loco.context.vk;
  vkWaitForFences(context.device, 1, &context.in_flight_fences[context.current_frame], VK_TRUE, UINT64_MAX);

  if (context.timestamp_query_pool) {
    vkGetQueryPoolResults(
      context.device,
      context.timestamp_query_pool,
      context.current_frame * 2,
      2,
      sizeof(context.gpu_timestamps),
      context.gpu_timestamps,
      sizeof(std::uint64_t),
      VK_QUERY_RESULT_64_BIT
    );
    if (context.gpu_timestamps[1] != 0 && context.gpu_timestamps[0] != 0 && context.gpu_timestamps[1] > context.gpu_timestamps[0]) {
      double gpu_time_ms = double(context.gpu_timestamps[1] - context.gpu_timestamps[0]) * context.timestamp_period / 1000000.0;
      fan::time::global_profiler.add_gpu_time("GPU Render", gpu_time_ms);
    }
  }

  context.get_current_deletion_queue(context.current_frame).flush();
  context.get_current_deletion_queue(context.current_frame).merge(context.pending_deletion_queue);
  flush_deferred_shape_pipeline_destroys();

  if (context.SwapChainRebuild) {
    close_swapchain_resources();
    context.recreate_swap_chain(&loco.window, VK_SUCCESS);
    open_swapchain_resources();
  }
    
  loco.vk->image_error = vkAcquireNextImageKHR(
    context.device,
    context.swap_chain,
    UINT64_MAX,
    context.image_available_semaphores[context.acquire_semaphore_index],
    VK_NULL_HANDLE,
    &context.image_index
  );

  if (loco.vk->image_error == VK_ERROR_OUT_OF_DATE_KHR || loco.vk->image_error == VK_SUBOPTIMAL_KHR) {
    close_swapchain_resources();
    context.recreate_swap_chain(&loco.window, loco.vk->image_error);
    open_swapchain_resources();
    loco.vk->image_error = vkAcquireNextImageKHR(
      context.device,
      context.swap_chain,
      UINT64_MAX,
      context.image_available_semaphores[context.acquire_semaphore_index],
      VK_NULL_HANDLE,
      &context.image_index
    );
  }

  if (loco.vk->image_error != VK_SUCCESS) { 
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

  if (context.timestamp_query_pool) {
    vkCmdResetQueryPool(context.command_buffers[context.current_frame], context.timestamp_query_pool, context.current_frame * 2, 2);
    vkCmdWriteTimestamp(context.command_buffers[context.current_frame], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, context.timestamp_query_pool, context.current_frame * 2);
  }

  if (context.cameras.viewport_dirty) {
    vkCmdSetViewport(context.command_buffers[context.current_frame], 0, 1, &context.cameras.pending_viewport);
    vkCmdSetScissor(context.command_buffers[context.current_frame], 0, 1, &context.cameras.pending_scissor);
    context.cameras.viewport_dirty = false;
  }

  for (auto& i : context.begin_cmd_cb) {
    i(context.command_buffers[context.current_frame]);
  }

  
  if (loco.get_render_shapes_top() == false) {
    begin_render_pass();
  }
}

void shapes_draw() {
  fan::time::global_profiler.begin("Memory Queue Process");
  loco.context.vk.memory_queue.process(loco.context.vk);
  fan::time::global_profiler.end("Memory Queue Process");

#if defined(FAN_2D)
  fan::time::global_profiler.begin("Variable Initialization");
  fan::graphics::shaper_t::KeyTraverse_t KeyTraverse;
  KeyTraverse.Init(fan::graphics::g_shapes->shaper);

  fan::graphics::viewport_t viewport;
  viewport.sic();
  loco_t::camera_t camera;
  camera.sic();
  decltype(camera.NRI) camera_id = 0;
  fan::graphics::image_t texture;
  texture.sic();
  fan::graphics::shader_t shader_nr;
  shader_nr.sic();

  bool visible = true;
  std::uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
  std::uint32_t vertex_count = (std::uint32_t)-1;
  bool did_draw = false;

  auto& context = loco.context.vk;
  auto& cmd_buffer = context.command_buffers[context.current_frame];
  auto& shaper = fan::graphics::g_shapes->shaper;

  auto texture_id = [&] (fan::graphics::image_t image) -> std::uint32_t {
    return image.iic() ? loco.default_texture.NRI : image.NRI;
  };

  auto set_viewport = [&] {
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
  };

  auto prepare = [&] (std::uint16_t shape_type, fan::graphics::shader_t shape_shader_nr) -> fan::vulkan::context_t::pipeline_t& {
    auto camera_data = loco.camera_get(camera);
    auto& shader = *(fan::vulkan::shader_t*)loco.context_functions.shader_get(&context, shape_shader_nr);
    shader.projection_view_block->edit_instance(
      context,
      camera_id,
      &fan::vulkan::view_projection_t::view,
      camera_data.view
    );
    shader.projection_view_block->edit_instance(
      context,
      camera_id,
      &fan::vulkan::view_projection_t::projection,
      camera_data.projection
    );
    shader.projection_view_block->write(context);

    auto& vk_data = shaper.GetShapeTypes(shape_type).renderer.vk;
    auto default_shader_nr = fan::graphics::g_shapes->shaper.GetShader(shape_type);
    bool use_shader_pipeline = shape_type == fan::graphics::shapes::shape_type_t::shader_shape ||
      shape_type == fan::graphics::shapes::shape_type_t::universal_image_renderer ||
      shape_shader_nr.gint() != default_shader_nr.gint();
    auto& pipeline = use_shader_pipeline ? get_shape_shader_pipeline(shape_type, shape_shader_nr, draw_mode) : vk_data.pipeline;

    if (shape_type == fan::graphics::shapes::shape_type_t::particles) {
      vk_data.shape_data.m_descriptor.m_properties[0].buffer = particle_draw_buffers[context.current_frame].buffer;
    } else if (shape_type == fan::graphics::shapes::shape_type_t::polygon) {
      vk_data.shape_data.m_descriptor.m_properties[0].buffer = polygon_draw_buffers[context.current_frame].buffer;
    } else {
      vk_data.shape_data.m_descriptor.m_properties[0].buffer =
        vk_data.shape_data.common.memory[context.current_frame].buffer;
    }
    vk_data.shape_data.m_descriptor.m_properties[0].range = VK_WHOLE_SIZE;
    vk_data.shape_data.m_descriptor.m_properties[1].buffer =
      shader.projection_view_block->common.memory[context.current_frame].buffer;
    vk_data.shape_data.m_descriptor.m_properties[1].range =
      shader.projection_view_block->m_size;
    vk_data.shape_data.m_descriptor.m_properties[2].image_infos = context.image_pool;
    vk_data.shape_data.m_descriptor.update(
      context,
      3,
      0,
      std::min((std::uint32_t)vk_data.shape_data.m_descriptor.m_properties[2].image_infos.size(), (std::uint32_t)fan::vulkan::max_textures)
    );

    set_viewport();
    vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.m_pipeline);
    vkCmdBindDescriptorSets(
      cmd_buffer,
      VK_PIPELINE_BIND_POINT_GRAPHICS,
      pipeline.m_layout,
      0,
      1,
      &vk_data.shape_data.m_descriptor.m_descriptor_set[context.current_frame],
      0,
      nullptr
    );
    return pipeline;
  };

  auto push = [&] (fan::vulkan::context_t::pipeline_t& pipeline, fan::vulkan::context_t::push_constants_t pc) {
    auto ambient = loco.renderer_state.lighting.ambient;
    pc.lighting_ambient = fan::vec4(ambient.x, ambient.y, ambient.z, 1.f);
    vkCmdPushConstants(
      cmd_buffer,
      pipeline.m_layout,
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      0,
      sizeof(fan::vulkan::context_t::push_constants_t),
      &pc
    );
  };

  auto make_particles_gpu = [&](const fan::graphics::shapes::particles_t::ri_t& ri) {
    particles_gpu_t gpu{};
    gpu.position_shape = fan::vec4(ri.position, (f32_t)ri.shape);
    gpu.loop_times = fan::vec4(
      (f32_t)ri.loop, 
      ri.loop_enabled_time, 
      ri.loop_disabled_time, 
      f32_t(f64_t(fan::time::now() - ri.begin_time) / 1e9)
    );
    gpu.count_life = fan::vec4((f32_t)ri.count, ri.alive_time, ri.respawn_time, ri.expansion_power);
    gpu.size = fan::vec4(ri.start_size, ri.end_size);
    gpu.color0 = ri.begin_color;
    gpu.color1 = ri.end_color;
    gpu.velocity = fan::vec4(ri.start_velocity, ri.end_velocity);
    gpu.angle_velocity0 = fan::vec4(ri.start_angle_velocity, ri.begin_angle);
    gpu.angle_velocity1 = fan::vec4(ri.end_angle_velocity, ri.end_angle);
    gpu.angle = fan::vec4(ri.angle, 0.f);
    gpu.spawn_spread0 = fan::vec4(ri.spawn_spacing, ri.start_spread);
    gpu.spread1_jitter = fan::vec4(ri.end_spread, ri.jitter_speed, 0.f);
    gpu.jitter_random_size = fan::vec4(ri.jitter_start, ri.jitter_end);
    gpu.color_random = ri.color_random_range;
    gpu.angle_random = fan::vec4(ri.angle_random_range, ri.size_random_range.x);
    return gpu;
  };

  auto ensure_host_buffer = [&] (
    fan::vulkan::context_t::buffer_t& buffer,
    VkDeviceSize& capacity,
    VkDeviceSize wanted
  ) {
    if (wanted <= capacity && buffer) { return; }
    if (buffer.mapped) { context.unmap_buffer(buffer); }
    context.destroy_buffer(buffer);
    capacity = std::max<VkDeviceSize>(wanted, capacity ? capacity * 2 : wanted);
    context.create_buffer(
      capacity,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      buffer
    );
    fan::vulkan::validate(context.map_buffer(buffer, &buffer.mapped));
  };

  std::vector<fan::graphics::polygon_vertex_t> polygon_vertices;
  std::vector<particles_gpu_t> particle_emitters;
  fan::time::global_profiler.end("Variable Initialization");

  fan::time::global_profiler.begin("Pre-traverse shapes");

  {
    fan::graphics::shaper_t::KeyTraverse_t pre_keys;
    pre_keys.Init(shaper);
    while (pre_keys.Loop(shaper)) {
      if (!pre_keys.isbm) { continue; }
      fan::graphics::shaper_t::BlockTraverse_t pre_block;
      auto pre_shape_type = pre_block.Init(shaper, pre_keys.bmid());
      if (pre_shape_type == fan::graphics::shapes::shape_type_t::polygon) {
        fan::time::global_profiler.begin(fan::graphics::shape_names[pre_shape_type]);
        do {
          auto* pri = (fan::graphics::shapes::polygon_t::ri_t*)pre_block.GetData(shaper);
          for (std::uint32_t i = 0; i < pre_block.GetAmount(shaper); ++i) {
            auto& ri = pri[i];
            fan::graphics::shapes::shape_ids_t::nr_t id;
            id.gint() = ri.shape_id;
            auto& sd = fan::graphics::g_shapes->shape_ids[id];
            auto& props = *static_cast<fan::graphics::shapes::polygon_t::properties_t*>(
              fan::graphics::g_shapes->shape_props_getters[sd.shape_type](
                fan::graphics::g_shapes->shape_pool_storage[sd.shape_type],
                sd.data_nr
              )
            );

            ri.vk_first_vertex = (std::uint32_t)polygon_vertices.size();
            ri.vk_vertex_count = (std::uint32_t)props.vertices.size();
            for (auto& v : props.vertices) {
              fan::graphics::polygon_vertex_t pv;
              pv.position = v.position;
              pv.color = v.color;
              pv.offset = props.position;
              pv.angle = props.angle;
              pv.rotation_point = props.rotation_point;
              polygon_vertices.push_back(pv);
            }
          }
        } while (pre_block.Loop(shaper));
        fan::time::global_profiler.end(fan::graphics::shape_names[pre_shape_type]);
      }
      else if (pre_shape_type == fan::graphics::shapes::shape_type_t::particles) {
        fan::time::global_profiler.begin(fan::graphics::shape_names[pre_shape_type]);
        do {
          auto* pri = (fan::graphics::shapes::particles_t::ri_t*)pre_block.GetData(shaper);
          for (std::uint32_t i = 0; i < pre_block.GetAmount(shaper); ++i) {
            pri[i].vk_emitter_index = (std::uint32_t)particle_emitters.size();
            particle_emitters.push_back(make_particles_gpu(pri[i]));
          }
        } while (pre_block.Loop(shaper));
        fan::time::global_profiler.end(fan::graphics::shape_names[pre_shape_type]);
      }
    }

    fan::time::global_profiler.end("Pre-traverse shapes");
    fan::time::global_profiler.begin("Buffer Mapping");

    auto frame = context.current_frame;
    if (!polygon_vertices.empty()) {
      VkDeviceSize bytes = sizeof(polygon_vertices[0]) * polygon_vertices.size();
      ensure_host_buffer(polygon_draw_buffers[frame], polygon_draw_capacity[frame], bytes);
      std::memcpy(polygon_draw_buffers[frame].mapped, polygon_vertices.data(), (std::size_t)bytes);
    }
    if (!particle_emitters.empty()) {
      VkDeviceSize bytes = sizeof(particle_emitters[0]) * particle_emitters.size();
      ensure_host_buffer(particle_draw_buffers[frame], particle_draw_capacity[frame], bytes);
      std::memcpy(particle_draw_buffers[frame].mapped, particle_emitters.data(), (std::size_t)bytes);
    }
  }

  fan::time::global_profiler.end("Buffer Mapping");
  fan::time::global_profiler.begin("Draw Loop passes");

  for (std::uint32_t draw_pass = 0; draw_pass < 2; ++draw_pass) {
    KeyTraverse.Init(shaper);
    viewport.sic();
    camera.sic();
    texture.sic();
    shader_nr.sic();
    visible = true;
    draw_mode = fan::graphics::primitive_topology_t::triangles;
    vertex_count = (std::uint32_t)-1;

    while (KeyTraverse.Loop(shaper)) {
    fan::graphics::shaper_t::KeyTypeIndex_t kti = KeyTraverse.kti(shaper);

    switch (kti) {
      case fan::graphics::Key_e::ShapeType: {
        if (*(fan::graphics::shaper_t::ShapeTypeIndex_t*)KeyTraverse.kd() == fan::graphics::shapes::shape_type_t::light_end) {
          continue;
        }
        break;
      }
      case fan::graphics::Key_e::image: {
        texture = *(fan::graphics::image_t*)KeyTraverse.kd();
        break;
      }
      case fan::graphics::Key_e::viewport: {
        viewport = *(fan::graphics::viewport_t*)KeyTraverse.kd();
        break;
      }
      case fan::graphics::Key_e::camera: {
        camera = *(loco_t::camera_t*)KeyTraverse.kd();
        camera_id = (std::uint32_t)camera.NRI;
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

    if (!KeyTraverse.isbm) {
      continue;
    }

    fan::graphics::shaper_t::BlockTraverse_t BlockTraverse;
    fan::graphics::shaper_t::ShapeTypeIndex_t shape_type = BlockTraverse.Init(shaper, KeyTraverse.bmid());

    if (shape_type == fan::graphics::shapes::shape_type_t::light_end) {
      continue;
    }
    if ((shape_type == fan::graphics::shapes::shape_type_t::light) != (draw_pass == 1)) {
      continue;
    }
    if (!visible) {
      continue;
    }

    auto default_shader_nr = fan::graphics::g_shapes->shaper.GetShader(shape_type);
    auto shape_shader_nr = shader_nr.iic() ? default_shader_nr : shader_nr;
    shader_nr.sic();
    auto& vk_data = shaper.GetShapeTypes(shape_type).renderer.vk;

    if (shape_type == fan::graphics::shapes::shape_type_t::polygon) {
      if (polygon_vertices.empty()) { continue; }
      if (!shape_shader_nr) continue;
      fan::time::global_profiler.begin(fan::graphics::shape_names[shape_type]);
      auto& pipeline = prepare(shape_type, shape_shader_nr);
      fan::vulkan::context_t::push_constants_t pc{};
      pc.camera_id = camera_id;
      pc.texture_id = texture_id(texture);
      push(pipeline, pc);
      do {
        auto* pri = (fan::graphics::shapes::polygon_t::ri_t*)BlockTraverse.GetData(shaper);
        for (std::uint32_t i = 0; i < BlockTraverse.GetAmount(shaper); ++i) {
          auto& ri = pri[i];
          if (ri.vk_vertex_count == 0) { continue; }
          vkCmdDraw(cmd_buffer, ri.vk_vertex_count, 1, ri.vk_first_vertex, 0);
          did_draw = true;
        }
      } while (BlockTraverse.Loop(shaper));
      fan::time::global_profiler.end(fan::graphics::shape_names[shape_type]);
      continue;
    }

    if (shape_type == fan::graphics::shapes::shape_type_t::particles) {
      if (particle_emitters.empty()) { 
        continue; 
      }
      fan::time::global_profiler.begin(fan::graphics::shape_names[shape_type]);
      auto& pipeline = prepare(shape_type, shape_shader_nr);
      fan::vulkan::context_t::push_constants_t pc{};
      pc.camera_id = camera_id;
      pc.texture_id = texture_id(texture);
      push(pipeline, pc);
      do {
        auto* pri = (fan::graphics::shapes::particles_t::ri_t*)BlockTraverse.GetData(shaper);
        if (BlockTraverse.GetAmount(shaper) > 0) {
          std::uint32_t max_count = 0;
          for (std::uint32_t i = 0; i < BlockTraverse.GetAmount(shaper); ++i) {
            max_count = std::max(max_count, pri[i].count);
          }
          vkCmdDraw(cmd_buffer, max_count * 6, BlockTraverse.GetAmount(shaper), 0, pri[0].vk_emitter_index);
          did_draw = true;
        }
      } while (BlockTraverse.Loop(shaper));
      fan::time::global_profiler.end(fan::graphics::shape_names[shape_type]);
      continue;
    }

    if (shape_shader_nr) {
      fan::time::global_profiler.begin(fan::graphics::shape_names[shape_type]);
      do {
        auto& pipeline = prepare(shape_type, shape_shader_nr);
        fan::vulkan::context_t::push_constants_t pc{};
        pc.camera_id = camera_id;
        pc.texture_id = texture_id(texture);

        auto off = BlockTraverse.GetRenderDataOffset(shaper) / shaper.GetRenderDataSize(shape_type);
        auto draw_vertex_count = vertex_count == (std::uint32_t)-1 ? vk_data.vertex_count : vertex_count;

        if (shape_type == fan::graphics::shapes::shape_type_t::universal_image_renderer) {
          auto* pri = (fan::graphics::shapes::universal_image_renderer_t::ri_t*)BlockTraverse.GetData(shaper);
          for (std::uint32_t i = 0; i < BlockTraverse.GetAmount(shaper); ++i) {
            pc.texture_id1 = texture_id(pri[i].images_rest[0]);
            pc.texture_id2 = texture_id(pri[i].images_rest[1]);
            pc.texture_id3 = texture_id(pri[i].images_rest[2]);
            push(pipeline, pc);
            vkCmdDraw(cmd_buffer, draw_vertex_count, 1, 0, off + i);
            did_draw = true;
          }
        }
        else {
          push(pipeline, pc);
          vkCmdDraw(
            cmd_buffer,
            draw_vertex_count,
            BlockTraverse.GetAmount(shaper),
            0,
            off
          );
          did_draw = true;
        }
      } while (BlockTraverse.Loop(shaper));
      fan::time::global_profiler.end(fan::graphics::shape_names[shape_type]);
    }
  }

  }

  fan::time::global_profiler.end("Draw Loop passes");

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
  loco.shader_set_compute(bloom_downsample_shader, "shaders/vulkan/downsample.comp", fan::graphics::read_shader("shaders/vulkan/downsample.comp"));
  loco.shader_compile(bloom_downsample_shader);

  bloom_upsample_shader = loco.shader_create();
  loco.shader_set_compute(bloom_upsample_shader, "shaders/vulkan/upsample.comp", fan::graphics::read_shader("shaders/vulkan/upsample.comp"));
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