struct alpha_shadow_renderer_t {
  loco_t* loco_ptr = nullptr;

  struct light_t {
    fan::vec2 position = 0;
    f32_t radius = 512.f;
    fan::color color = fan::colors::white;
    fan::graphics::render_view_t* render_view = &fan::graphics::get_orthographic_render_view();
    f32_t softness = 0.02f;
    f32_t falloff_power = 2.f;
    f32_t angle = 0.f;
    f32_t cone_inner = 6.28318530718f;
    f32_t cone_outer = 6.28318530718f;
  };

  struct caster_t {
    fan::graphics::shape_t* shape = nullptr;
    f32_t alpha_threshold = 0.05f;
  };

  void open(std::int32_t occluder_resolution_ = 1024, std::int32_t angle_resolution_ = 2048, std::int32_t radial_samples_ = 160) {
    occluder_resolution = occluder_resolution_;
    angle_resolution = angle_resolution_;
    radial_samples = radial_samples_;

    fan::vulkan::context_t& context = loco_ptr->context.vk;

    {
      fan::vulkan::vai_t::properties_t p{
        .swap_chain_size = fan::vec2(occluder_resolution, occluder_resolution),
        .format = VK_FORMAT_R16_SFLOAT,
        .usage_flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .aspect_flags = VK_IMAGE_ASPECT_COLOR_BIT,
      };
      occluder_texture.open(context, p);
    }
    {
      fan::vulkan::vai_t::properties_t p{
        .swap_chain_size = fan::vec2(angle_resolution, 1),
        .format = VK_FORMAT_R16_SFLOAT,
        .usage_flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        .aspect_flags = VK_IMAGE_ASPECT_COLOR_BIT,
      };
      shadow_texture.open(context, p);
    }

    fan::vulkan::image_load_properties_t nearest_lp;
    nearest_lp.min_filter = VK_FILTER_NEAREST;
    nearest_lp.mag_filter = VK_FILTER_NEAREST;
    nearest_lp.visual_output = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    loco_ptr->context.vk.create_texture_sampler(occluder_sampler, nearest_lp);

    fan::vulkan::image_load_properties_t linear_repeat_lp;
    linear_repeat_lp.min_filter = VK_FILTER_LINEAR;
    linear_repeat_lp.mag_filter = VK_FILTER_LINEAR;
    linear_repeat_lp.visual_output = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    loco_ptr->context.vk.create_texture_sampler(shadow_sampler, linear_repeat_lp);

    auto load_shader = [&](const char* vs, const char* fs) {
      fan::graphics::shader_t nr = loco_ptr->shader_create();
      loco_ptr->shader_set_vertex(nr, vs, fan::graphics::read_shader(vs));
      loco_ptr->shader_set_fragment(nr, fs, fan::graphics::read_shader(fs));
      loco_ptr->shader_compile(nr);
      return nr;
    };

    occluder_shader = load_shader(
      "shaders/vulkan/2D/effects/alpha_shadow_occluder.vert",
      "shaders/vulkan/2D/effects/alpha_shadow_occluder.frag"
    );
    radial_shader = loco_ptr->shader_create();
    loco_ptr->shader_set_compute(radial_shader,
      "shaders/vulkan/2D/effects/alpha_shadow_radial.comp",
      fan::graphics::read_shader("shaders/vulkan/2D/effects/alpha_shadow_radial.comp"));
    loco_ptr->shader_compile(radial_shader);
    light_shader = load_shader(
      "shaders/vulkan/2D/effects/alpha_shadow_light.vert",
      "shaders/vulkan/2D/effects/alpha_shadow_light.frag"
    );
    solid_shader = load_shader(
      "shaders/vulkan/2D/effects/alpha_shadow_solid.vert",
      "shaders/vulkan/2D/effects/alpha_shadow_solid.frag"
    );

    occluder_descriptor_layout = make_push_descriptor_layout({
      {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
    });
    radial_descriptor_layout = make_push_descriptor_layout({
      {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
    });
    light_descriptor_layout = make_push_descriptor_layout({
      {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
    });
    solid_descriptor_layout = make_push_descriptor_layout({});

    auto make_pipeline = [&](fan::vulkan::context_t::pipeline_t& pipe, fan::graphics::shader_t shader,
      VkDescriptorSetLayout dsl, uint32_t pc_size, bool) {
      fan::vulkan::context_t::pipeline_t::properties_t p;
      p.shader = shader;
      p.descriptor_layouts = {dsl};
      p.push_constants_size = pc_size;
      p.color_blend_attachments = {{}};
      p.enable_depth_test = false;
      pipe.open(context, p);
    };

    make_pipeline(occluder_pipeline, occluder_shader, occluder_descriptor_layout, sizeof(occluder_push_t), true);
    {
      fan::vulkan::context_t::compute_pipeline_t::properties_t p;
      p.shader = radial_shader;
      p.descriptor_layouts = {radial_descriptor_layout};
      p.push_constants_size = sizeof(radial_push_t);
      radial_pipeline.open(context, p);
    }
    make_pipeline(light_pipeline, light_shader, light_descriptor_layout, sizeof(light_push_t), true);
    make_pipeline(solid_pipeline, solid_shader, solid_descriptor_layout, sizeof(solid_push_t), true);

    resources_open = true;
  }

  void close() {
    fan::vulkan::context_t& context = loco_ptr->context.vk;
    vkDeviceWaitIdle(context.device);
    for (fan::graphics::shader_t s : {occluder_shader, radial_shader, light_shader, solid_shader}) {
      if (!s.iic()) { loco_ptr->shader_erase(s); }
    }
    occluder_pipeline.close(context);
    radial_pipeline.close(context);
    light_pipeline.close(context);
    solid_pipeline.close(context);
    vkDestroyDescriptorSetLayout(context.device, occluder_descriptor_layout, nullptr);
    vkDestroyDescriptorSetLayout(context.device, radial_descriptor_layout, nullptr);
    vkDestroyDescriptorSetLayout(context.device, light_descriptor_layout, nullptr);
    vkDestroyDescriptorSetLayout(context.device, solid_descriptor_layout, nullptr);
    if (occluder_sampler) { vkDestroySampler(context.device, occluder_sampler, nullptr); }
    if (shadow_sampler) { vkDestroySampler(context.device, shadow_sampler, nullptr); }
    occluder_texture.close(context);
    shadow_texture.close(context);
    casters.clear();
    lights.clear();
    *this = {};
  }

  void build_shadow_maps() {
    if (!resources_open || casters.empty() || lights.empty()) { return; }
    for (const light_t& light : lights) {
      render_occluders(light);
      barrier_occluder_to_read();
      render_radial();
      barrier_shadow_to_read();
    }
  }

  void render_overlay(VkImageView swapchain_image_view) {
    if (!resources_open || casters.empty() || lights.empty()) { return; }
    fan::vulkan::context_t& context = loco_ptr->context.vk;

    {
      VkRenderingAttachmentInfo att{};
      att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
      att.imageView = swapchain_image_view;
      att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
      att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      VkRenderingInfo ri{};
      ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
      ri.renderArea = {{0, 0}, {(uint32_t)loco_ptr->window.get_size().x, (uint32_t)loco_ptr->window.get_size().y}};
      ri.layerCount = 1;
      ri.colorAttachmentCount = 1;
      ri.pColorAttachments = &att;
      vkCmdBeginRendering(cmd(), &ri);
    }

    {
      fan::vec2ui sz{(uint32_t)loco_ptr->window.get_size().x, (uint32_t)loco_ptr->window.get_size().y};
      VkViewport vp{0, 0, (float)sz.x, (float)sz.y, 0, 1};
      vkCmdSetViewport(cmd(), 0, 1, &vp);
      VkRect2D sc{{0, 0}, {sz.x, sz.y}};
      vkCmdSetScissor(cmd(), 0, 1, &sc);
    }

    render_darkness();
    for (const light_t& light : lights) {
      render_light(light);
    }

    vkCmdEndRendering(cmd());
  }

  std::vector<caster_t> casters;
  std::vector<light_t> lights;
  f32_t darkness = 0.78f;

private:
  static fan::vec2 rotate(fan::vec2 p, f32_t a) {
    f32_t c = std::cos(a), s = std::sin(a);
    return {p.x * c - p.y * s, p.x * s + p.y * c};
  }

  static fan::vec2 world_to_screen(fan::vec2 p, const fan::graphics::render_view_t& rv) {
    return fan::graphics::world_to_screen(p, rv);
  }

  static fan::vec2 screen_to_ndc(fan::vec2 p, fan::vec2 window_size) {
    return {p.x / window_size.x * 2.f - 1.f, p.y / window_size.y * 2.f - 1.f};
  }

  VkCommandBuffer cmd() {
    return loco_ptr->context.vk.command_buffers[loco_ptr->context.vk.current_frame];
  }

  VkDescriptorSetLayout make_push_descriptor_layout(const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT;
    info.bindingCount = (uint32_t)bindings.size();
    info.pBindings = bindings.data();
    VkDescriptorSetLayout layout;
    fan::vulkan::validate(vkCreateDescriptorSetLayout(loco_ptr->context.vk.device, &info, nullptr, &layout));
    return layout;
  }

  void bind_pipeline(fan::vulkan::context_t::pipeline_t& pipeline) {
    VkShaderStageFlagBits stages[2] = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT};
    fan_vkCmdBindShadersEXT(cmd(), 2, stages, pipeline.m_shaders);
    vkCmdSetPrimitiveTopology(cmd(), VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    vkCmdSetRasterizerDiscardEnable(cmd(), VK_FALSE);
    fan_vkCmdSetPolygonModeEXT(cmd(), VK_POLYGON_MODE_FILL);
    fan_vkCmdSetCullMode(cmd(), VK_CULL_MODE_NONE);
    fan_vkCmdSetFrontFace(cmd(), VK_FRONT_FACE_CLOCKWISE);
    vkCmdSetDepthTestEnable(cmd(), VK_FALSE);
    vkCmdSetDepthWriteEnable(cmd(), VK_FALSE);
    fan_vkCmdSetVertexInputEXT(cmd(), 0, nullptr, 0, nullptr);
  }

  void set_blend(VkBool32 enable, VkBlendFactor src_rgb = VK_BLEND_FACTOR_SRC_ALPHA,
    VkBlendFactor dst_rgb = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
    VkBlendFactor src_a = VK_BLEND_FACTOR_ONE,
    VkBlendFactor dst_a = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) {
    fan_vkCmdSetColorBlendEnableEXT(cmd(), 0, 1, &enable);
    if (enable) {
      VkColorBlendEquationEXT eq{src_rgb, dst_rgb, VK_BLEND_OP_ADD, src_a, dst_a, VK_BLEND_OP_ADD};
      fan_vkCmdSetColorBlendEquationEXT(cmd(), 0, 1, &eq);
    }
  }

  void barrier_occluder_to_read() {
    VkImageMemoryBarrier2 b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    b.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.image = occluder_texture.image;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    VkDependencyInfo d{};
    d.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    d.imageMemoryBarrierCount = 1;
    d.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd(), &d);
  }

  void barrier_shadow_to_read() {
    VkImageMemoryBarrier2 b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    b.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.image = shadow_texture.image;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    VkDependencyInfo d{};
    d.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    d.imageMemoryBarrierCount = 1;
    d.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd(), &d);
  }

  void barrier_occluder_to_write() {
    VkImageMemoryBarrier2 b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    b.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    b.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    b.image = occluder_texture.image;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    VkDependencyInfo d{};
    d.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    d.imageMemoryBarrierCount = 1;
    d.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd(), &d);
  }

  void barrier_shadow_to_write() {
    VkImageMemoryBarrier2 b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    b.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    b.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.image = shadow_texture.image;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    VkDependencyInfo d{};
    d.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    d.imageMemoryBarrierCount = 1;
    d.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd(), &d);
  }

  void render_darkness() {
    set_blend(VK_TRUE);
    bind_pipeline(solid_pipeline);
    solid_push_t pc{fan::color(0, 0, 0, darkness)};
    vkCmdPushConstants(cmd(), solid_pipeline.m_layout,
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
    vkCmdDraw(cmd(), 6, 1, 0, 0);
  }

  void render_occluders(const light_t& light) {
    fan::vulkan::context_t& context = loco_ptr->context.vk;

    barrier_occluder_to_write();

    fan::vec2 ls = world_to_screen(light.position, *light.render_view);
    fan::vec2 le = world_to_screen(light.position + fan::vec2(light.radius, 0), *light.render_view);
    f32_t lr = std::max(1.f, std::abs(le.x - ls.x));

    {
      VkRenderingAttachmentInfo att{};
      att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
      att.imageView = occluder_texture.image_view;
      att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      att.clearValue.color = {};
      VkRenderingInfo ri{};
      ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
      ri.renderArea = {{0, 0}, {(uint32_t)occluder_resolution, (uint32_t)occluder_resolution}};
      ri.layerCount = 1;
      ri.colorAttachmentCount = 1;
      ri.pColorAttachments = &att;
      vkCmdBeginRendering(cmd(), &ri);
    }

    VkViewport vp{0, 0, (float)occluder_resolution, (float)occluder_resolution, 0, 1};
    vkCmdSetViewport(cmd(), 0, 1, &vp);
    VkRect2D sc{{0, 0}, {(uint32_t)occluder_resolution, (uint32_t)occluder_resolution}};
    vkCmdSetScissor(cmd(), 0, 1, &sc);

    bind_pipeline(occluder_pipeline);
    set_blend(VK_TRUE, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE,
      VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA);

    for (const caster_t& caster : casters) {
      if (!caster.shape || !*caster.shape) { continue; }
      fan::graphics::texture_pack::ti_t ti = caster.shape->get_tp();
      if (!ti.image.valid()) { continue; }
      fan::vec2 isz = ti.image.get_size();
      if (isz.x <= 0 || isz.y <= 0) { continue; }
      std::uint32_t tex_id = ti.image.NRI;
      if (tex_id >= context.image_pool.size()) { continue; }

      fan::vec2 pos = caster.shape->get_position();
      fan::vec2 size = caster.shape->get_size();
      fan::vec2 pivot = caster.shape->get_rotation_point();
      f32_t angle = caster.shape->get_angle().z;

      auto mp = [&](fan::vec2 local) -> fan::vec2 {
        fan::vec2 world = pos + pivot + rotate(local - pivot, angle);
        fan::vec2 sp = world_to_screen(world, *light.render_view);
        return (sp - ls) / lr;
      };

      fan::vec2 corners[4] = {
        mp({-size.x, -size.y}),
        mp({ size.x, -size.y}),
        mp({ size.x,  size.y}),
        mp({-size.x,  size.y}),
      };

      bool outside = true;
      for (auto& c : corners) {
        if (std::abs(c.x) <= 1.25f && std::abs(c.y) <= 1.25f) { outside = false; break; }
      }
      if (outside) { continue; }

      fan::vec2 uv0 = ti.position / isz;
      fan::vec2 uv1 = uv0 + ti.size / isz;

      occluder_push_t pc{corners[0], corners[1], corners[2], corners[3], uv0, uv1, caster.alpha_threshold};

      VkDescriptorImageInfo img_info{};
      img_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      img_info.imageView = context.image_pool[tex_id].imageView;
      img_info.sampler = context.image_pool[tex_id].sampler;

      VkWriteDescriptorSet w{};
      w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w.dstBinding = 0;
      w.descriptorCount = 1;
      w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w.pImageInfo = &img_info;
      vkCmdPushDescriptorSet(cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS, occluder_pipeline.m_layout, 0, 1, &w);

      vkCmdPushConstants(cmd(), occluder_pipeline.m_layout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
      vkCmdDraw(cmd(), 6, 1, 0, 0);
    }

    vkCmdEndRendering(cmd());
  }

  void render_radial() {
    barrier_shadow_to_write();

    VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
    fan_vkCmdBindShadersEXT(cmd(), 1, &stage, &radial_pipeline.shader);

    VkDescriptorImageInfo sampler_info{};
    sampler_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    sampler_info.imageView = occluder_texture.image_view;
    sampler_info.sampler = occluder_sampler;

    VkDescriptorImageInfo storage_info{};
    storage_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    storage_info.imageView = shadow_texture.image_view;
    storage_info.sampler = VK_NULL_HANDLE;

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &sampler_info;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &storage_info;
    vkCmdPushDescriptorSet(cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, radial_pipeline.pipeline_layout, 0, 2, writes);

    radial_push_t pc{(std::uint32_t)angle_resolution, (std::uint32_t)radial_samples};
    vkCmdPushConstants(cmd(), radial_pipeline.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    std::uint32_t groups = (angle_resolution + 255) / 256;
    vkCmdDispatch(cmd(), groups, 1, 1);
  }

  void render_light(const light_t& light) {
    fan::vec2 window_size = loco_ptr->window.get_size();

    fan::vec2 center = world_to_screen(light.position, *light.render_view);
    fan::vec2 edge = world_to_screen(light.position + fan::vec2(light.radius, 0), *light.render_view);
    f32_t r = std::max(1.f, std::abs(edge.x - center.x));
    fan::vec2 p0 = center - r, p1 = center + r;

    fan::vec2 ndc_light_min = screen_to_ndc(p0, window_size);
    fan::vec2 ndc_light_max = screen_to_ndc(p1, window_size);

    bind_pipeline(light_pipeline);
    set_blend(VK_TRUE, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE,
      VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA);

    VkDescriptorImageInfo img{};
    img.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img.imageView = shadow_texture.image_view;
    img.sampler = shadow_sampler;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &img;
    vkCmdPushDescriptorSet(cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS, light_pipeline.m_layout, 0, 1, &w);

    light_push_t pc{ndc_light_min, ndc_light_max, light.color, light.softness, light.falloff_power,
      1.f / f32_t(angle_resolution), light.angle, light.cone_inner, light.cone_outer};
    vkCmdPushConstants(cmd(), light_pipeline.m_layout,
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
    vkCmdDraw(cmd(), 6, 1, 0, 0);
  }

  std::int32_t occluder_resolution = 1024;
  std::int32_t angle_resolution = 2048;
  std::int32_t radial_samples = 160;

  fan::vulkan::vai_t occluder_texture;
  fan::vulkan::vai_t shadow_texture;
  VkSampler occluder_sampler = VK_NULL_HANDLE;
  VkSampler shadow_sampler = VK_NULL_HANDLE;

  fan::graphics::shader_t occluder_shader;
  fan::graphics::shader_t radial_shader;
  fan::graphics::shader_t light_shader;
  fan::graphics::shader_t solid_shader;

  fan::vulkan::context_t::pipeline_t occluder_pipeline;
  fan::vulkan::context_t::compute_pipeline_t radial_pipeline;
  fan::vulkan::context_t::pipeline_t light_pipeline;
  fan::vulkan::context_t::pipeline_t solid_pipeline;

  VkDescriptorSetLayout occluder_descriptor_layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout radial_descriptor_layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout light_descriptor_layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout solid_descriptor_layout = VK_NULL_HANDLE;

  bool resources_open = false;

  struct occluder_push_t { fan::vec2 c0; fan::vec2 c1; fan::vec2 c2; fan::vec2 c3; fan::vec2 uv_min; fan::vec2 uv_max; f32_t alpha_threshold; };
  struct radial_push_t { std::uint32_t angle_resolution; std::uint32_t radial_samples; };
  struct light_push_t { fan::vec2 ndc_min; fan::vec2 ndc_max; fan::color light_color; float softness; float falloff_power; float angle_texel; float cone_angle; float cone_inner; float cone_outer; };
  struct solid_push_t { fan::color color; };
};

alpha_shadow_renderer_t alpha_shadow_renderer;
