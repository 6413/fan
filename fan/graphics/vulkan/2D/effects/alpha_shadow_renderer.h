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
    auto& ctx = loco_ptr->context.vk;

    occluder_texture.open(ctx, {fan::vec2(occluder_resolution, occluder_resolution), VK_FORMAT_R16_SFLOAT,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT});
    shadow_texture.open(ctx, {fan::vec2(angle_resolution, 1), VK_FORMAT_R16_SFLOAT,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT});

    loco_ptr->context.vk.create_texture_sampler(occluder_sampler, fan::vulkan::image_load_properties_t{
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0, VK_FORMAT_R16_SFLOAT, VK_FILTER_NEAREST, VK_FILTER_NEAREST});
    loco_ptr->context.vk.create_texture_sampler(shadow_sampler, fan::vulkan::image_load_properties_t{
      VK_SAMPLER_ADDRESS_MODE_REPEAT, 0, VK_FORMAT_R16_SFLOAT, VK_FILTER_LINEAR, VK_FILTER_LINEAR});

    auto load = [&](const char* vs, const char* fs, bool compute = false) {
      fan::graphics::shader_t nr = loco_ptr->shader_create();
      if (compute) {
        loco_ptr->shader_set_compute(nr, vs, fan::graphics::read_shader(vs));
      }
      else {
        loco_ptr->shader_set_vertex(nr, vs, fan::graphics::read_shader(vs));
        loco_ptr->shader_set_fragment(nr, fs, fan::graphics::read_shader(fs));
      }
      loco_ptr->shader_compile(nr);
      return nr;
    };

    occluder_shader = load("shaders/vulkan/2D/effects/alpha_shadow_occluder.vert", "shaders/vulkan/2D/effects/alpha_shadow_occluder.frag");
    radial_shader   = load("shaders/vulkan/2D/effects/alpha_shadow_radial.comp", nullptr, true);
    light_shader    = load("shaders/vulkan/2D/effects/alpha_shadow_light.vert", "shaders/vulkan/2D/effects/alpha_shadow_light.frag");
    solid_shader    = load("shaders/vulkan/2D/effects/alpha_shadow_solid.vert", "shaders/vulkan/2D/effects/alpha_shadow_solid.frag");

    occluder_dsl = make_dsl({{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT}});
    radial_dsl   = make_dsl({{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                             {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,       1, VK_SHADER_STAGE_COMPUTE_BIT}});
    light_dsl    = make_dsl({{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT}});
    solid_dsl    = make_dsl({});

    auto mk_raster_pipe = [&](fan::vulkan::context_t::pipeline_t& pipe, fan::graphics::shader_t sh, VkDescriptorSetLayout dsl, uint32_t pc_size) {
      pipe.open(ctx, {.descriptor_layouts = {dsl}, .shader = sh, .push_constants_size = pc_size, .color_blend_attachments = {{}}, .enable_depth_test = false});
    };
    mk_raster_pipe(occluder_pipeline, occluder_shader, occluder_dsl, sizeof(occluder_push_t));
    mk_raster_pipe(light_pipeline,    light_shader,    light_dsl,    sizeof(light_push_t));
    mk_raster_pipe(solid_pipeline,    solid_shader,    solid_dsl,    sizeof(solid_push_t));
    radial_pipeline.open(ctx, {.descriptor_layouts = {radial_dsl}, .shader = radial_shader, .push_constants_size = sizeof(radial_push_t)});

    resources_open = true;
  }

  void close() {
    auto& ctx = loco_ptr->context.vk;
    vkDeviceWaitIdle(ctx.device);
    for (fan::graphics::shader_t s : {occluder_shader, radial_shader, light_shader, solid_shader}) {
      if (!s.iic()) { loco_ptr->shader_erase(s); }
    }
    for (auto* pipe : {&occluder_pipeline, &light_pipeline, &solid_pipeline}) { pipe->close(ctx); }
    radial_pipeline.close(ctx);
    for (auto* dsl : {occluder_dsl, radial_dsl, light_dsl, solid_dsl}) { vkDestroyDescriptorSetLayout(ctx.device, dsl, nullptr); }
    if (occluder_sampler) { vkDestroySampler(ctx.device, occluder_sampler, nullptr); }
    if (shadow_sampler) { vkDestroySampler(ctx.device, shadow_sampler, nullptr); }
    occluder_texture.close(ctx);
    shadow_texture.close(ctx);
    casters.clear();
    lights.clear();
    *this = {};
  }

  void build_shadow_maps() {
    if (!resources_open || casters.empty() || lights.empty()) { return; }
    for (const light_t& light : lights) {
      render_occluders(light);
      barrier(occluder_texture.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
      render_radial();
      barrier(shadow_texture.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    }
  }

  void render_overlay(VkImageView swapchain_image_view) {
    if (!resources_open || lights.empty()) { return; }
    auto& ctx = loco_ptr->context.vk;
    fan::vec2ui sz{(uint32_t)loco_ptr->window.get_size().x, (uint32_t)loco_ptr->window.get_size().y};
    VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, swapchain_image_view,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
      VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, {}};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0, 0}, {sz.x, sz.y}}, 1, 0, 1, &att, nullptr, nullptr};
    vkCmdBeginRendering(cmd(), &ri);
    VkViewport vp0{0, 0, (float)sz.x, (float)sz.y, 0, 1};
    VkRect2D sc0{{0, 0}, {sz.x, sz.y}};
    vkCmdSetViewport(cmd(), 0, 1, &vp0);
    vkCmdSetScissor(cmd(), 0, 1, &sc0);

    bind_and_blend(solid_pipeline, VK_TRUE);
    solid_push_t sc{fan::color(0, 0, 0, darkness)};
    vkCmdPushConstants(cmd(), solid_pipeline.m_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(sc), &sc);
    vkCmdDraw(cmd(), 6, 1, 0, 0);

    bool has_tiles = !tile_occluders.empty();
    for (std::size_t li = 0; li < lights.size(); ++li) {
      if (has_tiles) {
        vkCmdEndRendering(cmd());
        build_tile_shadow_map(tile_occluders, lights[li].position, lights[li].radius);
        barrier(shadow_texture.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
          VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        VkRenderingAttachmentInfo att2{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, swapchain_image_view,
          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
          VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, {}};
        VkRenderingInfo ri2{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0, 0}, {sz.x, sz.y}}, 1, 0, 1, &att2, nullptr, nullptr};
        vkCmdBeginRendering(cmd(), &ri2);
        VkViewport vp2{0, 0, (float)sz.x, (float)sz.y, 0, 1};
        VkRect2D sc2{{0, 0}, {sz.x, sz.y}};
        vkCmdSetViewport(cmd(), 0, 1, &vp2);
        vkCmdSetScissor(cmd(), 0, 1, &sc2);
      }
      render_light(lights[li]);
    }
    if (has_tiles && tile_data_dirty) { tile_occluders.clear(); }
    vkCmdEndRendering(cmd());
  }

  std::vector<caster_t> casters;
  std::vector<light_t> lights;
  std::vector<fan::vec4> tile_occluders;
  bool tile_mode_open = false;
  f32_t darkness = 0.78f;
  bool tile_data_dirty = true;
  uint32_t cached_tile_count = 0;

  VkCommandBuffer cmd() { return loco_ptr->context.vk.command_buffers[loco_ptr->context.vk.current_frame]; }

  VkDescriptorSetLayout make_dsl(const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr,
      VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT, (uint32_t)bindings.size(), bindings.data()};
    VkDescriptorSetLayout layout;
    fan::vulkan::validate(vkCreateDescriptorSetLayout(loco_ptr->context.vk.device, &info, nullptr, &layout));
    return layout;
  }

  void barrier(VkImage image, VkImageLayout old_layout, VkImageLayout new_layout,
    VkPipelineStageFlags2 src_stage, VkPipelineStageFlags2 dst_stage,
    VkAccessFlags2 src_access, VkAccessFlags2 dst_access)
  {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, nullptr, src_stage, src_access, dst_stage, dst_access,
      old_layout, new_layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image,
      {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo d{VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 0, nullptr, 0, nullptr, 1, &b};
    vkCmdPipelineBarrier2(cmd(), &d);
  }

  void bind_and_blend(fan::vulkan::context_t::pipeline_t& pipeline, VkBool32 blend,
    VkBlendFactor src_rgb = VK_BLEND_FACTOR_SRC_ALPHA, VkBlendFactor dst_rgb = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
    VkBlendFactor src_a = VK_BLEND_FACTOR_ONE, VkBlendFactor dst_a = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA)
  {
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
    fan_vkCmdSetColorBlendEnableEXT(cmd(), 0, 1, &blend);
    if (blend) {
      VkColorBlendEquationEXT eq{src_rgb, dst_rgb, VK_BLEND_OP_ADD, src_a, dst_a, VK_BLEND_OP_ADD};
      fan_vkCmdSetColorBlendEquationEXT(cmd(), 0, 1, &eq);
    }
  }

  void push_sampler_descriptor(VkPipelineLayout layout, VkImageView view, VkSampler sampler, uint32_t binding = 0) {
    VkDescriptorImageInfo info{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    if (sampler) { info.sampler = sampler; }
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, binding, 0, 1,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &info, nullptr, nullptr};
    vkCmdPushDescriptorSet(cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &w);
  }

  void render_occluders(const light_t& light) {
    auto& ctx = loco_ptr->context.vk;

    barrier(occluder_texture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_ACCESS_2_SHADER_READ_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    fan::vec2 ls = fan::graphics::world_to_screen(light.position, *light.render_view);
    fan::vec2 le = fan::graphics::world_to_screen(light.position + fan::vec2(light.radius, 0), *light.render_view);
    f32_t lr = std::max(1.f, std::abs(le.x - ls.x));

    VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, occluder_texture.image_view,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
      VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0, 0}, {(uint32_t)occluder_resolution, (uint32_t)occluder_resolution}}, 1, 0, 1, &att, nullptr, nullptr};
    vkCmdBeginRendering(cmd(), &ri);
    VkViewport vp_occ{0, 0, (float)occluder_resolution, (float)occluder_resolution, 0, 1};
    VkRect2D sc_occ{{0, 0}, {(uint32_t)occluder_resolution, (uint32_t)occluder_resolution}};
    vkCmdSetViewport(cmd(), 0, 1, &vp_occ);
    vkCmdSetScissor(cmd(), 0, 1, &sc_occ);

    bind_and_blend(occluder_pipeline, VK_TRUE, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA);

    for (const caster_t& caster : casters) {
      if (!caster.shape || !*caster.shape) { continue; }
      fan::graphics::texture_pack::ti_t ti = caster.shape->get_tp();
      if (!ti.image.valid()) { continue; }
      fan::vec2 isz = ti.image.get_size();
      if (isz.x <= 0 || isz.y <= 0) { continue; }
      std::uint32_t tex_id = ti.image.NRI;
      if (tex_id >= ctx.image_pool.size()) { continue; }

      fan::vec2 pos = caster.shape->get_position();
      fan::vec2 size = caster.shape->get_size();
      fan::vec2 pivot = caster.shape->get_rotation_point();
      f32_t a = caster.shape->get_angle().z;

      fan::vec2 c[4] = {
        mp(pos, pivot, size, a, fan::vec2(-size.x, -size.y), ls, lr, *light.render_view),
        mp(pos, pivot, size, a, fan::vec2( size.x, -size.y), ls, lr, *light.render_view),
        mp(pos, pivot, size, a, fan::vec2( size.x,  size.y), ls, lr, *light.render_view),
        mp(pos, pivot, size, a, fan::vec2(-size.x,  size.y), ls, lr, *light.render_view),
      };
      bool outside = true;
      for (auto& v : c) { if (std::abs(v.x) <= 1.25f && std::abs(v.y) <= 1.25f) { outside = false; break; } }
      if (outside) { continue; }

      fan::vec2 uv0 = ti.position / isz;
      fan::vec2 uv1 = uv0 + ti.size / isz;
      occluder_push_t pc{c[0], c[1], c[2], c[3], uv0, uv1, caster.alpha_threshold};

      VkDescriptorImageInfo img{VK_NULL_HANDLE, ctx.image_pool[tex_id].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      if (ctx.image_pool[tex_id].sampler) { img.sampler = ctx.image_pool[tex_id].sampler; }
      VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 0, 0, 1,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &img, nullptr, nullptr};
      vkCmdPushDescriptorSet(cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS, occluder_pipeline.m_layout, 0, 1, &w);

      vkCmdPushConstants(cmd(), occluder_pipeline.m_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
      vkCmdDraw(cmd(), 6, 1, 0, 0);
    }
    vkCmdEndRendering(cmd());
  }

  static fan::vec2 mp(fan::vec2 pos, fan::vec2 pivot, fan::vec2 size, f32_t a, fan::vec2 local, fan::vec2 ls, f32_t lr, const fan::graphics::render_view_t& rv) {
    f32_t c = std::cos(a), s = std::sin(a);
    fan::vec2 world = pos + pivot + fan::vec2((local.x - pivot.x) * c - (local.y - pivot.y) * s, (local.x - pivot.x) * s + (local.y - pivot.y) * c);
    fan::vec2 sp = fan::graphics::world_to_screen(world, rv);
    return (sp - ls) / lr;
  }

  void render_radial() {
    barrier(shadow_texture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_ACCESS_2_SHADER_READ_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);

    VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
    fan_vkCmdBindShadersEXT(cmd(), 1, &stage, &radial_pipeline.shader);

    VkDescriptorImageInfo si{VK_NULL_HANDLE, occluder_texture.image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    if (occluder_sampler) { si.sampler = occluder_sampler; }
    VkDescriptorImageInfo sti{VK_NULL_HANDLE, shadow_texture.image_view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet writes[2]{
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &si, nullptr, nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,       &sti, nullptr, nullptr},
    };
    vkCmdPushDescriptorSet(cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, radial_pipeline.pipeline_layout, 0, 2, writes);

    radial_push_t pc{(std::uint32_t)angle_resolution, (std::uint32_t)radial_samples};
    vkCmdPushConstants(cmd(), radial_pipeline.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd(), (angle_resolution + 255) / 256, 1, 1);
  }

  void render_light(const light_t& light) {
    fan::vec2 ws = loco_ptr->window.get_size();
    fan::vec2 center = fan::graphics::world_to_screen(light.position, *light.render_view);
    fan::vec2 edge = fan::graphics::world_to_screen(light.position + fan::vec2(light.radius, 0), *light.render_view);
    f32_t r = std::max(1.f, std::abs(edge.x - center.x));
    fan::vec2 p0 = center - r, p1 = center + r;

    bind_and_blend(light_pipeline, VK_TRUE, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA);

    push_sampler_descriptor(light_pipeline.m_layout, shadow_texture.image_view, shadow_sampler);

    light_push_t pc{{p0.x / ws.x * 2.f - 1.f, p0.y / ws.y * 2.f - 1.f},
      {p1.x / ws.x * 2.f - 1.f, p1.y / ws.y * 2.f - 1.f},
      light.color, light.softness, light.falloff_power,
      1.f / f32_t(angle_resolution), light.angle, light.cone_inner, light.cone_outer};
    vkCmdPushConstants(cmd(), light_pipeline.m_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
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

  VkDescriptorSetLayout occluder_dsl = VK_NULL_HANDLE;
  VkDescriptorSetLayout radial_dsl = VK_NULL_HANDLE;
  VkDescriptorSetLayout light_dsl = VK_NULL_HANDLE;
  VkDescriptorSetLayout solid_dsl = VK_NULL_HANDLE;

  bool resources_open = false;

  struct occluder_push_t { fan::vec2 c0; fan::vec2 c1; fan::vec2 c2; fan::vec2 c3; fan::vec2 uv_min; fan::vec2 uv_max; f32_t alpha_threshold; };
  struct radial_push_t { std::uint32_t angle_resolution; std::uint32_t radial_samples; };
  struct tile_shadow_push_t { fan::vec2 light_pos; float light_radius; uint32_t occluder_count; uint32_t angle_resolution; };
  std::vector<fan::vulkan::context_t::buffer_t> tile_occluder_buffers;
  fan::graphics::shader_t tile_shadow_shader;
  fan::vulkan::context_t::compute_pipeline_t tile_shadow_pipeline;
  VkDescriptorSetLayout tile_shadow_dsl = VK_NULL_HANDLE;

  void open_tile_shadows(std::uint32_t reserve_count = 2048) {
    auto& ctx = loco_ptr->context.vk;
    tile_occluder_buffers.resize(fan::vulkan::max_frames_in_flight);
    for (auto& buf : tile_occluder_buffers) {
      ctx.create_buffer(reserve_count * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buf);
      ctx.map_buffer(buf, &buf.mapped);
    }

    tile_shadow_shader = loco_ptr->shader_create();
    loco_ptr->shader_set_compute(tile_shadow_shader, "shaders/vulkan/2D/effects/tile_shadow.comp",
      fan::graphics::read_shader("shaders/vulkan/2D/effects/tile_shadow.comp"));
    loco_ptr->shader_compile(tile_shadow_shader);

    tile_shadow_dsl = make_dsl({
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT},
    });
    tile_shadow_pipeline.open(ctx, {.descriptor_layouts = {tile_shadow_dsl}, .shader = tile_shadow_shader, .push_constants_size = sizeof(tile_shadow_push_t)});
  }

  void build_tile_shadow_map(std::span<const fan::vec4> occluders, fan::vec2 light_pos, f32_t light_radius) {
    if (occluders.empty()) return;
    auto& ctx = loco_ptr->context.vk;
    auto frame = ctx.current_frame;
    auto& buf = tile_occluder_buffers[frame];
    std::uint64_t bytes = occluders.size() * sizeof(fan::vec4);
    if (bytes > buf.size) {
      ctx.destroy_buffer(buf);
      ctx.create_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buf);
      ctx.map_buffer(buf, &buf.mapped);
    }
    if (tile_data_dirty) {
      for (auto& b : tile_occluder_buffers) {
        if (b.mapped) { std::memcpy(b.mapped, occluders.data(), bytes); }
      }
      tile_data_dirty = false;
    }

    barrier(shadow_texture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_ACCESS_2_SHADER_READ_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);

    VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
    fan_vkCmdBindShadersEXT(cmd(), 1, &stage, &tile_shadow_pipeline.shader);

    VkDescriptorBufferInfo buf_info{buf, 0, bytes};
    VkDescriptorImageInfo img_info{VK_NULL_HANDLE, shadow_texture.image_view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet writes[2]{
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buf_info, nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  &img_info, nullptr, nullptr},
    };
    vkCmdPushDescriptorSet(cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, tile_shadow_pipeline.pipeline_layout, 0, 2, writes);

    tile_shadow_push_t pc{light_pos, light_radius, (uint32_t)occluders.size(), (uint32_t)angle_resolution};
    vkCmdPushConstants(cmd(), tile_shadow_pipeline.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd(), (angle_resolution + 255) / 256, 1, 1);
  }
  struct light_push_t { fan::vec2 ndc_min; fan::vec2 ndc_max; fan::color light_color; float softness; float falloff_power; float angle_texel; float cone_angle; float cone_inner; float cone_outer; };
  struct solid_push_t { fan::color color; };
};

alpha_shadow_renderer_t alpha_shadow_renderer;