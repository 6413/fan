struct blur_t {

  void open(const fan::vec2& resolution, uint32_t mip_count) {

    brightness_fbo.open(gloco->get_context());
    brightness_fbo.bind(gloco->get_context());

    fan::opengl::core::renderbuffer_t::properties_t rp;
    rp.size = gloco->window.get_size();
    rp.internalformat = fan::opengl::GL_DEPTH_COMPONENT;
    rbo.open(gloco->get_context());
    rbo.set_storage(gloco->get_context(), rp);
    rp.internalformat = fan::opengl::GL_DEPTH_ATTACHMENT;
    rbo.bind_to_renderbuffer(gloco->get_context(), rp);

    auto& context = gloco->get_context();
    shader_downsample = context.shader_create();

    context.shader_set_vertex(
      shader_downsample,
      context.read_shader("shaders/opengl/2D/effects/downsample.vs")
    );

    context.shader_set_fragment(
      shader_downsample,
      context.read_shader("shaders/opengl/2D/effects/downsample.fs")
    );

    context.shader_compile(shader_downsample);

    context.shader_set_value(shader_downsample, "_t00", 0);

    //
    shader_upsample = context.shader_create();

    context.shader_set_vertex(
      shader_upsample,
      context.read_shader("shaders/opengl/2D/effects/downsample.vs")
    );

    context.shader_set_fragment(
      shader_upsample,
      context.read_shader("shaders/opengl/2D/effects/upsample.fs")
    );

    context.shader_compile(shader_upsample);

    context.shader_set_value(shader_upsample, "_t00", 0);

    fan::vec2 mip_size = resolution;
    fan::vec2i mip_int_size = resolution;

    for (uint32_t i = 0; i < mip_count; i++) {
      mip_t mip;

      mip_size *= 0.5;
      mip_int_size /= 2;
      mip.size = mip_size;
      mip.int_size = mip_int_size;

      fan::opengl::context_t::image_load_properties_t lp;
      lp.internal_format = fan::opengl::GL_R11F_G11F_B10F;
      lp.format = fan::opengl::GL_RGB;
      lp.type = fan::opengl::GL_FLOAT;
      lp.min_filter = fan::opengl::GL_LINEAR;
      lp.mag_filter = fan::opengl::GL_LINEAR;
      lp.visual_output = fan::opengl::GL_CLAMP_TO_EDGE;
      fan::image::image_info_t ii;
      ii.data = nullptr;
      ii.size = mip_size;
      mip.image = context.image_load(ii, lp);
      mips.push_back(mip);
    }

    if (!brightness_fbo.ready(gloco->get_context())) {
      fan::throw_error_impl();
    }

    brightness_fbo.unbind(gloco->get_context());
  }

  inline static unsigned int quadVAO = 0;
  inline static unsigned int quadVBO;
  static void renderQuad()
  {
    auto& context = gloco->get_context();
    if (quadVAO == 0)
    {
      float quadVertices[] = {
        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
         1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
      };
      context.opengl.glGenVertexArrays(1, &quadVAO);
      context.opengl.glGenBuffers(1, &quadVBO);
      context.opengl.glBindVertexArray(quadVAO);
      context.opengl.glBindBuffer(fan::opengl::GL_ARRAY_BUFFER, quadVBO);
      context.opengl.glBufferData(fan::opengl::GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, fan::opengl::GL_STATIC_DRAW);
      context.opengl.glEnableVertexAttribArray(0);
      context.opengl.glVertexAttribPointer(0, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, 5 * sizeof(float), (void*)0);
      context.opengl.glEnableVertexAttribArray(1);
      context.opengl.glVertexAttribPointer(1, 2, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }
    context.opengl.glBindVertexArray(quadVAO);
    context.opengl.glDrawArrays(fan::opengl::GL_TRIANGLE_STRIP, 0, 4);
    context.opengl.glBindVertexArray(0);
  }

  void draw_downsamples(loco_t::image_t* image) {
    auto& context = gloco->get_context();
    context.set_depth_test(false);
    context.opengl.call(context.opengl.glDisable, fan::opengl::GL_BLEND);
    context.opengl.call(context.opengl.glBlendFunc, fan::opengl::GL_ONE, fan::opengl::GL_ONE);

    fan::vec2 window_size = gloco->window.get_size();

    context.shader_set_value(shader_downsample, "resolution", window_size);
    context.shader_set_value(shader_downsample, "mipLevel", 0);

    context.opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
    context.image_bind(*image);

    for (uint32_t i = 0; i < mips.size(); i++) {
      mip_t mip = mips[i];
#if fan_debug >= 3
      if ((int)mip.size.x < 1 || (int)mip.size.y < 1) {
        fan::throw_error("mip size too small (less than 1)");
      }
#endif
      gloco->get_context().opengl.glViewport(0, 0, mip.size.x, mip.size.y);
      brightness_fbo.bind_to_texture(
        gloco->get_context(),
        context.image_get(mip.image),
        fan::opengl::GL_COLOR_ATTACHMENT0
      );

      renderQuad();

      context.shader_set_value(shader_downsample, "resolution", mip.size);

      context.image_bind(mip.image);
      if (i == 0) {
        context.shader_set_value(shader_downsample, "mipLevel", 1);
      }
    }
  }

  void draw_upsamples(f32_t filter_radius) {
    auto& context = gloco->get_context();

    context.shader_set_value(shader_upsample, "filter_radius", filter_radius);

    context.opengl.call(context.opengl.glEnable, fan::opengl::GL_BLEND);
    context.opengl.call(context.opengl.glBlendFunc, fan::opengl::GL_ONE, fan::opengl::GL_ONE);
    context.opengl.call(context.opengl.glBlendEquation, fan::opengl::GL_FUNC_ADD);

    for (int i = (int)mips.size() - 1; i > 0; i--)
    {
      mip_t mip = mips[i];
      mip_t next_mip = mips[i - 1];

      context.opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
      context.image_bind(mip.image);

      gloco->get_context().opengl.glViewport(0, 0, next_mip.size.x, next_mip.size.y);

      fan::opengl::core::framebuffer_t::bind_to_texture(
        gloco->get_context(),
        context.image_get(next_mip.image),
        fan::opengl::GL_COLOR_ATTACHMENT0
      );
      renderQuad();
    }

    gloco->get_context().opengl.call(gloco->get_context().opengl.glDisable, fan::opengl::GL_BLEND);
  }

  void draw(loco_t::image_t* color_texture, f32_t filter_radius) {
    auto& context = gloco->get_context();

    brightness_fbo.bind(context);

    context.opengl.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    context.opengl.call(context.opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);

    draw_downsamples(color_texture);
    draw_upsamples(filter_radius);
    brightness_fbo.unbind(context);


    fan::vec2 window_size = gloco->window.get_size();
    context.opengl.glViewport(0, 0, window_size.x, window_size.y);
  }

  void draw(loco_t::image_t* color_texture) {
    draw(color_texture, bloom_filter_radius);
  }


  fan::opengl::core::framebuffer_t brightness_fbo;
  fan::opengl::core::renderbuffer_t rbo;

  struct mip_t {
    fan::vec2 size;
    fan::vec2i int_size;
    loco_t::image_t image;
  };

  std::vector<mip_t> mips;

  f32_t bloom_filter_radius = 0.005f;
  loco_t::shader_t shader_downsample;
  loco_t::shader_t shader_upsample;
};