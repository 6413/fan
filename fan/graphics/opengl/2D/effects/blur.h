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

    shader_downsample.open();
    shader_downsample.set_vertex(
      loco_t::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/effects/downsample.vs))
    );
    shader_downsample.set_fragment(
      loco_t::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/effects/downsample.fs))
    );
    shader_downsample.compile();

    shader_downsample.use();
    shader_downsample.set_int("_t00", 0);

    shader_upsample.open();
    shader_upsample.set_vertex(
      loco_t::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/effects/downsample.vs))
    );
    shader_upsample.set_fragment(
      loco_t::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/effects/upsample.fs))
    );
    shader_upsample.compile();

    shader_upsample.use();
    shader_upsample.set_int("_t00", 0);


    fan::vec2 mip_size = resolution;
    fan::vec2i mip_int_size = resolution;

    for (uint32_t i = 0; i < mip_count; i++) {
      mip_t mip;

      mip_size *= 0.5;
      mip_int_size /= 2;
      mip.size = mip_size;
      mip.int_size = mip_int_size;

      loco_t::image_t::load_properties_t lp;
      lp.internal_format = fan::opengl::GL_R11F_G11F_B10F;
      lp.format = fan::opengl::GL_RGB;
      lp.type = fan::opengl::GL_FLOAT;
      lp.min_filter = fan::opengl::GL_LINEAR;
      lp.mag_filter = fan::opengl::GL_LINEAR;
      lp.visual_output = fan::opengl::GL_CLAMP_TO_EDGE;
      fan::webp::image_info_t ii;
      ii.data = nullptr;
      ii.size = mip_size;
      mip.image.load(ii, lp);
      mips.push_back(mip);
    }
    brightness_fbo.unbind(gloco->get_context());
  }

  inline static unsigned int quadVAO = 0;
  inline static unsigned int quadVBO;
  static void renderQuad()
  {
    if (quadVAO == 0)
    {
      float quadVertices[] = {
        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
         1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
      };
      gloco->get_context().opengl.glGenVertexArrays(1, &quadVAO);
      gloco->get_context().opengl.glGenBuffers(1, &quadVBO);
      gloco->get_context().opengl.glBindVertexArray(quadVAO);
      gloco->get_context().opengl.glBindBuffer(fan::opengl::GL_ARRAY_BUFFER, quadVBO);
      gloco->get_context().opengl.glBufferData(fan::opengl::GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, fan::opengl::GL_STATIC_DRAW);
      gloco->get_context().opengl.glEnableVertexAttribArray(0);
      gloco->get_context().opengl.glVertexAttribPointer(0, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, 5 * sizeof(float), (void*)0);
      gloco->get_context().opengl.glEnableVertexAttribArray(1);
      gloco->get_context().opengl.glVertexAttribPointer(1, 2, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }
    gloco->get_context().opengl.glBindVertexArray(quadVAO);
    gloco->get_context().opengl.glDrawArrays(fan::opengl::GL_TRIANGLE_STRIP, 0, 4);
    gloco->get_context().opengl.glBindVertexArray(0);
  }

  void draw_downsamples(loco_t::image_t* image) {
    gloco->get_context().set_depth_test(false);
    gloco->get_context().opengl.call(gloco->get_context().opengl.glDisable, fan::opengl::GL_BLEND);
    gloco->get_context().opengl.call(gloco->get_context().opengl.glBlendFunc, fan::opengl::GL_ONE, fan::opengl::GL_ONE);

    fan::vec2 window_size = gloco->window.get_size();

    shader_downsample.use();
    shader_downsample.set_vec2("resolution", window_size);
    shader_downsample.set_int("mipLevel", 0);

    gloco->get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
    image->bind_texture();

    for (uint32_t i = 0; i < mips.size(); i++) {
      mip_t mip = mips[i];
      gloco->get_context().opengl.glViewport(0, 0, mip.size.x, mip.size.y);
      brightness_fbo.bind_to_texture(
        gloco->get_context(),
        mip.image.get_texture(),
        fan::opengl::GL_COLOR_ATTACHMENT0
      );

      renderQuad();

      shader_downsample.set_vec2("resolution", mip.size);

      mip.image.bind_texture();
      if (i == 0) {
        shader_downsample.set_int("mipLevel", 1);
      }
    }
  }

  void draw_upsamples(f32_t filter_radius) {
    shader_upsample.use();
    shader_upsample.set_float("filter_radius", filter_radius);

    gloco->get_context().opengl.call(gloco->get_context().opengl.glEnable, fan::opengl::GL_BLEND);
    gloco->get_context().opengl.call(gloco->get_context().opengl.glBlendFunc, fan::opengl::GL_ONE, fan::opengl::GL_ONE);
    gloco->get_context().opengl.call(gloco->get_context().opengl.glBlendEquation, fan::opengl::GL_FUNC_ADD);

    for (int i = (int)mips.size() - 1; i > 0; i--)
    {
      mip_t mip = mips[i];
      mip_t next_mip = mips[i - 1];

      gloco->get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
      mip.image.bind_texture();

      gloco->get_context().opengl.glViewport(0, 0, next_mip.size.x, next_mip.size.y);

      fan::opengl::core::framebuffer_t::bind_to_texture(
        gloco->get_context(),
        next_mip.image.get_texture(),
        fan::opengl::GL_COLOR_ATTACHMENT0
      );
      renderQuad();
    }

    gloco->get_context().opengl.call(gloco->get_context().opengl.glDisable, fan::opengl::GL_BLEND);
  }

  void draw(loco_t::image_t* color_texture, f32_t filter_radius) {
    brightness_fbo.bind(gloco->get_context());

    gloco->get_context().opengl.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    gloco->get_context().opengl.call(gloco->get_context().opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);

    draw_downsamples(color_texture);
    draw_upsamples(filter_radius);
    brightness_fbo.unbind(gloco->get_context());


    fan::vec2 window_size = gloco->get_window()->get_size();
    gloco->get_context().opengl.glViewport(0, 0, window_size.x, window_size.y);
  }

  void draw() {
    draw(&gloco->color_buffers[0], bloom_filter_radius);
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