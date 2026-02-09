struct blur_t {
  loco_t& get_loco() {
    return (*OFFSETLESS(this, loco_t, gl.blur));
  }
  #define loco get_loco()

  void open(const fan::vec2& resolution, uint32_t mip_count) {

    brightness_fbo.open(loco.context.gl);
    brightness_fbo.bind(loco.context.gl);

    fan::opengl::core::renderbuffer_t::properties_t rp;
    rp.size = gloco()->window.get_size();
    rp.internalformat = GL_DEPTH_COMPONENT;
    rbo.open(loco.context.gl);
    rbo.set_storage(loco.context.gl, rp);
    rp.internalformat = GL_DEPTH_ATTACHMENT;
    rbo.bind_to_renderbuffer(loco.context.gl, rp);

    shader_downsample = loco.shader_create();

    static constexpr const char* vert_path = "shaders/opengl/2D/effects/downsample.vs";
    static constexpr const char* downsmpl_frag_path = "shaders/opengl/2D/effects/downsample.fs";
    loco.shader_set_vertex(
      shader_downsample,
      fan::graphics::read_shader(vert_path)
    );

    loco.shader_set_fragment(
      shader_downsample,
      fan::graphics::read_shader(downsmpl_frag_path)
    );
    loco.shader_set_paths(shader_downsample, vert_path, downsmpl_frag_path);

    loco.shader_compile(shader_downsample);

    loco.shader_set_value(shader_downsample, "_t00", 0);

    //
    shader_upsample = loco.shader_create();
    static constexpr const char* upsmpl_frag_path = "shaders/opengl/2D/effects/upsample.fs";
    loco.shader_set_paths(shader_upsample, vert_path, upsmpl_frag_path);

    loco.shader_set_vertex(
      shader_upsample,
      fan::graphics::read_shader(vert_path)
    );

    loco.shader_set_fragment(
      shader_upsample,
      fan::graphics::read_shader(upsmpl_frag_path)
    );

    loco.shader_compile(shader_upsample);

    loco.shader_set_value(shader_upsample, "_t00", 0);

    fan::vec2 mip_size = resolution;
    fan::vec2i mip_int_size = resolution;

    for (uint32_t i = 0; i < mip_count; i++) {
      mip_t mip;

      mip_size *= 0.5;
      mip_int_size /= 2;
      mip.size = mip_size;
      mip.int_size = mip_int_size;

      fan::graphics::image_load_properties_t lp;
      lp.internal_format = fan::graphics::image_format_e::r11f_g11f_b10f;
      lp.format = fan::graphics::image_format_e::rgb_unorm;
      lp.type = fan::graphics::fan_float;
      lp.min_filter = fan::graphics::image_filter_e::linear;
      lp.mag_filter = fan::graphics::image_filter_e::linear;
      lp.visual_output = fan::graphics::image_sampler_address_mode_e::clamp_to_edge;
      fan::image::info_t ii;
      ii.data = nullptr;
      ii.size = mip_size;
      ii.channels = 3;
      mip.image = loco.image_load(ii, lp);
      mips.push_back(mip);
    }

    if (!brightness_fbo.ready(loco.context.gl)) {
      fan::throw_error_impl();
    }

    brightness_fbo.unbind(loco.context.gl);
  }

  void close() {

    loco.shader_erase(shader_upsample);
    loco.shader_erase(shader_downsample);

    brightness_fbo.close(loco.context.gl);
    rbo.close(loco.context.gl);
  }

  inline static unsigned int quadVAO = 0;
  inline static unsigned int quadVBO;
  static void render_quad()
  {
    if (quadVAO == 0)
    {
      float quadVertices[] = {
        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
         1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
      };
      fan_opengl_call(glGenVertexArrays(1, &quadVAO));
      fan_opengl_call(glGenBuffers(1, &quadVBO));
      fan_opengl_call(glBindVertexArray(quadVAO));
      fan_opengl_call(glBindBuffer(GL_ARRAY_BUFFER, quadVBO));
      fan_opengl_call(glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW));
      fan_opengl_call(glEnableVertexAttribArray(0));
      fan_opengl_call(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0));
      fan_opengl_call(glEnableVertexAttribArray(1));
      fan_opengl_call(glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float))));
    }
    fan_opengl_call(glBindVertexArray(quadVAO));
    fan_opengl_call(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
    fan_opengl_call(glBindVertexArray(0));
  }

  void draw(fan::graphics::image_t* color_texture, f32_t filter_radius) {
    auto& context = loco.context.gl;

    fan_opengl_call(glDisable(GL_BLEND));

    brightness_fbo.bind(context);

    fan_opengl_call(glClearColor(0.0f, 0.0f, 0.0f, loco.clear_color.a));
    fan_opengl_call(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

    loco.shader_use(shader_downsample);

    fan::vec2 window_size = gloco()->window.get_size();
    loco.shader_set_value(shader_downsample, "resolution", window_size);
    loco.shader_set_value(shader_downsample, "mipLevel", 0);
    loco.shader_set_value(shader_downsample, "_t00", 0);

    fan_opengl_call(glActiveTexture(GL_TEXTURE0));
    loco.image_bind(*color_texture);

    for (uint32_t i = 0; i < mips.size(); i++) {
      mip_t& mip = mips[i];

      fan_opengl_call(glViewport(0, 0, mip.int_size.x, mip.int_size.y));
      brightness_fbo.bind_to_texture(
        context,
        loco.image_get_handle(mip.image),
        GL_COLOR_ATTACHMENT0
      );

      render_quad();

      loco.shader_set_value(shader_downsample, "resolution", mip.size);
      loco.image_bind(mip.image);

      if (i == 0) {
        loco.shader_set_value(shader_downsample, "mipLevel", 1);
      }
    }

    loco.shader_use(shader_upsample);
    loco.shader_set_value(shader_upsample, "filter_radius", filter_radius);
    loco.shader_set_value(shader_upsample, "_t00", 0);

    fan_opengl_call(glEnable(GL_BLEND));
    fan_opengl_call(glBlendFunc(GL_ONE, GL_ONE));
    fan_opengl_call(glBlendEquation(GL_FUNC_ADD));

    for (int i = (int)mips.size() - 1; i > 0; i--) {
      mip_t& mip = mips[i];
      mip_t& next_mip = mips[i - 1];

      fan_opengl_call(glActiveTexture(GL_TEXTURE0));
      loco.image_bind(mip.image);

      fan_opengl_call(glViewport(0, 0, next_mip.int_size.x, next_mip.int_size.y));

      brightness_fbo.bind_to_texture(
        context,
        loco.image_get_handle(next_mip.image),
        GL_COLOR_ATTACHMENT0
      );

      render_quad();
    }

    fan_opengl_call(glDisable(GL_BLEND));

    brightness_fbo.unbind(context);

    fan::vec2 window_size_restore = gloco()->window.get_size();
    fan_opengl_call(glViewport(0, 0, window_size_restore.x, window_size_restore.y));
  }

  void draw(fan::graphics::image_t* color_texture) {
    draw(color_texture, bloom_filter_radius);
  }


  fan::opengl::core::framebuffer_t brightness_fbo;
  fan::opengl::core::renderbuffer_t rbo;

  struct mip_t {
    fan::vec2 size;
    fan::vec2i int_size;
    fan::graphics::image_t image;
  };

  std::vector<mip_t> mips;

  f32_t bloom_filter_radius = 0.0005f;
  fan::graphics::shader_t shader_downsample;
  fan::graphics::shader_t shader_upsample;
};
#undef loco