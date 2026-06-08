struct blur_t {
  loco_t& get_loco() { return *gloco(); }
  #define loco get_loco()

  enum class blur_mode_e {
    bloom,
    raw
  };

  void open(const fan::vec2& resolution, uint32_t mip_count) {
    auto& gl = loco.context.gl;
    brightness_fbo.open(gl);
    brightness_fbo.bind(gl);

    shader_downsample = loco.shader_create();

    loco.shader_set_vertex(shader_downsample, fan::shader_paths::gl::downsample_vs, fan::graphics::read_shader(fan::shader_paths::gl::downsample_vs));
    loco.shader_set_fragment(shader_downsample, fan::shader_paths::gl::downsample_fs, fan::graphics::read_shader(fan::shader_paths::gl::downsample_fs));
    loco.shader_compile(shader_downsample);
    loco.shader_set_value(shader_downsample, "_t00", 0);

    shader_upsample = loco.shader_create();
    loco.shader_set_vertex(shader_upsample, fan::shader_paths::gl::downsample_vs, fan::graphics::read_shader(fan::shader_paths::gl::downsample_vs)); /*reuse downsample_vs*/
    loco.shader_set_fragment(shader_upsample, fan::shader_paths::gl::upsample_fs, fan::graphics::read_shader(fan::shader_paths::gl::upsample_fs));
    loco.shader_compile(shader_upsample);
    loco.shader_set_value(shader_upsample, "_t00", 0);

    fan::vec2 mip_size = resolution;
    fan::vec2i mip_int_size = resolution;

    fan::graphics::image_load_properties_t lp;
    lp.internal_format = fan::graphics::image_format_e::r11f_g11f_b10f;
    lp.format = fan::graphics::image_format_e::rgb_unorm;
    lp.type = fan::graphics::fan_float;
    lp.min_filter = fan::graphics::image_filter_e::linear;
    lp.mag_filter = fan::graphics::image_filter_e::linear;
    lp.visual_output = fan::graphics::image_sampler_address_mode_e::clamp_to_edge;

    fan::image::info_t ii;
    ii.data = nullptr;
    ii.channels = 3;

    for (uint32_t i = 0; i < mip_count; i++) {
      mip_size *= 0.5f;
      mip_int_size /= 2;
      ii.size = mip_size;
      mips.push_back({mip_size, mip_int_size, loco.image_load(ii, lp)});
    }

    brightness_fbo.bind_to_texture(gl, loco.image_get_handle(mips[0].image), GL_COLOR_ATTACHMENT0);
    if (!brightness_fbo.ready(gl)) fan::throw_error_impl();
    brightness_fbo.unbind(gl);
  }

  void close() {
    loco.shader_erase(shader_upsample);
    loco.shader_erase(shader_downsample);
    brightness_fbo.close(loco.context.gl);
  }

  inline static unsigned int quadVAO = 0;
  inline static unsigned int quadVBO;
  static void render_quad() {
    if (quadVAO == 0) {
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

  void draw(fan::graphics::image_t* color_texture, f32_t filter_radius, blur_mode_e mode) {
    auto& context = loco.context.gl;

    fan_opengl_call(glDisable(GL_DEPTH_TEST));
    fan_opengl_call(glDisable(GL_CULL_FACE));
    fan_opengl_call(glDisable(GL_BLEND));

    brightness_fbo.bind(context);

    loco.shader_use(shader_downsample);

    fan::vec2 window_size = gloco()->window.get_size();
    loco.shader_set_value(shader_downsample, "resolution", window_size);
    loco.shader_set_value(shader_downsample, "mipLevel", 0);
    loco.shader_set_value(shader_downsample, "_t00", 0);
    loco.shader_set_value(shader_downsample, "threshold", threshold);
    loco.shader_set_value(shader_downsample, "knee", knee);
    loco.shader_set_value(shader_downsample, "blur_mode", (int)mode);

    fan_opengl_call(glActiveTexture(GL_TEXTURE0));
    loco.image_bind(*color_texture);

    for (uint32_t i = 0; i < mips.size(); i++) {
      mip_t& mip = mips[i];
      fan_opengl_call(glViewport(0, 0, mip.int_size.x, mip.int_size.y));
      brightness_fbo.bind_to_texture(context, loco.image_get_handle(mip.image), GL_COLOR_ATTACHMENT0);
      render_quad();
      loco.shader_set_value(shader_downsample, "resolution", mip.size);
      loco.image_bind(mip.image);
      if (i == 0) loco.shader_set_value(shader_downsample, "mipLevel", 1);
    }

    loco.shader_use(shader_upsample);
    loco.shader_set_value(shader_upsample, "_t00", 0);

    if (mode == blur_mode_e::bloom) {
      fan_opengl_call(glEnable(GL_BLEND));
      fan_opengl_call(glBlendFunc(GL_ONE, GL_ONE));
      fan_opengl_call(glBlendEquation(GL_FUNC_ADD));
    }

    for (int i = (int)mips.size() - 1; i > 0; i--) {
      mip_t& mip = mips[i];
      mip_t& next_mip = mips[i - 1];

      fan_opengl_call(glActiveTexture(GL_TEXTURE0));
      loco.image_bind(mip.image);
      fan_opengl_call(glViewport(0, 0, next_mip.int_size.x, next_mip.int_size.y));
      brightness_fbo.bind_to_texture(context, loco.image_get_handle(next_mip.image), GL_COLOR_ATTACHMENT0);

      fan::vec2 texel_size = fan::vec2(1.0f / mip.size.x, 1.0f / mip.size.y) * filter_radius;
      loco.shader_set_value(shader_upsample, "filter_radius", texel_size * 3.f);

      render_quad();
    }

    if (mode == blur_mode_e::bloom) {
      fan_opengl_call(glDisable(GL_BLEND));
    }
    brightness_fbo.unbind(context);
    fan_opengl_call(glViewport(0, 0, window_size.x, window_size.y));
  }

  void draw(fan::graphics::image_t* color_texture) {
    draw(color_texture, bloom_filter_radius, blur_mode_e::bloom);
  }
  void draw_raw(fan::graphics::image_t* color_texture, f32_t filter_radius) {
    draw(color_texture, filter_radius, blur_mode_e::raw);
  }

  fan::opengl::core::framebuffer_t brightness_fbo;

  struct mip_t {
    fan::vec2 size;
    fan::vec2i int_size;
    fan::graphics::image_t image;
  };

  std::vector<mip_t> mips;
  f32_t bloom_filter_radius = 0.1f;
  f32_t threshold = 0.0f;
  f32_t knee = 0.1f;
  fan::graphics::shader_t shader_downsample;
  fan::graphics::shader_t shader_upsample;
};
#undef loco