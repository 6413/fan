struct bloom_t {

  void open() {

    auto& context = gloco->get_context();

    shader_bloom = context.shader_create();

    context.shader_set_vertex(
      shader_bloom,
      context.read_shader("shaders/opengl/2D/effects/downsample.vs")
    );

    context.shader_set_fragment(
      shader_bloom,
      context.read_shader("shaders/opengl/2D/effects/bloom.fs")
    );

    context.shader_compile(shader_bloom);
  }

  void draw() {
    auto& context = gloco->get_context();

    context.shader_set_value(shader_bloom, "_t00", 0);
    context.shader_set_value(shader_bloom, "_t01", 1);
    context.shader_set_value(shader_bloom, "bloom", bloomamount);
    /*
    
        gloco->get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
    mips[0].image.bind_texture();
    color_texture->bind();
    fan::opengl::core::framebuffer_t::bind_to_texture(
      gloco->get_context(),
      gloco->color_buffers[0].get_texture(),
      fan::opengl::GL_COLOR_ATTACHMENT0
    );
    renderQuad();
    */

   /* fan::opengl::core::framebuffer_t::bind_to_texture(
      gloco->get_context(),
      gloco->color_buffers[0].get_texture(),
      fan::opengl::GL_COLOR_ATTACHMENT0
    );*/
    //gloco->m_framebuffer.bind(gloco->get_context());
    //gloco->get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
    //gloco->color_buffers[0].bind_texture();
    //
    //gloco->get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE1);
    //gloco->blur.mips.front().image.bind_texture();
    //
    //gloco->blur.renderQuad();
    //gloco->m_framebuffer.unbind(gloco->get_context());
  }

  f32_t bloomamount = 0.04f;
  loco_t::shader_t shader_bloom;
};