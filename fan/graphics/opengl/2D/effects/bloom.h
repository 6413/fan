struct bloom_t {

  void open() {
    shader_bloom.open();
    shader_bloom.set_vertex(
      loco_t::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/effects/downsample.vs))
    );
    shader_bloom.set_fragment(
      loco_t::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/effects/bloom.fs))
    );
    shader_bloom.compile();
  }

  void draw() {
    shader_bloom.use();
    shader_bloom.set_int("_t00", 0);
    shader_bloom.set_int("_t01", 1);
    shader_bloom.set_float("bloom", bloomamount);

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

  f32_t bloomamount = 0.04;
  loco_t::shader_t shader_bloom;
};