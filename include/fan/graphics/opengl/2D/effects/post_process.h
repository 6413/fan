struct post_process_t {
  bool open(loco_t* loco, const fan::opengl::core::renderbuffer_t::properties_t& p) {
    framebuffer.open(loco->get_context());
    framebuffer.bind(loco->get_context());

    fan::webp::image_info_t image_info;
    image_info.data = 0;
    image_info.size = p.size;
    fan::opengl::image_t::load_properties_t lp;
    lp.format = fan::opengl::GL_RGB;
    lp.internal_format = fan::opengl::GL_RGB;
    lp.filter = fan::opengl::GL_LINEAR;
    texture_colorbuffer.load(loco->get_context(), image_info);

    framebuffer.bind_to_texture(loco->get_context(), *texture_colorbuffer.get_texture(loco->get_context()));

    renderbuffer.open(loco->get_context(), p);
    framebuffer.bind_to_renderbuffer(loco->get_context(), renderbuffer.renderbuffer);

    bool ret = !framebuffer.ready(loco->get_context());
    framebuffer.unbind(loco->get_context());

    sprite.open(loco);

    return ret;
  }
  void close(loco_t* loco) {
    texture_colorbuffer.unload(loco->get_context());
    framebuffer.close(loco->get_context());
    renderbuffer.close(loco->get_context());
  }

  void push(loco_t* loco, fan::opengl::viewport_t* viewport, fan::opengl::matrices_t* matrices) {
    fan::opengl::cid_t cid;
    post_sprite_t::properties_t sp;
    sp.viewport = viewport;
    sp.matrices = matrices;
    sp.position = 0;
    sp.image = &texture_colorbuffer;
    sp.size = 1;
    sprite.push_back(loco, &cid, sp);
  }

  void update_renderbuffer(loco_t* loco, const fan::opengl::core::renderbuffer_t::properties_t& p) {
    renderbuffer.set_storage(loco->get_context(), p);
  }

  void start_capture(loco_t* loco) {
    //draw_nodereference = context->enable_draw(this, [](fan::opengl::context_t* context, void* d) { 
      post_process_t* post = (post_process_t*)this;
      post->framebuffer.bind(loco->get_context());
      loco->get_context()->opengl.call(loco->get_context()->opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
      loco->get_context()->set_depth_test(true);
      // probably want to glclear here if trash comes
    // });
  }

  void draw(loco_t* loco) {
    framebuffer.unbind(loco->get_context()); // not sure if necessary
    loco->get_context()->opengl.call(loco->get_context()->opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT);
    loco->get_context()->set_depth_test(false);
    sprite.draw(loco);
  }

  fan::opengl::core::renderbuffer_t renderbuffer;
  fan::opengl::core::framebuffer_t framebuffer;

  fan::opengl::image_t texture_colorbuffer;

  uint32_t draw_nodereference;

  #include _FAN_PATH(graphics/opengl/2D/objects/sprite.h)
  post_sprite_t sprite;
};