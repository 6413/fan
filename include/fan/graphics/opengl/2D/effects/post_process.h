struct post_process_t {

  loco_t* get_loco() {
    loco_t* loco = OFFSETLESS(this, loco_t, sb_post_process_var_name);
    return loco;
  }

  bool open(const fan::opengl::core::renderbuffer_t::properties_t& p) {
    auto loco = get_loco();

    framebuffer.open(loco->get_context());
    framebuffer.bind(loco->get_context());

    fan::webp::image_info_t image_info;
    image_info.data = 0;
    image_info.size = p.size;
    fan::opengl::image_t::load_properties_t lp;
    lp.format = fan::opengl::GL_RGBA;
    lp.internal_format = fan::opengl::GL_RGBA;
    lp.filter = fan::opengl::GL_LINEAR;
    texture_colorbuffer.load(loco->get_context(), image_info, lp);

    framebuffer.bind_to_texture(loco->get_context(), *texture_colorbuffer.get_texture(loco->get_context()));

    renderbuffer.open(loco->get_context(), p);
    framebuffer.bind_to_renderbuffer(loco->get_context(), renderbuffer.renderbuffer);

    bool ret = !framebuffer.ready(loco->get_context());
    framebuffer.unbind(loco->get_context());

    sprite.open();

    return ret;
  }
  void close() {
    auto loco = get_loco();

    texture_colorbuffer.unload(loco->get_context());
    framebuffer.close(loco->get_context());
    renderbuffer.close(loco->get_context());
  }

  void push(fan::opengl::viewport_t* viewport, fan::opengl::matrices_t* matrices) {

    fan::opengl::cid_t cid;
    post_sprite_t::properties_t sp;
    sp.viewport = viewport;
    sp.matrices = matrices;
    sp.position = 0;
    sp.image = &texture_colorbuffer;
    sp.size = 10;
    sprite.push_back(&cid, sp);
  }

  void update_renderbuffer( const fan::opengl::core::renderbuffer_t::properties_t& p) {
    auto loco = get_loco();
    renderbuffer.set_storage(loco->get_context(), p);
  }

  void start_capture() {
    auto loco = get_loco();

    post_process_t* post = (post_process_t*)this;
    post->framebuffer.bind(loco->get_context());
    loco->get_context()->opengl.call(loco->get_context()->opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
    loco->get_context()->set_depth_test(true);
  }
  void end_capture() {

  }

  void draw() {
    auto loco = get_loco();

    framebuffer.unbind(loco->get_context()); // not sure if necessary
    loco->get_context()->opengl.call(loco->get_context()->opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT);
    loco->get_context()->set_depth_test(false);
    sprite.draw();
  }

  fan::opengl::core::renderbuffer_t renderbuffer;
  fan::opengl::core::framebuffer_t framebuffer;

  fan::opengl::image_t texture_colorbuffer;

  uint32_t draw_nodereference;

  #define temp_shape_var_name sb_shape_var_name
  #define sb_shape_var_name sprite
  #define sb_get_loco \
    loco_t* get_loco() { \
      loco_t* loco = OFFSETLESS(OFFSETLESS(this, post_process_t, sb_shape_var_name), loco_t, sb_post_process_var_name); \
      return loco; \
    }
  #include _FAN_PATH(graphics/opengl/2D/objects/sprite.h)
  post_sprite_t sb_shape_var_name;
  #undef sb_offsetless
};