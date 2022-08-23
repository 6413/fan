struct post_process_t {

  loco_t* get_loco() {
    loco_t* loco = OFFSETLESS(this, loco_t, sb_post_process_var_name);
    return loco;
  }

  bool open(const fan::opengl::core::renderbuffer_t::properties_t& p) {
    auto loco = get_loco();

    for (uint32_t i = 0; i < std::size(textures); i++) {

      textures[i].framebuffer.open(loco->get_context());
      textures[i].framebuffer.bind(loco->get_context());

      fan::webp::image_info_t image_info;
      image_info.data = 0;
      image_info.size = p.size;
      fan::opengl::image_t::load_properties_t lp;
      lp.format = fan::opengl::GL_RGBA;
      lp.internal_format = fan::opengl::GL_RGBA;
      lp.filter = fan::opengl::GL_LINEAR;
      textures[i].texture_colorbuffer.load(loco->get_context(), image_info, lp);

      textures[i].framebuffer.bind_to_texture(loco->get_context(), *textures[i].texture_colorbuffer.get_texture(loco->get_context()));

      textures[i].renderbuffer.open(loco->get_context(), p);
      textures[i].framebuffer.bind_to_renderbuffer(loco->get_context(), textures[i].renderbuffer.renderbuffer);

      while (!textures[i].framebuffer.ready(loco->get_context())) {

      }
      textures[i].framebuffer.unbind(loco->get_context());
    }

    sprite.open();

    return 0;
  }
  void close() {
    auto loco = get_loco();

    for (uint32_t i = 0; i < std::size(textures); i++) {
      textures[i].texture_colorbuffer.unload(loco->get_context());
      textures[i].framebuffer.close(loco->get_context());
      textures[i].renderbuffer.close(loco->get_context());
    }
  }

  void push(fan::opengl::viewport_t* viewport, fan::opengl::matrices_t* matrices) {

    for (uint32_t i = 0; i < 1; i++) {
      post_sprite_t::properties_t sp;
      sp.viewport = viewport;
      sp.matrices = matrices;
      sp.position = 0;
      sp.image = &textures[i].texture_colorbuffer;
      sp.size = 10;
      sprite.push_back(&textures[i].cid, sp);
    }
  }

  //void update_renderbuffer( const fan::opengl::core::renderbuffer_t::properties_t& p) {
  //  auto loco = get_loco();
  //  renderbuffer.set_storage(loco->get_context(), p);
  //}

  void start_capture(uint32_t i) {
    auto loco = get_loco();

    textures[i].framebuffer.bind(loco->get_context());
    loco->get_context()->opengl.call(loco->get_context()->opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
    loco->get_context()->set_depth_test(true);
  }
  void end_capture(uint32_t i) {
    auto loco = get_loco();
    textures[i].framebuffer.unbind(loco->get_context());
  }

  void draw() {
    auto loco = get_loco();

    sprite.m_shader.use(loco->get_context());
    sprite.m_shader.set_int(loco->get_context(), "_t01", 1);
    loco->get_context()->opengl.call(loco->get_context()->opengl.glActiveTexture, fan::opengl::GL_TEXTURE1);
    loco->get_context()->opengl.call(loco->get_context()->opengl.glBindTexture, fan::opengl::GL_TEXTURE_2D, *textures[1].texture_colorbuffer.get_texture(loco->get_context()));
    end_capture(0);
    loco->get_context()->opengl.call(loco->get_context()->opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT);
    loco->get_context()->set_depth_test(false);

    start_capture(1);

    sprite.draw();

    end_capture(1);

    sprite.m_shader.use(loco->get_context());
    sprite.m_shader.set_int(loco->get_context(), "_t01", 1);
    loco->get_context()->opengl.call(loco->get_context()->opengl.glActiveTexture, fan::opengl::GL_TEXTURE1);
    loco->get_context()->opengl.call(loco->get_context()->opengl.glBindTexture, fan::opengl::GL_TEXTURE_2D, *textures[1].texture_colorbuffer.get_texture(loco->get_context()));

    loco->get_context()->opengl.call(loco->get_context()->opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT);
    loco->get_context()->set_depth_test(false);
    sprite.draw();
  }

  // normal, blurred
  struct {
    fan::opengl::cid_t cid;
    fan::opengl::core::renderbuffer_t renderbuffer;
    fan::opengl::core::framebuffer_t framebuffer;
    fan::opengl::image_t texture_colorbuffer;
  }textures[2];

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