loco_t& get_loco() {
  return (*OFFSETLESS(this, loco_t, gl));
}
#define loco get_loco()


#if defined(FAN_2D)

template <typename T, typename T2, typename T3, typename T4>
static void modify_render_data_element_arr(fan::graphics::shapes::shape_t* shape, fan::graphics::shaper_t::ShapeRenderData_t* data, T2 T::* attribute, std::size_t j, auto T4::*arr_member, const T3& value) {
  if ((gloco()->context.gl.opengl.major > 3) || (gloco()->context.gl.opengl.major == 3 && gloco()->context.gl.opengl.minor >= 3)) {
    (((T*)data)->*attribute)[j].*arr_member = value;
    auto& data = fan::graphics::g_shapes->shaper.ShapeList[*shape];
    fan::graphics::g_shapes->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(attribute) + sizeof(std::remove_all_extents_t<T2>) * j + fan::member_offset(arr_member),
      sizeof(T3)
    );
  }
  else {
    for (int i = 0; i < 6; ++i) {
      auto& v = ((T*)data)[i];
      (((T*)&v)->*attribute)[j].*arr_member = value;
      auto& data = fan::graphics::g_shapes->shaper.ShapeList[*shape];
      fan::graphics::g_shapes->shaper.ElementIsPartiallyEdited(
        data.sti,
        data.blid,
        data.ElementIndex,
        fan::member_offset(attribute) + sizeof(std::remove_all_extents_t<T2>) * (j + i) + fan::member_offset(arr_member),
        sizeof(T3)
      );
    }
  }
}

// remove gloco
template <typename T, typename T2, typename T3>
static void modify_render_data_element(fan::graphics::shapes::shape_t* shape, fan::graphics::shaper_t::ShapeRenderData_t* data, T2 T::* attribute, const T3& value) {
  if ((gloco()->context.gl.opengl.major > 3) || (gloco()->context.gl.opengl.major == 3 && gloco()->context.gl.opengl.minor >= 3)) {
    ((T*)data)->*attribute = value;
    auto& data = fan::graphics::g_shapes->shaper.ShapeList[*shape];
    fan::graphics::g_shapes->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(attribute),
      sizeof(T3)
    );
  }
  else {
    for (int i = 0; i < 6; ++i) {
      auto& v = ((T*)data)[i];
      ((T*)&v)->*attribute = value;
      auto& data = fan::graphics::g_shapes->shaper.ShapeList[*shape];
      fan::graphics::g_shapes->shaper.ElementIsPartiallyEdited(
        data.sti,
        data.blid,
        data.ElementIndex,
        fan::member_offset(attribute) + sizeof(T) * i,
        sizeof(T3)
      );
    }
  }
}
#endif

void open() {
  if (loco.window.renderer == fan::window_t::renderer_t::opengl) {
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_FALSE);

    if (loco.window.renderer == fan::window_t::renderer_t::renderer_t::vulkan) {
      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    }
    else if (loco.window.renderer == fan::window_t::renderer_t::opengl) {
      glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    }

    GLFWwindow* dummy_window = glfwCreateWindow(640, 400, "dummy", nullptr, nullptr);
    if (dummy_window == nullptr) {
      fan::throw_error("failed to open dummy window");
    }

    glfwMakeContextCurrent(dummy_window);

    loco.context.gl.open();

    if (loco.context.gl.opengl.major == -1 || loco.context.gl.opengl.minor == -1) {
      const char* gl_version = (const char*)fan_opengl_call(glGetString(GL_VERSION));
      sscanf(gl_version, "%d.%d", &loco.context.gl.opengl.major, &loco.context.gl.opengl.minor);
    }
    glfwMakeContextCurrent(nullptr);
    glfwDestroyWindow(dummy_window);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
  }
  {
    #if 1
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, loco.context.gl.opengl.major);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, loco.context.gl.opengl.minor);
    glfwWindowHint(GLFW_SAMPLES, 0);

    if ((loco.context.gl.opengl.major > 3) || (loco.context.gl.opengl.major == 3 && loco.context.gl.opengl.minor > 2)) {
      glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    }

    if ((loco.context.gl.opengl.major > 3) || (loco.context.gl.opengl.major == 3 && loco.context.gl.opengl.minor > 0)) {
      glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, true);
    }
  #else // renderdoc debug
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_SAMPLES, 0);

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, true);
  #endif

    glfwSetErrorCallback(loco.context.gl.error_callback);
  }
}

void close() {
#if defined(LOCO_FRAMEBUFFER)
  blur.close();
  loco.shader_erase(loco.gl.m_fbo_final_shader);
#endif
}

void init_framebuffer() {
  if (!((loco.context.gl.opengl.major > 3) || (loco.context.gl.opengl.major == 3 && loco.context.gl.opengl.minor >= 3))) {
    return;
  }

#if defined(LOCO_FRAMEBUFFER)
  loco.gl.m_framebuffer.open(loco.context.gl);
  // can be GL_RGB16F
  loco.gl.m_framebuffer.bind(loco.context.gl);
#endif


#if defined(LOCO_FRAMEBUFFER)
  //
  static auto load_texture = [&](fan::image::info_t& image_info, fan::graphics::image_t& color_buffer, GLenum attachment, bool reload = false) {
    fan::graphics::image_load_properties_t load_properties;
    load_properties.visual_output = fan::graphics::image_sampler_address_mode::repeat;
    load_properties.internal_format = fan::graphics::image_format::r8b8g8a8_unorm;
    load_properties.format = fan::graphics::image_format::r8b8g8a8_unorm;
    load_properties.type = fan::graphics::fan_float;
    load_properties.min_filter = fan::graphics::image_filter::linear;
    load_properties.mag_filter = fan::graphics::image_filter::linear;
    if (reload == true) {
      loco.image_reload(color_buffer, image_info, load_properties);
    }
    else {
      color_buffer = loco.image_load(image_info, load_properties);
    }
    fan_opengl_call(glGenerateMipmap(GL_TEXTURE_2D));
    loco.image_bind(color_buffer);
    fan::opengl::core::framebuffer_t::bind_to_texture(loco.context.gl, loco.image_get_handle(color_buffer), attachment);
    };

  fan::image::info_t image_info;
  image_info.data = nullptr;
  image_info.size = loco.window.get_size();
  image_info.channels = 4;

  loco.gl.m_framebuffer.bind(loco.context.gl);
  for (uint32_t i = 0; i < (uint32_t)std::size(loco.gl.color_buffers); ++i) {
    load_texture(image_info, loco.gl.color_buffers[i], GL_COLOR_ATTACHMENT0 + i);
  }

  window_resize_handle = loco.window.add_resize_callback([&](const auto& d) {
    loco.viewport_set(loco.orthographic_render_view.viewport, fan::vec2(0, 0), d.size);
    loco.viewport_set(loco.perspective_render_view.viewport, fan::vec2(0, 0), d.size);

    loco.camera_set_ortho(
      loco.orthographic_render_view.camera,
      fan::vec2(0, d.size.x),
      fan::vec2(0, d.size.y)
    );

    fan::image::info_t image_info;
    image_info.data = nullptr;
    image_info.size = loco.window.get_size();

    loco.gl.m_framebuffer.bind(loco.context.gl);
    for (uint32_t i = 0; i < (uint32_t)std::size(loco.gl.color_buffers); ++i) {
      load_texture(image_info, loco.gl.color_buffers[i], GL_COLOR_ATTACHMENT0 + i, true);
    }

    fan::opengl::core::renderbuffer_t::properties_t renderbuffer_properties;
    loco.gl.m_framebuffer.bind(loco.context.gl);
    renderbuffer_properties.size = image_info.size;
    renderbuffer_properties.internalformat = GL_DEPTH_COMPONENT;
    loco.gl.m_rbo.set_storage(loco.context.gl, renderbuffer_properties);
  });

  fan::opengl::core::renderbuffer_t::properties_t renderbuffer_properties;
  loco.gl.m_framebuffer.bind(loco.context.gl);
  renderbuffer_properties.size = image_info.size;
  renderbuffer_properties.internalformat = GL_DEPTH_COMPONENT;
  loco.gl.m_rbo.open(loco.context.gl);
  loco.gl.m_rbo.set_storage(loco.context.gl, renderbuffer_properties);
  renderbuffer_properties.internalformat = GL_DEPTH_ATTACHMENT;
  loco.gl.m_rbo.bind_to_renderbuffer(loco.context.gl, renderbuffer_properties);

  unsigned int attachments[sizeof(loco.gl.color_buffers) / sizeof(loco.gl.color_buffers[0])];

  for (uint8_t i = 0; i < std::size(loco.gl.color_buffers); ++i) {
    attachments[i] = GL_COLOR_ATTACHMENT0 + i;
  }

  fan_opengl_call(glDrawBuffers(std::size(attachments), attachments));

  if (!loco.gl.m_framebuffer.ready(loco.context.gl)) {
    fan::throw_error("framebuffer not ready");
  }

  static constexpr uint32_t mip_count = 6;
  // always open. it goes a bit complex to make blur open and close in the middle of frame
  loco.gl.blur.open(loco.window.get_size(), mip_count);

  loco.gl.m_framebuffer.unbind(loco.context.gl);

#endif
}

#if defined(FAN_2D)

void shapes_open() {
  {
    if (loco.context.gl.opengl.major == 2 && loco.context.gl.opengl.minor == 1) {
      loco.shape_open(
        fan::graphics::shapes::sprite_t::shape_type,
        sizeof(fan::graphics::shapes::sprite_t::vi_t),
        sizeof(fan::graphics::shapes::sprite_t::ri_t),
        fan::graphics::shape_gl_init_list_t{
          .ptr=fan::graphics::g_shapes->sprite.get_locations().data(),
          .count=static_cast<int>(fan::graphics::g_shapes->sprite.get_locations().size())
        },
        "shaders/opengl/2D/objects/sprite_2_1.vs",
        "shaders/opengl/2D/objects/sprite_2_1.fs",
        6 // set instance count to 6 vertices, in opengl 2.1 there is no instancing,
          // so sending same 6 elements per shape
      );
    }
    else {
      loco.shape_open(
        fan::graphics::shapes::sprite_t::shape_type,
        sizeof(fan::graphics::shapes::sprite_t::vi_t),
        sizeof(fan::graphics::shapes::sprite_t::ri_t),
        fan::graphics::shape_gl_init_list_t{
          .ptr=fan::graphics::g_shapes->sprite.get_locations().data(),
          .count=static_cast<int>(fan::graphics::g_shapes->sprite.get_locations().size())
        },
        "shaders/opengl/2D/objects/sprite.vs",
        "shaders/opengl/2D/objects/sprite.fs"
      );
    }
  }
  {
    if (loco.context.gl.opengl.major == 2 && loco.context.gl.opengl.minor == 1) {
      // todo implement line
    }
    else {
      loco.shape_open(
        fan::graphics::shapes::line_t::shape_type,
        sizeof(fan::graphics::shapes::line_t::vi_t),
        sizeof(fan::graphics::shapes::line_t::ri_t),
        fan::graphics::shape_gl_init_list_t{
          .ptr=fan::graphics::g_shapes->line.get_locations().data(),
          .count=static_cast<int>(fan::graphics::g_shapes->line.get_locations().size())
        },
        "shaders/opengl/2D/objects/line.vs",
        "shaders/opengl/2D/objects/line.fs"
      );
    }
  }

  {
    if (loco.context.gl.opengl.major == 2 && loco.context.gl.opengl.minor == 1) {
      // todo
    }
    else {
      loco.shape_open(
        fan::graphics::shapes::rectangle_t::shape_type,
        sizeof(fan::graphics::shapes::rectangle_t::vi_t),
        sizeof(fan::graphics::shapes::rectangle_t::ri_t),
        fan::graphics::shape_gl_init_list_t{
          .ptr=fan::graphics::g_shapes->rectangle.get_locations().data(),
          .count=static_cast<int>(fan::graphics::g_shapes->rectangle.get_locations().size())
        },
        "shaders/opengl/2D/objects/rectangle.vs",
        "shaders/opengl/2D/objects/rectangle.fs"
      );
    }
  }

  {
    if (loco.context.gl.opengl.major == 2 && loco.context.gl.opengl.minor == 1) {
      // todo
    }
    else {
      loco.shape_open(
        fan::graphics::shapes::light_t::shape_type,
        sizeof(fan::graphics::shapes::light_t::vi_t),
        sizeof(fan::graphics::shapes::light_t::ri_t),
        fan::graphics::shape_gl_init_list_t{
          .ptr=fan::graphics::g_shapes->light.get_locations().data(),
          .count=static_cast<int>(fan::graphics::g_shapes->light.get_locations().size())
        },
        "shaders/opengl/2D/objects/light.vs",
        "shaders/opengl/2D/objects/light.fs"
      );
    }
  }
  {
    if (loco.context.gl.opengl.major == 2 && loco.context.gl.opengl.minor == 1) {
      // todo
    }
    else {
      loco.shape_open(
        fan::graphics::shapes::unlit_sprite_t::shape_type,
        sizeof(fan::graphics::shapes::unlit_sprite_t::vi_t),
        sizeof(fan::graphics::shapes::unlit_sprite_t::ri_t),
        fan::graphics::shape_gl_init_list_t{
          .ptr=fan::graphics::g_shapes->unlit_sprite.get_locations().data(),
          .count=static_cast<int>(fan::graphics::g_shapes->unlit_sprite.get_locations().size())
        },
        "shaders/opengl/2D/objects/sprite.vs",
        "shaders/opengl/2D/objects/unlit_sprite.fs"
      );
    }
  }

  {
    if (loco.context.gl.opengl.major == 2 && loco.context.gl.opengl.minor == 1) {

    }
    else {
      loco.shape_open(
        fan::graphics::shapes::circle_t::shape_type,
        sizeof(fan::graphics::shapes::circle_t::vi_t),
        sizeof(fan::graphics::shapes::circle_t::ri_t),
        fan::graphics::shape_gl_init_list_t{
          .ptr=fan::graphics::g_shapes->circle.get_locations().data(),
          .count=static_cast<int>(fan::graphics::g_shapes->circle.get_locations().size())
        },
        "shaders/opengl/2D/objects/circle.vs",
        "shaders/opengl/2D/objects/circle.fs"
      );
    }
  }
  {
    if (loco.context.gl.opengl.major == 2 && loco.context.gl.opengl.minor == 1) {

    }
    else {
      loco.shape_open(
        fan::graphics::shapes::capsule_t::shape_type,
        sizeof(fan::graphics::shapes::capsule_t::vi_t),
        sizeof(fan::graphics::shapes::capsule_t::ri_t),
        fan::graphics::shape_gl_init_list_t{
          .ptr=fan::graphics::g_shapes->capsule.get_locations().data(),
          .count=static_cast<int>(fan::graphics::g_shapes->capsule.get_locations().size())
        },
        "shaders/opengl/2D/objects/capsule.vs",
        "shaders/opengl/2D/objects/capsule.fs"
      );
    }
  }
  {
    if (loco.context.gl.opengl.major == 2 && loco.context.gl.opengl.minor == 1) {

    }
    else {
      loco.shape_open(
        fan::graphics::shapes::polygon_t::shape_type,
        sizeof(fan::graphics::shapes::polygon_t::vi_t),
        sizeof(fan::graphics::shapes::polygon_t::ri_t),
        fan::graphics::shape_gl_init_list_t{
          .ptr=fan::graphics::g_shapes->polygon.get_locations().data(),
          .count=static_cast<int>(fan::graphics::g_shapes->polygon.get_locations().size())
        },
        "shaders/opengl/2D/objects/polygon.vs",
        "shaders/opengl/2D/objects/polygon.fs",
        1,
        false
      );
    }
  }

  {
    if (loco.context.gl.opengl.major == 2 && loco.context.gl.opengl.minor == 1) {

    }
    else {
      loco.shape_open(
        fan::graphics::shapes::grid_t::shape_type,
        sizeof(fan::graphics::shapes::grid_t::vi_t),
        sizeof(fan::graphics::shapes::grid_t::ri_t),
        fan::graphics::shape_gl_init_list_t{
          .ptr=fan::graphics::g_shapes->grid.get_locations().data(),
          .count=static_cast<int>(fan::graphics::g_shapes->grid.get_locations().size())
        },
        "shaders/opengl/2D/objects/grid.vs",
        "shaders/opengl/2D/objects/grid.fs"
      );
    }
  }

  // vfi must be in this order
  fan::graphics::g_shapes->vfi.open();

  {
    if (loco.context.gl.opengl.major == 2 && loco.context.gl.opengl.minor == 1) {

    }
    else {
      loco.shape_open(
        fan::graphics::shapes::particles_t::shape_type,
        sizeof(fan::graphics::shapes::particles_t::vi_t),
        sizeof(fan::graphics::shapes::particles_t::ri_t),
        fan::graphics::shape_gl_init_list_t{
          .ptr=fan::graphics::g_shapes->particles.get_locations().data(),
          .count=static_cast<int>(fan::graphics::g_shapes->particles.get_locations().size())
        },
        "shaders/opengl/2D/effects/particles.vs",
        "shaders/opengl/2D/effects/particles.fs"
      );
    }
  }

  {
    if (loco.context.gl.opengl.major == 2 && loco.context.gl.opengl.minor == 1) {

    }
    else {
      loco.shape_open(
        fan::graphics::shapes::universal_image_renderer_t::shape_type,
        sizeof(fan::graphics::shapes::universal_image_renderer_t::vi_t),
        sizeof(fan::graphics::shapes::universal_image_renderer_t::ri_t),
        fan::graphics::shape_gl_init_list_t{
          .ptr=fan::graphics::g_shapes->universal_image_renderer.get_locations().data(),
          .count=static_cast<int>(fan::graphics::g_shapes->universal_image_renderer.get_locations().size())
        },
        "shaders/opengl/2D/objects/pixel_format_renderer.vs",
        "shaders/opengl/2D/objects/yuv420p.fs"
      );
    }
  }

  {
    if (loco.context.gl.opengl.major == 2 && loco.context.gl.opengl.minor == 1) {

    }
    else {
      loco.shape_open(
        fan::graphics::shapes::gradient_t::shape_type,
        sizeof(fan::graphics::shapes::gradient_t::vi_t),
        sizeof(fan::graphics::shapes::gradient_t::ri_t),
        fan::graphics::shape_gl_init_list_t{
          .ptr=fan::graphics::g_shapes->gradient.get_locations().data(),
          .count=static_cast<int>(fan::graphics::g_shapes->gradient.get_locations().size())
        },
        "shaders/opengl/2D/effects/gradient.vs",
        "shaders/opengl/2D/effects/gradient.fs"
      );
    }
  }

  {
    if (loco.context.gl.opengl.major == 2 && loco.context.gl.opengl.minor == 1) {

    }
    else {
      loco.shape_open(
        fan::graphics::shapes::shader_shape_t::shape_type,
        sizeof(fan::graphics::shapes::shader_shape_t::vi_t),
        sizeof(fan::graphics::shapes::shader_shape_t::ri_t),
        fan::graphics::shape_gl_init_list_t{
          .ptr=fan::graphics::g_shapes->shader_shape.get_locations().data(),
          .count=static_cast<int>(fan::graphics::g_shapes->shader_shape.get_locations().size())
        },
        "shaders/opengl/2D/objects/sprite.vs",
        "shaders/opengl/2D/objects/sprite.fs"
      );
    }
  }

  {
#if defined(FAN_3D)
    loco.shape_open(
      fan::graphics::shapes::rectangle3d_t::shape_type,
      sizeof(fan::graphics::shapes::rectangle3d_t::vi_t),
      sizeof(fan::graphics::shapes::rectangle3d_t::ri_t),
      fan::graphics::shape_gl_init_list_t{
        .ptr=fan::graphics::g_shapes->rectangle3d.get_locations().data(),
        .count=static_cast<int>(fan::graphics::g_shapes->rectangle3d.get_locations().size())
      },
      "shaders/opengl/3D/objects/rectangle.vs",
      "shaders/opengl/3D/objects/rectangle.fs",
      (loco.context.gl.opengl.major == 2 && loco.context.gl.opengl.minor == 1) ? 36 : 1
    );
#endif
  }
  #if defined(FAN_3D)
  {
    if (loco.context.gl.opengl.major == 2 && loco.context.gl.opengl.minor == 1) {
      // todo implement line
    }
    else {
      loco.shape_open(
        fan::graphics::shapes::line3d_t::shape_type,
        sizeof(fan::graphics::shapes::line3d_t::vi_t),
        sizeof(fan::graphics::shapes::line3d_t::ri_t),
        fan::graphics::shape_gl_init_list_t{
          .ptr=fan::graphics::g_shapes->line3d.get_locations().data(),
          .count=static_cast<int>(fan::graphics::g_shapes->line3d.get_locations().size())
        },
        "shaders/opengl/3D/objects/line.vs",
        "shaders/opengl/3D/objects/line.fs"
      );
    }
  }
  #endif

  { // shadow
    if (loco.context.gl.opengl.major == 2 && loco.context.gl.opengl.minor == 1) {
      // todo
    }
    else {
      loco.shape_open(
        fan::graphics::shapes::shadow_t::shape_type,
        sizeof(fan::graphics::shapes::shadow_t::vi_t),
        sizeof(fan::graphics::shapes::shadow_t::ri_t),
        fan::graphics::shape_gl_init_list_t{
          .ptr=fan::graphics::g_shapes->shadow.get_locations().data(),
          .count=static_cast<int>(fan::graphics::g_shapes->shadow.get_locations().size())
        },
        "shaders/opengl/2D/objects/shadow.vs",
        "shaders/opengl/2D/objects/shadow.fs"
      );
    }
  }

  fan::graphics::shader_t shader = loco.shader_create();

  loco.shader_set_vertex(shader,
    fan::graphics::read_shader("shaders/empty.vs")
  );

  loco.shader_set_fragment(shader,
    fan::graphics::read_shader("shaders/empty.fs")
  );

  loco.shader_compile(shader);

  fan::graphics::shaper_t::BlockProperties_t::gl_t st_gl;
  st_gl.locations = {};
  st_gl.shader = shader;

  fan::graphics::shaper_t::BlockProperties_t bp;
  bp.MaxElementPerBlock = (fan::graphics::shaper_t::MaxElementPerBlock_t)fan::graphics::MaxElementPerBlock,
  bp.RenderDataSize = 0,
  bp.DataSize = 0,
  bp.renderer.gl = st_gl;

  fan::graphics::g_shapes->shaper.SetShapeType(
    fan::graphics::shapes::shape_type_t::light_end,
    bp
  );
  fan::graphics::g_shapes->shape_add(
    fan::graphics::shapes::shape_type_t::light_end,
    0,
    0,
    fan::graphics::Key_e::light_end, (uint8_t)0,
    fan::graphics::Key_e::ShapeType, (fan::graphics::shaper_t::ShapeTypeIndex_t)fan::graphics::shapes::shape_type_t::light_end
  );
}

void add_shape_type(fan::graphics::shaper_t::ShapeTypes_NodeData_t& st, const fan::graphics::shaper_t::BlockProperties_t& bp) {
  auto& bpdata = bp.renderer.gl;
  auto& data = st.renderer.gl;
  data.m_vao.open(loco.context.gl);
  data.m_vbo.open(loco.context.gl, GL_ARRAY_BUFFER);
  data.m_vao.bind(loco.context.gl);
  data.m_vbo.bind(loco.context.gl);
  data.shader = bpdata.shader;
  data.locations = bpdata.locations;
  data.instanced = bpdata.instanced;
  data.vertex_count = bpdata.vertex_count;
  fan::graphics::context_shader_t shader;
  if (!data.shader.iic()) {
    shader = loco.shader_get(data.shader);
  }
  uint64_t ptr_offset = 0;
  if (data.locations.ptr == nullptr) {
    return;
  }

  fan::graphics::shape_gl_init_list_t& locations = data.locations;

  for (int i = 0; i < locations.count; ++i) {
    auto& location = locations.ptr[i];
    if ((loco.context.gl.opengl.major == 2 && loco.context.gl.opengl.minor == 1) && !data.shader.iic()) {
      location.index.first = fan_opengl_call(glGetAttribLocation(shader.gl.id, location.index.second));
    }
    fan_opengl_call(glEnableVertexAttribArray(location.index.first));
    switch (location.type) {
    case GL_UNSIGNED_INT:
    case GL_INT: {
      fan_opengl_call(glVertexAttribIPointer(location.index.first, location.size, location.type, location.stride, (void*)ptr_offset));
      break;
    }
    default: {
      fan_opengl_call(glVertexAttribPointer(location.index.first, location.size, location.type, GL_FALSE, location.stride, (void*)ptr_offset));
    }
    }
    // instancing
    if ((loco.context.gl.opengl.major > 3) || (loco.context.gl.opengl.major == 3 && loco.context.gl.opengl.minor >= 3)) {
      if (data.instanced) {
        fan_opengl_call(glVertexAttribDivisor(location.index.first, 1));
      }
    }
    uint64_t sizeof_type = 0;
    switch (location.type) {
    case GL_FLOAT: {
      sizeof_type = sizeof(GLfloat);
      break;
    }
    case GL_UNSIGNED_INT: {
      sizeof_type = sizeof(GLuint);
      break;
    }
    case GL_INT: {
      sizeof_type = sizeof(GLint);
      break;
    }
    default: {
      fan::throw_error_impl();
    }
    }
    ptr_offset += location.size * sizeof_type;
  }
}

void draw_all_shape_aabbs() {
  auto& shapes = fan::graphics::get_shapes();
  auto& shaper = shapes.shaper;

  fan::graphics::shaper_t::KeyTraverse_t kt;
  kt.Init(shaper);

  while (kt.Loop(shaper)) {

    if (!kt.isbm) {
      continue;
    }

    fan::graphics::shaper_t::BlockTraverse_t bt;
    auto sti = bt.Init(shaper, kt.bmid());

    if (sti == fan::graphics::shapes::shape_type_t::vfi || 
        sti == fan::graphics::shapes::shape_type_t::light_end ||
        sti == fan::graphics::shapes::shape_type_t::line) 
    {
      continue;
    }

    do {
      auto blid = bt.GetBlockID();
      auto count = bt.GetAmount(shaper);

      for (fan::graphics::shaper_t::MaxElementPerBlock_t i = 0; i < count; ++i) {

        auto sid = *shaper.GetShapeID(sti, blid, i);
        if (sid.iic()) {
          continue;
        }
        //fan::print(fan::graphics::shapes::shape_names[sti]);

        auto* shape = ((fan::graphics::shapes::shape_t*)&sid);

        fan::physics::aabb_t aabb = shape->get_aabb();

        fan::vec2 p0{ aabb.min.x, aabb.min.y };
        fan::vec2 p1{ aabb.max.x, aabb.min.y };
        fan::vec2 p2{ aabb.max.x, aabb.max.y };
        fan::vec2 p3{ aabb.min.x, aabb.max.y };

        auto emit = [this, shape](fan::vec2 a, fan::vec2 b) {
          fan::graphics::shapes::line_t::properties_t lp;
          lp.src = fan::vec3(a, 0xFFA0);
          lp.dst = fan::vec3(b, 0);
          lp.color = shape->get_color();
          lp.camera = shape->get_camera();
          lp.viewport = shape->get_viewport();
          //static std::vector<fan::graphics::shapes::ShapeId_t> as;
          loco.add_shape_to_immediate_draw(lp);
        };

        emit(p0, p1);
        emit(p1, p2);
        emit(p2, p3);
        emit(p3, p0);
      }

    } while (bt.Loop(shaper));
  }
}

#endif

void shapes_draw() {
#if defined(FAN_2D)
  fan::graphics::shaper_t::KeyTraverse_t KeyTraverse;
  KeyTraverse.Init(fan::graphics::g_shapes->shaper);

  uint32_t texture_count = 0;
  viewport_t viewport;
  viewport.sic();
  camera_t camera;
  camera.sic();

  fan::graphics::shaper_t::ShapeTypeIndex_t prev_st = -1;
  bool light_buffer_enabled = false;

  {
    auto& camera_perspective = loco.camera_get(loco.perspective_render_view.camera);
    camera_perspective.update_view();
    camera_perspective.m_view = camera_perspective.get_view_matrix();
  }

  uint64_t draw_mode = -1;
  uint32_t vertex_count = -1;
  bool visible = true;

  fan::graphics::shapes& shapes = loco.shapes;

  while (KeyTraverse.Loop(fan::graphics::g_shapes->shaper)) {

    fan::graphics::shaper_t::KeyTypeIndex_t kti =
      KeyTraverse.kti(fan::graphics::g_shapes->shaper);

    switch (kti) {
    case fan::graphics::Key_e::visible:
      visible = *(fan::graphics::visible_t*)KeyTraverse.kd();
      break;

    case fan::graphics::Key_e::blending: {
      uint8_t Key = *(uint8_t*)KeyTraverse.kd();
      if (Key) {
        loco.context.gl.set_depth_test(false);
        fan_opengl_call(glEnable(GL_BLEND));
        fan_opengl_call(glBlendFunc(loco.gl.blend_src_factor, loco.gl.blend_dst_factor));
      }
      else {
        fan_opengl_call(glDisable(GL_BLEND));
        loco.context.gl.set_depth_test(true);
      }
      break;
    }

    case fan::graphics::Key_e::image: {
      fan::graphics::image_t texture = *(fan::graphics::image_t*)KeyTraverse.kd();
      if (!texture.iic()) {
        fan_opengl_call(glActiveTexture(GL_TEXTURE0));
        fan_opengl_call(glBindTexture(GL_TEXTURE_2D, loco.image_get_handle(texture)));
      }
      break;
    }

    case fan::graphics::Key_e::viewport:
      viewport = *(fan::graphics::viewport_t*)KeyTraverse.kd();
      break;

    case fan::graphics::Key_e::camera:
      camera = *(loco_t::camera_t*)KeyTraverse.kd();

      //if (using_culling) {
      //  fan::graphics::culling::cull_camera(
      //    shapes.visibility,
      //    fan::graphics::g_shapes->shaper,
      //    camera.iic() ? loco.orthographic_render_view.camera : camera/*,
      //    loco.window.get_size()*/
      //  );
      //}
      break;

    case fan::graphics::Key_e::ShapeType:
      prev_st = *(fan::graphics::shaper_t::ShapeTypeIndex_t*)KeyTraverse.kd();
      if (prev_st == fan::graphics::shapes::shape_type_t::light_end) {
        continue;
      }
      break;
    case fan::graphics::Key_e::light:
      if (!((loco.context.gl.opengl.major > 3) || (loco.context.gl.opengl.major == 3 && loco.context.gl.opengl.minor >= 3))) {
        break;
      }
      if (light_buffer_enabled == false) {
      #if defined(LOCO_FRAMEBUFFER)
        loco.context.gl.set_depth_test(false);
        fan_opengl_call(glEnable(GL_BLEND));
        fan_opengl_call(glBlendFunc(GL_ONE, GL_ONE));
        unsigned int attachments[sizeof(loco.gl.color_buffers) / sizeof(loco.gl.color_buffers[0])];

        for (uint8_t i = 0; i < std::size(loco.gl.color_buffers); ++i) {
          attachments[i] = GL_COLOR_ATTACHMENT0 + i;
        }

        fan_opengl_call(glDrawBuffers(std::size(attachments), attachments));
        light_buffer_enabled = true;
      #endif
      }
      break;
    case fan::graphics::Key_e::light_end:
      if (!((loco.context.gl.opengl.major > 3) || (loco.context.gl.opengl.major == 3 && loco.context.gl.opengl.minor >= 3))) {
        break;
      }
      if (light_buffer_enabled) {
      #if defined(LOCO_FRAMEBUFFER)
        loco.context.gl.set_depth_test(true);
        unsigned int attachments[sizeof(loco.gl.color_buffers) / sizeof(color_buffers[0])];

        for (uint8_t i = 0; i < std::size(color_buffers); ++i) {
          attachments[i] = GL_COLOR_ATTACHMENT0 + i;
        }

        fan_opengl_call(glDrawBuffers(1, attachments));
        light_buffer_enabled = false;
      #endif
        continue;
      }
      break;
    case fan::graphics::Key_e::draw_mode:
      draw_mode = fan::graphics::get_draw_mode(*(uint8_t*)KeyTraverse.kd());
      break;

    case fan::graphics::Key_e::vertex_count:
      vertex_count = *(uint32_t*)KeyTraverse.kd();
      break;
    case fan::graphics::Key_e::shadow:
      GLenum blend_src_factor = GL_DST_COLOR;
      GLenum blend_dst_factor = GL_ONE_MINUS_SRC_ALPHA;
      fan_opengl_call(glEnable(GL_BLEND));
      fan_opengl_call(glBlendFunc(blend_src_factor, blend_dst_factor));
      break;
    }

    if (!KeyTraverse.isbm) {
      continue;
    }

    if (!visible) {
      continue;
    }

    auto shape_type = prev_st;

    if (loco.force_line_draw &&
      shape_type != fan::graphics::shapes::shape_type_t::line &&
      shape_type != fan::graphics::shapes::shape_type_t::grid) {
      continue;
    }

    if (shape_type == fan::graphics::shape_type_t::vfi) {
      continue;
    }

    auto current_bmid = KeyTraverse.bmid();

   /* static int frame = 0;
    if (frame++ < 10) {
      fan::print("Drawing bmid:", current_bmid.gint(), "shape_type:", shape_type);
    }*/

    auto shader = fan::graphics::g_shapes->shaper.GetShader(shape_type);
    loco.shader_use(shader);

    if (!camera.iic()) {
      loco.shader_set_camera(shader, camera);
    }
    else {
      loco.shader_set_camera(shader, loco.orthographic_render_view.camera);
    }

    if (!viewport.iic()) {
      auto v = loco.viewport_get(viewport);
      loco.viewport_set(v.position, v.size);
    }

    loco.shader_set_value(shader, "_t00", 0);
    if ((loco.context.gl.opengl.major > 3) ||
      (loco.context.gl.opengl.major == 3 && loco.context.gl.opengl.minor >= 3)) {
      loco.shader_set_value(shader, "_t01", 1);
    }

    auto m_vao = fan::graphics::g_shapes->shaper.GetVAO(shape_type);
    auto m_vbo = fan::graphics::g_shapes->shaper.GetVBO(shape_type);

    m_vao.bind(loco.context.gl);
    m_vbo.bind(loco.context.gl);

    if (shape_type == fan::graphics::shapes::shape_type_t::universal_image_renderer) {
      auto& ri = *(fan::graphics::shapes::universal_image_renderer_t::ri_t*)
        fan::graphics::g_shapes->shaper.GetData(shape_type, 
          fan::graphics::g_shapes->shaper.BlockManager[current_bmid].FirstBlockNR, 0);

      if (ri.images_rest[0].iic() == false && ri.images_rest[0] != loco.default_texture) {
        fan_opengl_call(glActiveTexture(GL_TEXTURE0 + 1));
        fan_opengl_call(glBindTexture(GL_TEXTURE_2D, loco.image_get_handle(ri.images_rest[0])));
        loco.shader_set_value(shader, "_t01", 1);
      }
      if (ri.images_rest[1].iic() == false && ri.images_rest[1] != loco.default_texture) {
        fan_opengl_call(glActiveTexture(GL_TEXTURE0 + 2));
        fan_opengl_call(glBindTexture(GL_TEXTURE_2D, loco.image_get_handle(ri.images_rest[1])));
        loco.shader_set_value(shader, "_t02", 2);
      }
      if (ri.images_rest[2].iic() == false && ri.images_rest[2] != loco.default_texture) {
        fan_opengl_call(glActiveTexture(GL_TEXTURE0 + 3));
        fan_opengl_call(glBindTexture(GL_TEXTURE_2D, loco.image_get_handle(ri.images_rest[2])));
        loco.shader_set_value(shader, "_t03", 3);
      }
    }
    else if (shape_type == fan::graphics::shapes::shape_type_t::sprite ||
      shape_type == fan::graphics::shapes::shape_type_t::unlit_sprite ||
      shape_type == fan::graphics::shapes::shape_type_t::shader_shape) {

      auto& ri = *(fan::graphics::shapes::sprite_t::ri_t*)
        fan::graphics::g_shapes->shaper.GetData(shape_type,
          fan::graphics::g_shapes->shaper.BlockManager[current_bmid].FirstBlockNR, 0);

      loco.shader_set_value(shader, "has_normal_map", int(!ri.images[0].iic() && ri.images[0] != loco.default_texture));
      loco.shader_set_value(shader, "has_specular_map", int(!ri.images[1].iic() && ri.images[1] != loco.default_texture));
      loco.shader_set_value(shader, "has_occlusion_map", int(!ri.images[2].iic() && ri.images[2] != loco.default_texture));

      for (std::size_t i = 2; i < std::size(ri.images) + 2; ++i) {
        if (ri.images[i - 2].iic() == false) {
          loco.shader_set_value(shader, "_t0" + std::to_string(i), i);
          fan_opengl_call(glActiveTexture(GL_TEXTURE0 + i));
          fan_opengl_call(glBindTexture(GL_TEXTURE_2D, loco.image_get_handle(ri.images[i - 2])));
        }
      }
    }
    if (shape_type != fan::graphics::shapes::shape_type_t::light) {
    /*  if (shape_type == fan::graphics::shapes::shape_type_t::sprite || 
        shape_type == fan::graphics::shapes::shape_type_t::unlit_sprite) {
        if ((loco.context.gl.opengl.major > 3) || 
          (loco.context.gl.opengl.major == 3 && loco.context.gl.opengl.minor >= 3)) {
          fan_opengl_call(glActiveTexture(GL_TEXTURE1));
          fan_opengl_call(glBindTexture(GL_TEXTURE_2D, loco.image_get_handle(color_buffers[1])));
        }
      }*/

      auto c = loco.camera_get(camera);

      loco.shader_set_value(shader, "matrix_size",
        fan::vec2(c.coordinates.right - c.coordinates.left, 
          c.coordinates.bottom - c.coordinates.top).abs() / c.zoom);
      loco.shader_set_value(shader, "viewport",
        fan::vec4(loco.viewport_get_position(viewport),
          loco.viewport_get_size(viewport)));
      loco.shader_set_value(shader, "window_size",
        fan::vec2(loco.window.get_size()));
      loco.shader_set_value(shader, "camera_position", c.position);
      loco.shader_set_value(shader, "_time",
        f32_t((fan::time::now() - loco.start_time.m_time) / 1e+9));
      loco.shader_set_value(shader, "mouse_position",
        loco.get_mouse_position());
      loco.shader_set_value(shader, "camera_zoom",
        loco.camera_get_zoom(camera));
    }

    loco.shader_set_value(shader, fan::graphics::lighting_t::ambient_name, 
      gloco()->lighting.ambient);


    auto& shaper = fan::graphics::g_shapes->shaper;
    auto& st = shaper.GetShapeTypes(shape_type);
    auto& shape_gl = st.renderer.gl;

    switch (shape_type) {
    #if defined(FAN_3D)
    case fan::graphics::shapes::shape_type_t::rectangle3d:
      loco.context.gl.set_depth_test(false);
      break;
    case fan::graphics::shapes::shape_type_t::line3d:
      loco.context.gl.set_depth_test(false);
      break;
    #endif
    case fan::graphics::shapes::shape_type_t::particles:
    {
      fan::graphics::shaper_t::BlockTraverse_t BlockTraverse;
      BlockTraverse.Init(shaper, current_bmid);

      do {
        fan::graphics::shapes::particles_t::ri_t* pri =
          (fan::graphics::shapes::particles_t::ri_t*)BlockTraverse.GetData(shaper);

        for (int i = 0; i < BlockTraverse.GetAmount(shaper); ++i) {
          auto& ri = pri[i];
          loco.shader_set_value(shader, "loop", ri.loop);
          loco.shader_set_value(shader, "loop_enabled_time", ri.loop_enabled_time);
          loco.shader_set_value(shader, "loop_disabled_time", ri.loop_disabled_time);
          loco.shader_set_value(shader, "time", (f32_t)((fan::time::now()) / 1e+9));
          loco.shader_set_value(shader, "vertex_count", 6);
          loco.shader_set_value(shader, "count", ri.count);
          loco.shader_set_value(shader, "alive_time", (f32_t)(ri.alive_time));
          loco.shader_set_value(shader, "respawn_time", (f32_t)(ri.respawn_time));
          loco.shader_set_value(shader, "position", *(fan::vec2*)&ri.position);

          loco.shader_set_value(shader, "start_size", ri.start_size);
          loco.shader_set_value(shader, "end_size", ri.end_size);

          loco.shader_set_value(shader, "start_velocity", ri.start_velocity);
          loco.shader_set_value(shader, "end_velocity", ri.end_velocity);

          loco.shader_set_value(shader, "start_angle_velocity", ri.start_angle_velocity);
          loco.shader_set_value(shader, "end_angle_velocity", ri.end_angle_velocity);

          loco.shader_set_value(shader, "begin_angle", ri.begin_angle);
          loco.shader_set_value(shader, "end_angle", ri.end_angle);
          loco.shader_set_value(shader, "angle", ri.angle);

          loco.shader_set_value(shader, "begin_color", ri.begin_color);
          loco.shader_set_value(shader, "end_color", ri.end_color);

          loco.shader_set_value(shader, "spawn_spacing", ri.spawn_spacing);
          loco.shader_set_value(shader, "expansion_power", ri.expansion_power);

          loco.shader_set_value(shader, "start_spread", ri.start_spread);
          loco.shader_set_value(shader, "end_spread", ri.end_spread);

          loco.shader_set_value(shader, "jitter_start", ri.jitter_start);
          loco.shader_set_value(shader, "jitter_end", ri.jitter_end);
          loco.shader_set_value(shader, "jitter_speed", ri.jitter_speed);

          loco.shader_set_value(shader, "shape", ri.shape);

          fan_opengl_call(glDrawArrays(GL_TRIANGLES, 0, 6 * ri.count));
        }

      } while (BlockTraverse.Loop(shaper));

      continue;  // skip normal drawing
    }
    case fan::graphics::shapes::shape_type_t::polygon:
    {
      fan::graphics::shaper_t::BlockTraverse_t BlockTraverse;
      BlockTraverse.Init(shaper, current_bmid);

      do {
        fan::graphics::shapes::polygon_t::ri_t* pri =
          (fan::graphics::shapes::polygon_t::ri_t*)BlockTraverse.GetData(shaper);

        for (int i = 0; i < BlockTraverse.GetAmount(shaper); ++i) {
          auto& ri = pri[i];
          ri.vao.bind(loco.context.gl);
          ri.vbo.bind(loco.context.gl);
          fan_opengl_call(glDrawArrays(draw_mode, 0,
            ri.buffer_size / sizeof(fan::graphics::polygon_vertex_t)));
        }
      } while (BlockTraverse.Loop(shaper));

      continue;  // skip normal drawing
    }
    }

    fan::graphics::shaper_t::BlockTraverse_t BlockTraverse;
    BlockTraverse.Init(shaper, current_bmid);

    do {
      if (((loco.context.gl.opengl.major > 4) ||
        (loco.context.gl.opengl.major == 4 && loco.context.gl.opengl.minor >= 2)) &&
        shape_gl.instanced) {
        fan_opengl_call(glDrawArraysInstancedBaseInstance(
          draw_mode,
          0,
          vertex_count,
          BlockTraverse.GetAmount(shaper),
          BlockTraverse.GetRenderDataOffset(shaper) / shaper.GetRenderDataSize(shape_type)
        ));
      }
      else if (((loco.context.gl.opengl.major > 3) ||
        (loco.context.gl.opengl.major == 3 && loco.context.gl.opengl.minor >= 3)) &&
        shape_gl.instanced) {

        fan_opengl_call(glDrawArraysInstanced(
          draw_mode,
          0,
          vertex_count,
          BlockTraverse.GetAmount(shaper)
        ));
      }
      else {
        fan_opengl_call(glDrawArrays(
          draw_mode,
          (!!!(loco.context.gl.opengl.major == 2 && loco.context.gl.opengl.minor == 1)) *
          (BlockTraverse.GetRenderDataOffset(shaper) / shaper.GetRenderDataSize(shape_type)) *
          shape_gl.vertex_count,
          vertex_count * BlockTraverse.GetAmount(shaper)
        ));
      }
    } while (BlockTraverse.Loop(shaper));
  }

#endif

  {
  #if defined(LOCO_FRAMEBUFFER)

    if ((loco.context.gl.opengl.major > 3) || (loco.context.gl.opengl.major == 3 && loco.context.gl.opengl.minor >= 3)) {
      loco.gl.m_framebuffer.unbind(loco.context.gl);

      if (loco.window.renderer == fan::window_t::renderer_t::opengl && loco.open_props.enable_bloom) {
        loco.gl.blur.draw(&loco.gl.color_buffers[0]);
      }

      //blur[1].draw(&color_buffers[3]);

      fan_opengl_call(glClearColor(loco.clear_color.r, loco.clear_color.g, loco.clear_color.b, loco.clear_color.a));
      fan_opengl_call(glClear(GL_COLOR_BUFFER_BIT));
      fan::vec2 window_size = loco.window.get_size();
      loco.viewport_set(0, window_size);

      loco.shader_set_value(loco.gl.m_fbo_final_shader, "_t00", 0);
      loco.shader_set_value(loco.gl.m_fbo_final_shader, "_t01", 1);
      loco.shader_set_value(loco.gl.m_fbo_final_shader, "framebuffer_alpha", loco.clear_color.a);
      loco.shader_set_value(loco.gl.m_fbo_final_shader, "enable_bloom", loco.open_props.enable_bloom);

      loco.shader_set_value(loco.gl.m_fbo_final_shader, "window_size", window_size);


      if (loco.window.renderer == fan::window_t::renderer_t::opengl) {
        fan_opengl_call(glActiveTexture(GL_TEXTURE0));
        loco.image_bind(loco.gl.color_buffers[0]);
      }

      if (loco.window.renderer == fan::window_t::renderer_t::opengl) {
        if (loco.open_props.enable_bloom) {
          fan_opengl_call(glActiveTexture(GL_TEXTURE1));
          loco.image_bind(loco.gl.blur.mips.front().image);
        }

        render_final_fb();
      }
    }
    #endif
  }
}

void begin_process_frame() {
  fan_opengl_call(glViewport(
    0,
    0,
    loco.window.get_size().x,
    loco.window.get_size().y
  ));

#if defined(LOCO_FRAMEBUFFER)
  if ((loco.context.gl.opengl.major > 3) ||
    (loco.context.gl.opengl.major == 3 &&
      loco.context.gl.opengl.minor >= 3)) {

    loco.gl.m_framebuffer.bind(loco.context.gl);

    fan_opengl_call(glClearColor(0, 0, 0, loco.clear_color.a));

    for (std::size_t i = 0; i < std::size(loco.gl.color_buffers); ++i) {
      fan_opengl_call(glActiveTexture(GL_TEXTURE0 + i));
      loco.image_bind(loco.gl.color_buffers[i]);

      uint32_t attachment =
        (uint32_t)std::size(loco.gl.color_buffers) - 1 - i;

      fan_opengl_call(glDrawBuffer(GL_COLOR_ATTACHMENT0 + attachment));

      fan_opengl_call(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
    }
  }
#else
  fan_opengl_call(glClearColor(
    loco.clear_color.r,
    loco.clear_color.g,
    loco.clear_color.b,
    loco.clear_color.a
  ));
  fan_opengl_call(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
#endif
}

void initialize_fb_vaos() {
  static constexpr f32_t quad_vertices[] = {
     -1.0f, 1.0f, 0, 0.0f, 1.0f,
     -1.0f, -1.0f, 0, 0.0f, 0.0f,
     1.0f, 1.0f, 0, 1.0f, 1.0f,
     1.0f, -1.0f, 0, 1.0f, 0.0f,
  };
  fan_opengl_call(glGenVertexArrays(1, &fb_vao));
  fan_opengl_call(glGenBuffers(1, &fb_vbo));
  fan_opengl_call(glBindVertexArray(fb_vao));
  fan_opengl_call(glBindBuffer(GL_ARRAY_BUFFER, fb_vbo));
  fan_opengl_call(glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), &quad_vertices, GL_STATIC_DRAW));
  fan_opengl_call(glEnableVertexAttribArray(0));
  fan_opengl_call(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(f32_t), (void*)0));
  fan_opengl_call(glEnableVertexAttribArray(1));
  fan_opengl_call(glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(f32_t), (void*)(3 * sizeof(f32_t))));
}

void render_final_fb() {
  fan_opengl_call(glBindVertexArray(loco.gl.fb_vao));
  fan_opengl_call(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
  fan_opengl_call(glBindVertexArray(0));
}

void init() {
#if defined(LOCO_FRAMEBUFFER)
  init_framebuffer();

  loco.gl.m_fbo_final_shader = loco.shader_create();

  loco.shader_set_vertex(
    loco.gl.m_fbo_final_shader,
    fan::graphics::read_shader("shaders/opengl/2D/effects/loco_fbo.vs")
  );
  loco.shader_set_fragment(
    loco.gl.m_fbo_final_shader,
    fan::graphics::read_shader("shaders/opengl/2D/effects/loco_fbo.fs")
  );
  loco.shader_compile(loco.gl.m_fbo_final_shader);
#endif
}

#undef loco