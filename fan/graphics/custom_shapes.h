// add custom shapes to this file

// TODO add vulkan support, basically just .vert, .frag files
#if defined(loco_opengl)
#if defined(loco_rectangle)
struct line_grid_t {

  static constexpr typename loco_t::shape_type_t shape_type = loco_t::shape_type_t::line_grid;

  struct vi_t {
    loco_t::position3_t position = 0;
    f32_t pad;
    fan::vec2 size = 0;
    fan::vec2 grid_size;
    fan::vec2 rotation_point = 0;
    f32_t pad2[2];
    fan::color color = fan::colors::white;
    fan::vec3 angle = 0;
    f32_t pad3;
  };

  struct context_key_t {
    using parsed_masterpiece_t = fan::masterpiece_t<
      loco_t::camera_list_NodeReference_t,
      fan::graphics::viewport_list_NodeReference_t
    >;
    struct key_t : parsed_masterpiece_t {}key;
  };

  struct cid_t;

  struct ri_t {
    bool blending = false;
  };

  struct properties_t : vi_t, ri_t, context_key_t {
    using type_t = line_grid_t;
    loco_t::camera_t* camera = &gloco->default_camera->camera;
    loco_t::viewport_t* viewport = &gloco->default_camera->viewport;
  };

  void push_back(loco_t::cid_nt_t& id, properties_t p) {
    get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;
    sb_push_back(id, p);
  }
  void erase(loco_t::cid_nt_t& id) {
    sb_erase(id);
  }

  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(vi_t) / 4));

  #define sb_shader_vertex_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/line_grid.vs)
  #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/line_grid.fs)

  line_grid_t() {
    sb_open(sb_shader_vertex_path, sb_shader_fragment_path);
  }
  ~line_grid_t() {
    sb_close();
  }

  void draw(const loco_t::redraw_key_t& redraw_key, loco_bdbt_NodeReference_t key_root) {
    if (redraw_key.blending) {
      m_current_shader = &m_blending_shader;
    }
    else {
      m_current_shader = &m_shader;
    }
    sb_draw(key_root);
  }


  #include _FAN_PATH(graphics/shape_builder.h)
};
line_grid_t line_grid;

#endif

#if defined(loco_grass_2d)
struct grass_2d_t {

  static constexpr typename loco_t::shape_type_t shape_type = loco_t::shape_type_t::grass_2d;

  struct vi_t {
    loco_t::position3_t position = 0;
    float pad;
    fan::vec2 size = 0;
    fan::vec2 pad2;
    fan::color color = fan::colors::white;
    fan::vec2 tc_position = 0;
    fan::vec2 tc_size = 1;
  };

  struct context_key_t {
    using parsed_masterpiece_t = fan::masterpiece_t<
      loco_t::textureid_t<0>,
      loco_t::camera_list_NodeReference_t,
      fan::graphics::viewport_list_NodeReference_t
    >;
    struct key_t : parsed_masterpiece_t {}key;
  };

  struct cid_t;

  struct ri_t {
    bool blending = false;
  };

  struct properties_t : vi_t, ri_t, context_key_t {
    using type_t = grass_2d_t;

    loco_t::image_t* image = &gloco->default_texture;
    loco_t::camera_t* camera = &gloco->default_camera->camera;
    fan::graphics::viewport_t* viewport = &gloco->default_camera->viewport;
    bool load_tp(loco_t::texturepack_t::ti_t* ti) {
      auto& im = *ti->image;
      image = &im;
      tc_position = ti->position / im.size;
      tc_size = ti->size / im.size;
      return 0;
    }
  };
  inline static  int ccc = 0;
  void push_back(loco_t::cid_nt_t& id, properties_t p) {

    get_key_value(loco_t::textureid_t<0>) = p.image;
    get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;

    sb_push_back(id, p);
  }
  void erase(loco_t::cid_nt_t& id) {
    sb_erase(id);
  }

  void draw(const loco_t::redraw_key_t& redraw_key, loco_bdbt_NodeReference_t key_root) {
    if (redraw_key.blending) {
      m_current_shader = &m_blending_shader;
    }
    else {
      m_current_shader = &m_shader;
    }
    sb_draw1(key_root, fan::opengl::GL_TRIANGLES);
  }

  // can be bigger with vulkan
  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(vi_t) / 4));
  #if defined(loco_opengl)
  #define sb_shader_vertex_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/grass_renderer.vs)
  #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/grass_renderer.fs)
  #endif

  grass_2d_t() {
    sb_open(sb_shader_vertex_path, sb_shader_fragment_path);
  }
  ~grass_2d_t() {
    sb_close();
  }
  
  // change in vs too
  static constexpr int planes_per_leaf = 8;
  #define sb_vertex_count 6 * planes_per_leaf

  void custom_draw(auto* block_node, uint32_t draw_mode) {
    for (int i = 0; i < block_node->data.uniform_buffer.size(); ++i) {
      block_node->data.uniform_buffer.bind_buffer_range(
        gloco->get_context(),
        block_node->data.uniform_buffer.size()
      );
      block_node->data.uniform_buffer.draw(
        gloco->get_context(),
        gloco->begin * sb_vertex_count,
        sb_vertex_count,
        draw_mode
      );
      gloco->begin += 1;
    }
  }

  #include _FAN_PATH(graphics/shape_builder.h)

  void set_image(loco_t::cid_nt_t& id, loco_t::textureid_t<0> n) {
    #if defined(loco_opengl)
    sb_set_context_key<loco_t::textureid_t<0>>(id, n);
    #endif
  }

  properties_t get_properties(loco_t::cid_nt_t& id) {
    properties_t p = sb_get_properties(id);
    return p;
  }
};

grass_2d_t grass_2d;
#endif

#if defined(loco_sprite)
struct shader_t {

  static constexpr typename loco_t::shape_type_t shape_type = loco_t::shape_type_t::shader;

  struct vi_t {
    loco_t::position3_t position = 0;
    f32_t parallax_factor = 0;
    fan::vec2 size = 0;
    fan::vec2 rotation_point = 0;
    fan::color color = fan::colors::white;
    fan::vec3 angle = fan::vec3(0);
    f32_t pad;
    fan::vec2 tc_position = 0;
    fan::vec2 tc_size = 1;
  };

  struct context_key_t {
    using parsed_masterpiece_t = fan::masterpiece_t<
      loco_t::shader_list_NodeReference_t,
      loco_t::textureid_t<0>,
      loco_t::camera_list_NodeReference_t,
      fan::graphics::viewport_list_NodeReference_t
    >;
    struct key_t : parsed_masterpiece_t {}key;
  };

  struct cid_t;

  struct ri_t {
    bool blending = false;
  };

  struct properties_t : vi_t, ri_t, context_key_t {
    using type_t = shader_t;

    loco_t::shader_t* shader = 0;
    loco_t::image_t* image = &gloco->default_texture;
    loco_t::camera_t* camera = &gloco->default_camera->camera;
    fan::graphics::viewport_t* viewport = &gloco->default_camera->viewport;

    bool load_tp(loco_t::texturepack_t::ti_t* ti) {
      auto& im = *ti->image;
      image = &im;
      tc_position = ti->position / im.size;
      tc_size = ti->size / im.size;
      return 0;
    }
  };

  void push_back(loco_t::cid_nt_t& id, properties_t p) {

    get_key_value(loco_t::shader_list_NodeReference_t) = p.shader;
    get_key_value(loco_t::textureid_t<0>) = p.image;
    get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;

    sb_push_back(id, p);
  }
  void erase(loco_t::cid_nt_t& id) {
    sb_erase(id);
  }

  void draw(const loco_t::redraw_key_t& redraw_key, loco_bdbt_NodeReference_t key_root) {
    if (redraw_key.blending) {
      m_current_shader = &m_blending_shader;
    }
    else {
      m_current_shader = &m_shader;
    }
    sb_draw(key_root);
  }

  // can be bigger with vulkan
  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(vi_t) / 4));
  #if defined(loco_opengl)
  #define sb_shader_vertex_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/sprite.vs)
  #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/sprite.fs)
  #endif

  shader_t() {
    sb_open(sb_shader_vertex_path, sb_shader_fragment_path);
  }
  ~shader_t() {
    sb_close();
  }

  #include _FAN_PATH(graphics/shape_builder.h)

  void set_image(loco_t::cid_nt_t& id, loco_t::textureid_t<0> n) {
    sb_set_context_key<loco_t::textureid_t<0>>(id, n);
  }

  properties_t get_properties(loco_t::cid_nt_t& id) {
    properties_t p = sb_get_properties(id);
    return p;
  }

  bool load_tp(loco_t::shape_t& id, loco_t::texturepack_t::ti_t* ti) {
    auto& im = *ti->image;

    set_image(id, &im);

    set(id, &vi_t::tc_position, ti->position / im.size);
    set(id, &vi_t::tc_size, ti->size / im.size);

    return 0;
  }
};

shader_t shader;

#endif

#if defined(loco_sprite) && defined(loco_light)

struct shader_light_t {

  static constexpr typename loco_t::shape_type_t shape_type = loco_t::shape_type_t::shader_light;

  struct vi_t {
    loco_t::position3_t position = 0;
    f32_t parallax_factor = 0;
    fan::vec2 size = 0;
    fan::vec2 rotation_point = 0;
    fan::color color = fan::colors::white;
    fan::vec3 angle = fan::vec3(0);
    f32_t pad;
    fan::vec2 tc_position = 0;
    fan::vec2 tc_size = 1;
  };

  struct context_key_t {
    using parsed_masterpiece_t = fan::masterpiece_t<
      loco_t::shader_list_NodeReference_t,
      loco_t::textureid_t<0>,
      loco_t::camera_list_NodeReference_t,
      fan::graphics::viewport_list_NodeReference_t
    >;
    struct key_t : parsed_masterpiece_t {}key;
  };

  struct cid_t;

  struct ri_t {
    bool blending = false;
  };

  struct properties_t : vi_t, ri_t, context_key_t {
    using type_t = shader_t;

    loco_t::shader_t* shader = 0;
    loco_t::image_t* image = &gloco->default_texture;
    loco_t::camera_t* camera = &gloco->default_camera->camera;
    fan::graphics::viewport_t* viewport = &gloco->default_camera->viewport;

    bool load_tp(loco_t::texturepack_t::ti_t* ti) {
      auto& im = *ti->image;
      image = &im;
      tc_position = ti->position / im.size;
      tc_size = ti->size / im.size;
      return 0;
    }
  };

  void push_back(loco_t::cid_nt_t& id, properties_t p) {

    get_key_value(loco_t::shader_list_NodeReference_t) = p.shader;
    get_key_value(loco_t::textureid_t<0>) = p.image;
    get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;

    sb_push_back(id, p);
  }
  void erase(loco_t::cid_nt_t& id) {
    sb_erase(id);
  }

  void draw() {
    gloco->get_context().set_depth_test(false);
    gloco->get_context().opengl.call(gloco->get_context().opengl.glEnable, fan::opengl::GL_BLEND);
    gloco->get_context().opengl.call(gloco->get_context().opengl.glBlendFunc, fan::opengl::GL_ONE, fan::opengl::GL_ONE);

    unsigned int attachments[sizeof(gloco->color_buffers) / sizeof(gloco->color_buffers[0])];

    for (uint8_t i = 0; i < std::size(gloco->color_buffers); ++i) {
      attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
    }

    gloco->get_context().opengl.call(gloco->get_context().opengl.glDrawBuffers, std::size(attachments), attachments);
    //
    sb_draw(key_root);
    //
    gloco->get_context().set_depth_test(true);

    for (uint8_t i = 0; i < std::size(gloco->color_buffers); ++i) {
      attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
    }

    gloco->get_context().opengl.call(gloco->get_context().opengl.glDrawBuffers, 1, attachments);
  }

  // can be bigger with vulkan
  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(vi_t) / 4));
  #if defined(loco_opengl)
  #define sb_shader_vertex_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/light.vs)
  #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/light.fs)
  #endif

  shader_light_t() {
    sb_open(sb_shader_vertex_path, sb_shader_fragment_path);

    m_current_shader = &m_blending_shader;
    gloco->m_draw_queue_light.push_back([&] {
      draw();
    });
  }
  ~shader_light_t() {
    sb_close();
  }
  #define sb_has_own_key_root 1
  #define sb_ignore_3_key 1
  #include _FAN_PATH(graphics/shape_builder.h)

  void set_image(loco_t::cid_nt_t& id, loco_t::textureid_t<0> n) {
    sb_set_context_key<loco_t::textureid_t<0>>(id, n);
  }

  properties_t get_properties(loco_t::cid_nt_t& id) {
    properties_t p = sb_get_properties(id);
    return p;
  }

  bool load_tp(loco_t::shape_t& id, loco_t::texturepack_t::ti_t* ti) {
    auto& im = *ti->image;

    set_image(id, &im);

    set(id, &vi_t::tc_position, ti->position / im.size);
    set(id, &vi_t::tc_size, ti->size / im.size);

    return 0;
  }
};

shader_light_t shader_light;
#endif

#endif