struct particles_t {

  #define sb_vertex_count 6

  static constexpr typename loco_t::shape_type_t shape_type = loco_t::shape_type_t::mark;

  struct vi_t {
    fan::vec3 position = 0;
    fan::vec2 size = 100;
    fan::vec3 angle = 0;
    fan::color color = fan::colors::red;
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

    uint64_t begin_time;
    uint64_t alive_time = 1e+9;
    uint64_t respawn_time = 0;
    uint32_t count = 10;
    fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
    fan::vec2 position_velocity = 130;
    fan::vec3 angle_velocity = fan::vec3(0, 0, 0.1);
    f32_t begin_angle = 0;
    f32_t end_angle = fan::math::pi * 2;

    bool blending = true;
  };

  struct properties_t : vi_t, ri_t, context_key_t {
    using type_t = particles_t;

    loco_t::camera_t* camera = &gloco->default_camera->camera;
    loco_t::viewport_t* viewport = &gloco->default_camera->viewport;
    loco_t::image_t* image = &gloco->default_texture;
  };

  void push_back(loco_t::cid_nt_t& id, properties_t p) {
    get_key_value(loco_t::textureid_t<0>) = p.image;
    get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;
    p.begin_time = fan::time::clock::now();
    sb_push_back(id, p);
  }
  void erase(loco_t::cid_nt_t& id) {
    sb_erase(id);
  }

  int draw_instance = 0;

  void draw(const loco_t::redraw_key_t& redraw_key, loco_bdbt_NodeReference_t key_root) {
    if (redraw_key.blending) {
      m_current_shader = &m_blending_shader;
    }
    else {
      m_current_shader = &m_shader;
    }
    auto& context = gloco->get_context();
    draw_instance = 0;
    sb_draw1(key_root, fan::opengl::GL_TRIANGLES);
  }

  void custom_draw(auto* block_node, uint32_t draw_mode) {

    auto& context = gloco->get_context();
    
    context.set_depth_test(false);

    context.opengl.call(context.opengl.glEnable, fan::opengl::GL_BLEND);
    context.opengl.call(context.opengl.glBlendFunc, fan::opengl::GL_SRC_ALPHA, fan::opengl::GL_ONE_MINUS_SRC_ALPHA);

    uint32_t begin = 0;
    for (int i = 0; i < block_node->data.uniform_buffer.size(); ++i) {
                                                            // can be illegal
      auto& vi = ((vi_t*)block_node->data.uniform_buffer.buffer)[draw_instance];
      auto& ri = block_node->data.ri[draw_instance++];

      
      m_current_shader->set_float("time", (f64_t)(fan::time::clock::now() - ri.begin_time) / 1e+9);;
      m_current_shader->set_uint("vertex_count", sb_vertex_count);
      m_current_shader->set_uint("count", ri.count);
      m_current_shader->set_float("alive_time", (f32_t)ri.alive_time / 1e+9);
      m_current_shader->set_float("respawn_time", (f32_t)ri.respawn_time / 1e+9);
      m_current_shader->set_vec2("position", *(fan::vec2*)&vi.position);
      m_current_shader->set_vec2("size", vi.size);
      m_current_shader->set_vec2("position_velocity", ri.position_velocity);
      m_current_shader->set_vec3("angle_velocity", ri.angle_velocity);
      m_current_shader->set_vec3("rotation_vector", ri.rotation_vector);
      m_current_shader->set_float("begin_angle", ri.begin_angle);
      m_current_shader->set_float("end_angle", ri.end_angle);
      m_current_shader->set_vec4("color", vi.color);

      block_node->data.uniform_buffer.draw(
        gloco->get_context(),
        begin * sb_vertex_count,
        ri.count * sb_vertex_count,
        draw_mode
      );
      begin += ri.count;
    }
    context.set_depth_test(true);
  }

  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(vi_t) / 4));
  #if defined(loco_opengl)
    #define sb_shader_vertex_path  _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/effects/particles.vs)
    #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/effects/particles.fs)
  #endif

  particles_t() {
    sb_open(sb_shader_vertex_path, sb_shader_fragment_path);
  }
  ~particles_t() {
    sb_close();
  }

  #include _FAN_PATH(graphics/shape_builder.h)

  void set_camera(loco_t::cid_nt_t& id, loco_t::camera_list_NodeReference_t n) {
    sb_set_context_key<decltype(n)>(id, n);
  }

  void set_viewport(loco_t::cid_nt_t& id, fan::graphics::viewport_list_NodeReference_t n) {
    sb_set_context_key<decltype(n)>(id, n);
  }

  properties_t get_properties(loco_t::cid_nt_t& id) {
    properties_t p = sb_get_properties(id);
    return p;
  }
};

#undef vulkan_buffer_count