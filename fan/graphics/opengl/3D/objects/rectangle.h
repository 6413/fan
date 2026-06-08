struct rectangle_3d_t {

  static constexpr typename loco_t::shape_type_t shape_type = loco_t::shape_type_t::rectangle;

  struct vi_t {
    loco_t::position3_t position = 0;
    f32_t pad[1];
    fan::vec3 size = 0;
    f32_t pad3;
    fan::color color = fan::colors::white;
    fan::vec3 angle = 0;
    f32_t pad2;
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
    using type_t = rectangle_3d_t;
    loco_t::camera_t* camera = &gloco()->default_camera_3d->camera;
    fan::graphics::viewport_t* viewport = &gloco()->default_camera_3d->viewport;
  };

  void push_back(loco_t::cid_nt_t& id, properties_t p) {
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

  static constexpr uint32_t max_instance_size = std::min(256, 4096 / (sizeof(vi_t) / 4));

  #define sb_shader_vertex_path _FAN_PATH_QUOTE(graphics/glsl/opengl/3D/objects/rectangle.vs)
  #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/opengl/3D/objects/rectangle.fs)

  rectangle_3d_t() {
    sb_open(sb_shader_vertex_path, sb_shader_fragment_path);
  }
  ~rectangle_3d_t() {
    sb_close();
  }

  #define sb_vertex_count 36
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