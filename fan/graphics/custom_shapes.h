// add custom shapes to this file
#if defined(loco_rectangle)
struct line_grid_t {

  static constexpr typename loco_t::shape_type_t shape_type = loco_t::shape_type_t::custom;

  struct vi_t {
    loco_t::position3_t position = 0;
    f32_t pad;
    fan::vec2 size = 0;
    fan::vec2 grid_size;
    fan::vec2 rotation_point = 0;
    f32_t pad2[2];
    fan::color color = fan::colors::white;
    fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
    f32_t angle = 0;
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
    using type_t = rectangle_t;
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