struct rectangle_t {

  struct instance_t {
    fan::vec3 position = 0;
  private:
    f32_t pad[1];
  public:
    fan::vec2 size = 0;
    fan::vec2 rotation_point = 0;
    fan::color color = fan::colors::white;
    fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
    f32_t angle = 0;
  };

  #define hardcode0_t loco_t::matrices_list_NodeReference_t
  #define hardcode0_n matrices
  #define hardcode1_t fan::graphics::viewport_list_NodeReference_t
  #define hardcode1_n viewport
  #include _FAN_PATH(graphics/opengl/2D/objects/hardcode_open.h)

  struct instance_properties_t {
    struct key_t : parsed_masterpiece_t {}key;
    expand_get_functions
  };
      
  struct properties_t : instance_t, instance_properties_t {
    properties_t() = default;
    properties_t(const instance_t& i) : instance_t(i) {}
    properties_t(const instance_properties_t& p) : instance_properties_t(p) {}
  };
  
  void push_back(fan::graphics::cid_t* cid, properties_t& p) {
    #if defined(loco_vulkan)
      auto loco = get_loco();
      auto& matrices = loco->matrices_list[p.get_matrices()];
      if (matrices.matrices_index.rectangle == (decltype(matrices.matrices_index.rectangle))-1) {
        matrices.matrices_index.rectangle = m_matrices_index++;
        m_shader.set_matrices(loco, matrices.matrices_id, &loco->m_write_queue, matrices.matrices_index.rectangle);
      }
    #endif

    sb_push_back(cid, p);
  }
  void erase(fan::graphics::cid_t* cid) {
    sb_erase(cid);
  }

  void draw() {
    sb_draw();
  }

  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(instance_t) / 4));
  #if defined(loco_opengl)
    #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle.vs)
    #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle.fs)
  #elif defined(loco_vulkan)
    #define vulkan_buffer_count 2
    #define sb_shader_vertex_path _FAN_PATH_QUOTE(graphics/glsl/vulkan/2D/objects/rectangle.vert.spv)
    #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/vulkan/2D/objects/rectangle.frag.spv)
  #endif
  #include _FAN_PATH(graphics/shape_builder.h)

  rectangle_t() {
    #define vk_sb_ssbo
    #define vk_sb_vp
    #include _FAN_PATH(graphics/shape_open_settings.h)

    fan::vulkan::pipeline_t::properties_t p;


    auto context = get_loco()->get_context();

    render_fullscreen_shader.open(context);
    render_fullscreen_shader.set_vertex(context, _FAN_PATH_QUOTE(graphics/glsl/vulkan/2D/objects/fullscreen.vert.spv));
    render_fullscreen_shader.set_fragment(context, _FAN_PATH_QUOTE(graphics/glsl/vulkan/2D/objects/rectangle.frag.spv));
    p.descriptor_layout_count = 1;
    p.descriptor_layout = &m_descriptor.m_layout;
    p.shader = &render_fullscreen_shader;
    p.push_constants_size = p.push_constants_size = sizeof(loco_t::push_constants_t);
    p.subpass = 1;
    context->render_fullscreen_pl.open(context, p);
  }
  ~rectangle_t() {
    sb_close();
  }

  void set_matrices(fan::graphics::cid_t* cid, loco_t::matrices_list_NodeReference_t n) {
    sb_set_key<instance_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
  }

  void set_viewport(fan::graphics::cid_t* cid, fan::graphics::viewport_list_NodeReference_t n) {
    sb_set_key<instance_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
  }
  
  #if defined(loco_vulkan)
    uint32_t m_matrices_index = 0;
  #endif

};

#include _FAN_PATH(graphics/opengl/2D/objects/hardcode_close.h)
#undef vulkan_buffer_count