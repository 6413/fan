struct line_t {

  struct vi_t {
    fan::color color;
    fan::vec3 src;
  private:
    f32_t pad;
  public:
    fan::vec3 dst;
  private:
    f32_t pad2[1];
  };

  struct bm_properties_t {
     using parsed_masterpiece_t = fan::masterpiece_t<
      loco_t::matrices_list_NodeReference_t,
      fan::graphics::viewport_list_NodeReference_t
    >;
    struct key_t : parsed_masterpiece_t {
    }key;
  };

  struct cid_t;

	struct ri_t : bm_properties_t {
		cid_t* cid;
	};

   #define make_key_value(type, name) \
    type& name = *key.get_value<decltype(key)::get_index_with_type<type>()>();

  struct properties_t : vi_t, ri_t {

    make_key_value(loco_t::matrices_list_NodeReference_t, matrices);
    make_key_value(fan::graphics::viewport_list_NodeReference_t, viewport);

    properties_t() = default;
    properties_t(const vi_t& i) : vi_t(i) {}
    properties_t(const ri_t& p) : ri_t(p) {}
  };

  #undef make_key_value
  
  void push_back(fan::graphics::cid_t* cid, properties_t& p) {
    
    #if defined(loco_vulkan)
      auto loco = get_loco();
      auto& matrices = loco->matrices_list[p.matrices];
      if (matrices.matrices_index.line == (decltype(matrices.matrices_index.line))-1) {
        matrices.matrices_index.line = m_matrices_index++;
        m_shader.set_matrices(loco, matrices.matrices_id, &loco->m_write_queue, matrices.matrices_index.line);
      }
    #endif

    sb_push_back(cid, p);
  }
  void erase(fan::graphics::cid_t* cid) {
    sb_erase(cid);
  }

  void draw() {
    sb_draw(
      #if defined (loco_opengl)
        fan::opengl::GL_LINES
      #endif
    );
  }

  static constexpr uint32_t max_instance_size = fan::min(256ull, 4096 / (sizeof(vi_t) / 4));

  #define sb_vertex_count 2

 #if defined(loco_opengl)
    #ifndef sb_shader_vertex_path
      #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/line.vs)
    #endif
    #ifndef sb_shader_fragment_path
      #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/line.fs)
    #endif
  #elif defined(loco_vulkan)
    #define vulkan_buffer_count 2
    #define sb_shader_vertex_path _FAN_PATH_QUOTE(graphics/glsl/vulkan/2D/objects/line.vert.spv)
    #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/vulkan/2D/objects/line.frag.spv)
  #endif

  #define vk_sb_ssbo
  #define vk_sb_vp
  #include _FAN_PATH(graphics/shape_builder.h)

  line_t() {
    sb_open();
  }
  ~line_t() {
    sb_close();
  }

  void set_line(fan::graphics::cid_t* cid, const fan::vec3& src, const fan::vec3& dst) {
    set(cid, &vi_t::src, src);
    set(cid, &vi_t::dst, dst);
  }

  #if defined(loco_vulkan)
    uint32_t m_matrices_index = 0;
  #endif
};