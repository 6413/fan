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

  static constexpr uint32_t max_instance_size = std::min(256ull, 4096 / (sizeof(instance_t) / 4));

  //struct rectangle_block_properties_t {
  //  fan::opengl::matrices_list_NodeReference_t matrices;
  //  fan::opengl::viewport_list_NodeReference_t viewport;
  //};

  typedef fan::masterpiece_t<
    fan::opengl::matrices_list_NodeReference_t,
    fan::opengl::viewport_list_NodeReference_t
  >
  block_properties_t;
      
  struct properties_t : instance_t {
    union {
      struct {
        //// sb block properties contents come here
        //block_properties_t.iterate_element_types([] {
        //  .get_type() nametable(.get_type())
        //});
        fan::opengl::matrices_list_NodeReference_t matrices;
        fan::opengl::viewport_list_NodeReference_t viewport;
      };

      block_properties_t block_properties;
    };
  };

  void push_back(loco_t* loco, fan::opengl::cid_t* cid, properties_t& p) {
    sb_push_back(loco, cid, p);
  }

  void draw(loco_t* loco) {
    m_shader.use(loco->get_context());
    loco_bdbt_Key_t<8> k0;
    decltype(k0)::Traverse_t kt0;
    kt0.init(root);
    fan::opengl::matrices_list_NodeReference_t o0;
    while (kt0.Traverse(&loco->bdbt, &o0)) {
      auto node = fan::opengl::matrices_list_GetNodeByReference(&loco->get_context()->matrices_list, o0);
      m_shader.set_matrices(loco->get_context(), node->data.matrices_id);
      loco_bdbt_Key_t<8> k1;
      decltype(k1)::Traverse_t kt1;
      kt1.init(kt0.Output);
      fan::opengl::viewport_list_NodeReference_t o1;
      while (kt1.Traverse(&loco->bdbt, &o1)) {
        auto node = fan::opengl::viewport_list_GetNodeByReference(&loco->get_context()->viewport_list, o1);
        node->data.viewport_id->set_viewport(
          loco->get_context(),
          node->data.viewport_id->get_viewport_position(),
          node->data.viewport_id->get_viewport_size()
        );
        auto bmn = shape_bm_GetNodeByReference(&bm_list, *(shape_bm_NodeReference_t*)&kt1.Output);
        auto bnr = bmn->data.first_block;

        while(1) {
          auto node = bll_block_GetNodeByReference(&blocks, bnr);

          node->data.block.uniform_buffer.bind_buffer_range(loco->get_context(), node->data.block.uniform_buffer.size());

          node->data.block.uniform_buffer.draw(
            loco->get_context(),
            0 * 6,
            node->data.block.uniform_buffer.size() * 6
          );
          if (bll_block_IsNodeReferenceEqual(bnr, bmn->data.last_block)) {
            break;
          }
          bnr = node->NextNodeReference;
        }
      }
    }
  }

  struct block_t;

  #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle.vs)
  #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle.fs)
  #include _FAN_PATH(graphics/opengl/2D/objects/shape_builder.h)

  void open(loco_t* loco) {
    sb_open(loco);
  }
  void close(loco_t* loco) {
    sb_close(loco);
  }
};